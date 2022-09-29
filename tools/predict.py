from pathlib import Path
from typing import Dict, List
from tower_equipment_segmentation.dataset_converter.dataset_converter import DatasetConverter
from tower_equipment_segmentation.dvo.known_class_labels import KnownClassLabels
from tower_equipment_segmentation.util.logging_util import setup_logger

from tower_equipment_segmentation.util.dataset_util import \
    get_site_ref_ids_in_dataset_dir, \
    get_pcd_csv_file_path_for_site_ref_id, \
    get_softgroup_pcd_pth_file_path_for_site_ref_id_and_area_idx

from tower_equipment_segmentation.util.dir_util import create_dir
from tower_equipment_segmentation.dvo.point_cloud_asset_segmentation import PointCloudAssetSegmentation

import argparse, os, torch, yaml
import os.path as osp
import multiprocessing as mp
import numpy as np

from torch.nn.parallel import DistributedDataParallel
from munch import Munch
from softgroup.data import build_dataloader, build_dataset
from softgroup.evaluation import (PanopticEval, ScanNetEval, evaluate_offset_mae,
                                  evaluate_semantic_acc, evaluate_semantic_miou)
from softgroup.model import SoftGroup
from softgroup.util import (collect_results_cpu, get_dist_info, get_root_logger, init_dist,
                            is_main_process, load_checkpoint, rle_decode)
from tqdm import tqdm

logger = setup_logger(__name__)

label_to_idx_map: Dict[str, int] = {KnownClassLabels.antenna: 1, KnownClassLabels.transceiver_junction: 2,
                                    KnownClassLabels.head_frame_mount: 3, KnownClassLabels.shelter: 4,
                                    KnownClassLabels.aviation_light: 0, KnownClassLabels.background: 0}


def convert_to_softgroup(src_dataset_dir: Path, dst_dataset_dir: Path, area_idx: int,
                         class_label_to_class_idx_map: Dict[str, int]):
    ds_conv = DatasetConverter()
    ds_conv.convert_tower_asset_segmentation_dataset_to_softgroup_format(
        src_dataset_dir=src_dataset_dir,
        dst_dataset_dir=dst_dataset_dir,
        class_label_to_class_idx_map=class_label_to_class_idx_map,
        area_index=area_idx)


def _get_instance_scores_and_preds(instances: List, pcd_size: int):
    class_labels = [KnownClassLabels.antenna, KnownClassLabels.transceiver_junction, KnownClassLabels.head_frame_mount,
                    KnownClassLabels.shelter, KnownClassLabels.background]

    def _convert_to_label(id):
        return class_labels[id]


    keep = []
    masks = [rle_decode(x['pred_mask']) for x in instances]
    confs = np.array([x['conf'] for x in instances])
    mask_areas = np.array([m.sum() for m in masks])

    order = np.argsort(confs)[::-1]
    while order.size > 0:
        i = order.item(0)
        keep.append(i)

        intersection = [np.sum((masks[i] == 1) & (masks[j] == 1)) for j in order[1:]]
        union = [np.sum((masks[i] == 1) | (masks[j] == 1)) for j in order[1:]]
        min_area = [min(mask_areas[i], mask_areas[j]) for j in order[1:]]
        iou = np.array(intersection)/np.array(union)

        inds = np.where(iou < 0.8)[0]
        order = order[inds+1]

    instances = [instances[i] for i in keep if instances[i]['conf'] > 0.09]
    instance_class_labels = np.ones((pcd_size, ), dtype=np.int) * -1
    predicted_instance_ids = np.ones((pcd_size,)) * -1
    instance_confs = np.zeros((pcd_size, len(class_labels)))
    class_labels_to_instance_conf_scores = dict()

    instances = sorted(instances, key=lambda x: x['conf'])
    for i, inst in enumerate(instances):
        label_id = inst['label_id']
        logits = inst['all_score_pred']
        mask = rle_decode(inst['pred_mask'])

        instance_class_labels[mask == 1] = (label_id - 1)
        predicted_instance_ids[mask == 1] = i
        instance_confs[mask == 1, :] = logits
    predicted_instance_class_labels = list(map(_convert_to_label, instance_class_labels))
    for i, label in enumerate(class_labels):
        class_labels_to_instance_conf_scores[label] = instance_confs[:, i].reshape(-1, )
    return predicted_instance_class_labels, predicted_instance_ids, class_labels_to_instance_conf_scores


def _get_semantic_scores(semantic_scores: List):
    class_labels = [KnownClassLabels.background, KnownClassLabels.antenna, KnownClassLabels.transceiver_junction,
                    KnownClassLabels.head_frame_mount, KnownClassLabels.shelter]
    semantic_scores_dict = dict()
    for i, label in enumerate(class_labels):
        semantic_scores_dict[label] = semantic_scores[:, i].reshape(-1, )
    return semantic_scores_dict


def _save_softgroup_logits_and_instances_in_pcd_csv(dataset_dir: Path, result: Dict, overwrite_existing_predictions):
    site_ref_id = result['site_ref_id']
    predicted_instances = result['pred_instances']
    predicted_semantic_scores = result['semantic_scores']

    src_pcd_csv_fp = get_pcd_csv_file_path_for_site_ref_id(dataset_dir, site_ref_id)
    pcd_asset_seg = PointCloudAssetSegmentation.from_csv_file(src_pcd_csv_fp)
    pcd_size = pcd_asset_seg.num_points

    predicted_instance_class_labels, predicted_instance_ids, class_labels_to_instance_conf_scores \
        = _get_instance_scores_and_preds(predicted_instances, pcd_size)
    semantic_scores = _get_semantic_scores(predicted_semantic_scores)
    if pcd_asset_seg.predicted_instance_class_labels is None or overwrite_existing_predictions:
        pcd_asset_seg.predicted_instance_class_labels = predicted_instance_class_labels
    else:
        pcd_asset_seg.predicted_instance_class_labels.update(predicted_instance_class_labels)

    if pcd_asset_seg.class_labels_to_instance_conf_scores is None or overwrite_existing_predictions:
        pcd_asset_seg.class_labels_to_instance_conf_scores = class_labels_to_instance_conf_scores
    else:
        pcd_asset_seg.class_labels_to_instance_conf_scores.update(class_labels_to_instance_conf_scores)

    if pcd_asset_seg.predicted_instance_ids is None or overwrite_existing_predictions:
        pcd_asset_seg.predicted_instance_ids = predicted_instance_ids
    else:
        pcd_asset_seg.predicted_instance_ids.update(predicted_instance_ids)

    if pcd_asset_seg.class_labels_to_conf_scores is None or overwrite_existing_predictions:
        pcd_asset_seg.class_labels_to_conf_scores = semantic_scores
    else:
        pcd_asset_seg.class_labels_to_conf_scores.update(semantic_scores)

    pcd_asset_seg.to_csv_file(src_pcd_csv_fp)


def save_softgroup_logits_and_instances_in_pcd_csv(dataset_dir: Path, results: List,
                                                   overwrite_existing_predictions: bool):
    for res in results:
        scan_id = res['scan_id']
        res.update(dict(site_ref_id=scan_id))
        _save_softgroup_logits_and_instances_in_pcd_csv(dataset_dir, res, overwrite_existing_predictions)


def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file for model params')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
    parser.add_argument('--input_dir', type=str, help='input dataset directory in telco format')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cfg_txt = open(args.config, 'r').read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    logger.info(f"Initializing Distributed Environment")
    if args.dist:
        init_dist()

    model = SoftGroup(**cfg.model).cuda()
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    logger.info(f'Load state dict from {args.checkpoint}')
    load_checkpoint(args.checkpoint, logger, model)

    logger.info(f'Converting dataset from telstra format to softgroup format')

    temp_dir = './tmp/telstra/converted_data'
    filename = './tmp/telstra/results/results.csv'
    src_ds_dir = Path(args.input_dir)
    dst_ds_dir = Path(temp_dir)
    area_idx = 'n'
    convert_to_softgroup(src_dataset_dir=src_ds_dir, dst_dataset_dir=dst_ds_dir,
                         class_label_to_class_idx_map=label_to_idx_map, area_idx=area_idx)

    results = []
    scan_ids, coords, colors, sem_preds, sem_labels = [], [], [], [], []
    offset_preds, offset_labels, inst_labels, pred_insts, gt_insts = [], [], [], [], []
    panoptic_preds = []
    final_results = []

    cfg.data.test.data_root = temp_dir
    cfg.data.test.prefix = ['Area_n']
    cfg.data.test.with_label = True
    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, dist=args.dist, **cfg.dataloader.test)

    _, world_size = get_dist_info()
    progress_bar_outer = tqdm(total=len(get_site_ref_ids_in_dataset_dir(dst_ds_dir)) * world_size,
                              disable=not is_main_process(), position=0, leave=False)

    eval_tasks = ['semantic', 'instance']
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            result = model(batch)
            results.append(result)
            progress_bar_outer.update(world_size)
        progress_bar_outer.close()
        results = collect_results_cpu(results, len(dataset))

    save_softgroup_logits_and_instances_in_pcd_csv(src_ds_dir, results, True)

    if is_main_process():
        for res in results:
            scan_ids.append(res['scan_id'])
            if 'semantic' in eval_tasks or 'panoptic' in eval_tasks:
                sem_labels.append(res['semantic_labels'])
                inst_labels.append(res['instance_labels'])
            if 'semantic' in eval_tasks:
                coords.append(res['coords_float'])
                colors.append(res['color_feats'])
                sem_preds.append(res['semantic_preds'])
                offset_preds.append(res['offset_preds'])
                offset_labels.append(res['offset_labels'])
            if 'instance' in eval_tasks:
                pred_insts.append(res['pred_instances'])
                gt_insts.append(res['gt_instances'])
            if 'panoptic' in eval_tasks:
                panoptic_preds.append(res['panoptic_preds'])

        if cfg.data.test.with_label:
            if 'instance' in eval_tasks:
                logger.info('Evaluate instance segmentation')
                eval_min_npoint = getattr(cfg, 'eval_min_npoint', None)
                scannet_eval = ScanNetEval(dataset.CLASSES, eval_min_npoint)
                scannet_eval.evaluate(pred_insts, gt_insts, filename=filename)
            if 'panoptic' in eval_tasks:
                logger.info('Evaluate panoptic segmentation')
                eval_min_npoint = getattr(cfg, 'eval_min_npoint', None)
                panoptic_eval = PanopticEval(dataset.THING, dataset.STUFF, min_points=eval_min_npoint)
                panoptic_eval.evaluate(panoptic_preds, sem_labels, inst_labels)
            if 'semantic' in eval_tasks:
                logger.info('Evaluate semantic segmentation and offset MAE')
                ignore_label = cfg.model.ignore_label
                evaluate_semantic_miou(sem_preds, sem_labels, ignore_label, logger,
                                       classes=['background'] + list(dataset.CLASSES),
                                       filename=filename)
                evaluate_semantic_acc(sem_preds, sem_labels, ignore_label, logger)
                evaluate_offset_mae(offset_preds, offset_labels, inst_labels, ignore_label, logger)


if __name__ == '__main__':
    main()
