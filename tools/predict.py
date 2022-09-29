import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
import torch
from munch import Munch

from torch.nn.parallel import DistributedDataParallel
from tower_equipment_segmentation.dataset_converter.dataset_converter import DatasetConverter
from tower_equipment_segmentation.dvo.known_class_labels import KnownClassLabels
from tower_equipment_segmentation.dvo.point_cloud_asset_segmentation import PointCloudAssetSegmentation
from tower_equipment_segmentation.util.dataset_util import get_site_ref_ids_in_dataset_dir, \
    get_pcd_csv_file_path_for_site_ref_id
from tower_equipment_segmentation.util.logging_util import setup_logger
from tqdm import tqdm

from softgroup.data import build_dataloader, build_dataset
from softgroup.model import SoftGroup
from softgroup.util import (collect_results_cpu, get_dist_info, init_dist,
                            is_main_process, load_checkpoint, rle_decode)

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


def _get_instance_scores_and_preds(instances: List[Dict[str, Any]], pcd_size: int, iou_threshold: float,
                                   conf_threshold: float) -> Tuple[List[str], List[int], Dict[str, np.ndarray]]:
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
        # min_area = [min(mask_areas[i], mask_areas[j]) for j in order[1:]]
        iou = np.array(intersection) / np.array(union)

        inds = np.where(iou < iou_threshold)[0]
        order = order[inds + 1]

    instances = [instances[i] for i in keep if instances[i]['conf'] > conf_threshold]
    instance_class_labels = np.ones((pcd_size,), dtype=int) * -1
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
    return predicted_instance_class_labels, list(predicted_instance_ids), class_labels_to_instance_conf_scores


def _get_semantic_scores(semantic_scores: np.ndarray) -> Dict[str, np.ndarray]:
    class_labels = [KnownClassLabels.background, KnownClassLabels.antenna, KnownClassLabels.transceiver_junction,
                    KnownClassLabels.head_frame_mount, KnownClassLabels.shelter]
    semantic_scores_dict = dict()
    for i, label in enumerate(class_labels):
        semantic_scores_dict[label] = semantic_scores[:, i].reshape(-1, )
    return semantic_scores_dict


def _save_softgroup_logits_and_instances_in_pcd_csv(dataset_dir: Path, result: Dict[str, Any],
                                                    overwrite_existing_predictions: bool, iou_threshold: float,
                                                    conf_threshold: float):
    site_ref_id = result['site_ref_id']
    predicted_instances = result['pred_instances']
    predicted_semantic_scores = result['semantic_scores']

    src_pcd_csv_fp = get_pcd_csv_file_path_for_site_ref_id(dataset_dir, site_ref_id)
    pcd_asset_seg = PointCloudAssetSegmentation.from_csv_file(src_pcd_csv_fp)
    pcd_size = pcd_asset_seg.num_points

    predicted_instance_class_labels, predicted_instance_ids, class_labels_to_instance_conf_scores \
        = _get_instance_scores_and_preds(predicted_instances, pcd_size, iou_threshold, conf_threshold)
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


def save_softgroup_logits_and_instances_in_pcd_csv(dataset_dir: Path, results: List[Dict[str, Any]],
                                                   overwrite_existing_predictions: bool, iou_threshold: float,
                                                   conf_thresold: float):
    for res in results:
        scan_id = res['scan_id']
        res.update(dict(site_ref_id=scan_id))
        _save_softgroup_logits_and_instances_in_pcd_csv(dataset_dir, res, overwrite_existing_predictions, iou_threshold,
                                                        conf_thresold)


def get_args():
    parser = argparse.ArgumentParser('SoftGroup')
    parser.add_argument('config', type=str, help='path to config file for model params')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--dist', action='store_true', help='run with distributed parallel')
    parser.add_argument('--input_dir', type=str, help='input dataset directory in telco format')
    parser.add_argument('--iou_threshold', type=str, help='iou threshold for nms', default=0.8)
    parser.add_argument('--conf_threshold', type=str, help='confidence threshold for predictions', default=0.09)

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
    src_ds_dir = Path(args.input_dir)
    dst_ds_dir = Path(temp_dir)
    area_idx = 'n'
    convert_to_softgroup(src_dataset_dir=src_ds_dir, dst_dataset_dir=dst_ds_dir,
                         class_label_to_class_idx_map=label_to_idx_map, area_idx=area_idx)

    results = []
    cfg.data.test.data_root = temp_dir
    cfg.data.test.prefix = ['Area_n']
    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, dist=args.dist, **cfg.dataloader.test)

    _, world_size = get_dist_info()
    progress_bar_outer = tqdm(total=len(get_site_ref_ids_in_dataset_dir(dst_ds_dir)) * world_size,
                              disable=not is_main_process(), position=0, leave=False)

    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dataloader):
            result = model(batch)
            results.append(result)
            progress_bar_outer.update(world_size)
        progress_bar_outer.close()
        results = collect_results_cpu(results, len(dataset))
    save_softgroup_logits_and_instances_in_pcd_csv(src_ds_dir, results, True, iou_threshold=args.iou_threshold,
                                                   conf_thresold=args.conf_threshold)


if __name__ == '__main__':
    main()
