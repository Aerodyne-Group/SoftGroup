import numpy as np


def evaluate_semantic_acc(pred_list, gt_list, ignore_label=-100, logger=None):
    gt = np.concatenate(gt_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    assert gt.shape == pred.shape
    correct = (gt[gt != ignore_label] == pred[gt != ignore_label]).sum()
    whole = (gt != ignore_label).sum()
    acc = correct.astype(float) / whole * 100
    logger.info(f'Acc: {acc:.1f}')
    return acc


def evaluate_semantic_miou(pred_list, gt_list, ignore_label=-100, logger=None, classes=None, filename = None):
    gt = np.concatenate(gt_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    pos_inds = gt != ignore_label
    gt = gt[pos_inds]
    pred = pred[pos_inds]
    assert gt.shape == pred.shape
    iou_list = []
    if classes: iou_dict = dict(zip(classes, [0]*len(classes)))
    for _index in np.unique(gt):
        if _index != ignore_label:
            intersection = ((gt == _index) & (pred == _index)).sum()
            union = ((gt == _index) | (pred == _index)).sum()
            iou = intersection.astype(float) / union * 100
            iou_list.append(iou)
            if classes: iou_dict[classes[_index]] = iou
    miou = np.mean(iou_list)

    logger.info('Class-wise mIoU: ' + ' '.join(f'{x:.1f}' for x in iou_list))
    logger.info(f'mIoU: {miou:.1f}')
    if filename and classes:
        for c in classes:
            if c not in iou_dict:
                iou_dict[c] = 0.0
        iou_dict['all'] = miou
        import pandas as pd
        import os
        if os.path.exists(filename):
            temp = pd.read_csv(filename, index_col=0)
            dfn = pd.DataFrame({'mIOU': iou_dict})
            re = pd.merge(temp, dfn, left_on=temp.index, right_on=dfn.index)
            re.to_csv(filename, float_format='%.4f')
        else:
            pd.DataFrame({'mIOU': iou_dict}).to_csv(filename, float_format='%.4f')

    return miou

def evaluate_offset_mae(pred_list, gt_list, gt_instance_list, ignore_label=-100, logger=None):
    gt = np.concatenate(gt_list, axis=0)
    pred = np.concatenate(pred_list, axis=0)
    gt_instance = np.concatenate(gt_instance_list, axis=0)
    pos_inds = gt_instance != ignore_label
    gt = gt[pos_inds]
    pred = pred[pos_inds]
    mae = np.abs(gt - pred).sum() / pos_inds.sum()
    logger.info(f'Offset MAE: {mae:.3f}')
    return mae
