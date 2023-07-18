from ensemble_boxes import *
import pickle as pkl
from copy import deepcopy
import IPython
import torch
from tqdm import tqdm
import numpy as np

import mmengine
from mmengine import Config, DictAction
from mmengine.evaluator import Evaluator
from mmengine.registry import init_default_scope

from mmdet.registry import DATASETS
import nni

params = {
    'skip_box_thr': 0.14,
    'iou_thr': 0.75,
    'conf_type': 'avg',
    'w0': 0.6,
    'w1': 0.6,
    'w2': 0.7,
}
optimized_params = nni.get_next_parameter()
params.update(optimized_params)


def eval(predictions):

    cfg = Config.fromfile("./example/dataset_info.py")
    init_default_scope(cfg.get('default_scope', 'mmdet'))

    dataset = DATASETS.build(cfg.test_dataloader.dataset)

    evaluator = Evaluator(cfg.val_evaluator)
    evaluator.dataset_meta = dataset.metainfo
    eval_results = evaluator.offline_evaluate(predictions)
    return eval_results['coco/bbox_mAP']
    

names = [
    "./example/out_faster_rcnn.pkl",
    "./example/out_retinanet.pkl",
    "./example/out_fcos.pkl",
]

def load_results(name):
    results = pkl.load(open(name, "rb"))
    return results

results_list = []
for name in names:
    results = load_results(name)
    results_list.append(results)

final_list = []
weights = [params['w0'], params['w1'], params['w2'],]
iou_thr = params['iou_thr']
conf_type = params['conf_type']
skip_box_thr = params['skip_box_thr']
sigma = 0.1

for i in tqdm(range(len(results_list[0]))):
    boxes_list = []
    scores_list = []
    labels_list = []
    h,w = results_list[0][i]['ori_shape']
    for j in range(len(results_list)):
        tmp = results_list[j][i]
        bboxes = deepcopy(tmp['pred_instances']['bboxes'])
        #norm
        bboxes[:,0] /= w     
        bboxes[:,1] /= h     
        bboxes[:,2] /= w     
        bboxes[:,3] /= h     

        boxes_list.append(bboxes)    
        scores_list.append(tmp['pred_instances']['scores'])    
        labels_list.append(tmp['pred_instances']['labels'])
    #out_boxes, out_scores, out_labels = nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr)
    #out_boxes, out_scores, out_labels = soft_nms(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, sigma=sigma, thresh=skip_box_thr)
    #out_boxes, out_scores, out_labels = non_maximum_weighted(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    out_boxes, out_scores, out_labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type=conf_type)
    
    #norm
    out_boxes[:,0] *= w     
    out_boxes[:,1] *= h     
    out_boxes[:,2] *= w     
    out_boxes[:,3] *= h     

    out_item = deepcopy(results_list[0][i])
    out_item['pred_instances']['bboxes'] = torch.Tensor(out_boxes)
    out_item['pred_instances']['scores'] = torch.Tensor(out_scores)
    out_item['pred_instances']['labels'] = torch.Tensor(out_labels).long()
    final_list.append(out_item)
mAP = eval(final_list)

nni.report_final_result(mAP)
#pkl.dump(final_list, open("./out_results.pkl", "wb"))
