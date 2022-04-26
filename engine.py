# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from util.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, box_iou


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('mIoU', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    # cnt = 0
    for samples, points, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)
        points = [{k: v.to(device) for k, v in t.items()} for t in points]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # targets['boxes']
        # targets['labels']
        # targets['object_ids']
        # targets['points']

        # print('tar .. gets ... ', targets)

        outputs = model(samples, points)

        # print('outputs ... ', outputs)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        losses = losses + 0 * sum(p.sum() for p in model.parameters())

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # if cnt % 100 == 0:
        #     from util.gradflow_check import plot_grad_flow
        #     plot_grad_flow('epoch-{}-points-log-{:05}'.format(epoch, cnt), model.named_parameters())

        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(mIoU=loss_dict_reduced['mIoU'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # cnt += 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def visual_delta(query, deltas):
    #deltas 2000x100x2

    print('deltas size .. ', deltas.size())
    N, query_num, _ = deltas.size()

    for i in range(query_num):
        x, y = deltas[:, i, 0], deltas[:, i, 1]
        x, y = x.cpu().numpy(), y.cpu().numpy()

        import matplotlib.pyplot as plt
        plt.clf()
        axes = plt.gca()
        axes.set_xlim([-1., 1.])
        axes.set_ylim([-1., 1.])
        plt.plot(x, y, 'bo')
        plt.savefig('./visual/by-query/deltas/query-{}-{:02}.jpg'.format(query, i))

    pass


def visualize(results, targets, output_dir, idx, visual_order, need_print, fixed_query=[0]):
    def inter(x1, x2, x3, x4):
        return max(min(x2, x4) - max(x1, x3), 0)

    def iou(bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        w1, h1 = x2 - x1, y2 - y1

        x2, y2, x3, y3 = bbox2
        w2, h2 = x3 - x2, y3 - y2

        # x1, y1, w1, h1 = bbox1
        # x2, y2, w2, h2 = bbox2

        s1 = w1 * h1
        s2 = w2 * h2
        # x1 ~ x1 + w1
        # x2 ~ x2 + w2
        s3 = inter(x1, x1 + w1, x2, x2 + w2) * inter(y1, y1 + h1, y2, y2 + h2)

        # if (s1 + s2 - s3 < 0.0001):
        #     print(bbox1, bbox2)
        #     return 1.
        return s3 / (s1 + s2 - s3)

    coco_path = '/data/Datasets/COCO'
    coco_dirs = {'train2017', 'val2017'}
    import os
    import cv2

    # label_text = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']
    label_text = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    # print('in visualize .. ')
    # print(results)
    # print('targets .. ', targets)
    # print(output_dir)
    # print(idx)

    colors_accept = (0, 255, 87)
    colors_reject = (0, 0, 255)

    qqq = []

    cnt = 0
    for img_id in results:
        output_path = os.path.join(output_dir, str(img_id))

        try:
            os.mkdir(output_path)
        except:
            pass

        img_file_path = None

        for dir in coco_dirs:
            img_file_path = os.path.join(coco_path, dir, '{:012}.jpg'.format(img_id))
            if os.path.exists(img_file_path):
                break

        # img = cv2.imread(img_file_path)

        scores = results[img_id]['scores']
        labels = results[img_id]['labels']
        boxes = results[img_id]['boxes'] #xyxy

        if visual_order == 'score':
            _, idxes = scores.sort(descending=True)
        else:
            idxes = [i for i in range(len(scores))]
        scores = scores[idxes]
        labels = labels[idxes]
        boxes = boxes[idxes]

        each_target = targets[cnt]
        h, w = each_target['orig_size'] #todo, may be h, w

        cx = 0.5 * (boxes[:, 0] + boxes[:, 2]) / w
        cy = 0.5 * (boxes[:, 1] + boxes[:, 3]) / h

        qqq_query = []
        for each_fixed_query in fixed_query:
            delta_x = cx - cx[each_fixed_query]
            delta_y = cy - cy[each_fixed_query]
            qqq_query.append(torch.stack([delta_x, delta_y], dim=1))
        qqq.append(torch.stack(qqq_query, dim=1))
        # qqq.append(torch.stack([delta_x, delta_y], dim=1))

        # fixed_query, boxes[]



        gt_boxes = each_target['boxes']
        gt_boxes = [[(bbox[0]-0.5*bbox[2])*w , (bbox[1]-0.5*bbox[3])*h, (bbox[0]+0.5*bbox[2])*w, (bbox[1]+0.5*bbox[3])*h] for bbox in gt_boxes]

        if not need_print:
            continue
        #cx, cy, w, h

        # print('gt_boxes .. ', gt_boxes)
        gt_labels = each_target['labels']

        for i in range(len(scores)):
            score = scores[i]
            label = int(labels[i])
            # label = labels[i]

            bbox = boxes[i]

            img = cv2.imread(img_file_path)
            img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
                                (int(bbox[2]), int(bbox[3])), colors_reject, 2)

            matched = False

            for j in range(len(gt_boxes)):
                if gt_labels[j] == label and iou(gt_boxes[j], bbox) > 0.5:
                    gt_bbox = gt_boxes[j]
                    img = cv2.rectangle(img, (int(gt_bbox[0]), int(gt_bbox[1])),
                                (int(gt_bbox[2]), int(gt_bbox[3])), colors_accept, 2)
                    matched = True

            file_name = 'obj-{:02}-label-{}-score-{:.4f}-query-{}-matched-{}.jpg'.format(i, label_text[int(label)], float(score), int(idxes[i]), 1 if matched else 0)
            cv2.imwrite(os.path.join(output_path, file_name), img)
        cnt += 1
    return torch.stack(qqq, dim=0)

def visual_point(pred_boxes, point_supervision, target):
    # print('point ..... ', point_supervision)
    # print(' target ... ', target)

    labels = point_supervision["labels"]
    points = point_supervision["points"]
    object_ids = point_supervision["object_ids"]

    boxes = target["boxes"]
    img_id = target["image_id"]
    h, w = target["orig_size"]

    coco_path = '/data/Datasets/COCO/'
    coco_dirs = {'train2017', 'val2017'}
    import os
    import cv2

    img_file_path = ''
    for dir in coco_dirs:
        img_file_path = os.path.join(coco_path, dir, '{:012}.jpg'.format(int(img_id)))
        if os.path.exists(img_file_path):
            break

    colors_accept = (0, 255, 87)
    colors_reject = (0, 0, 255)


    for obj_id in range(len(boxes)):
        bbox = box_cxcywh_to_xyxy(pred_boxes[obj_id])
        img = cv2.imread(img_file_path)
        # print('img .. ', img.shape, '.... ', w, h)
        img = cv2.rectangle(img, (int(w*bbox[0]), int(h*bbox[1])), (int(w*bbox[2]), int(h*bbox[3])), colors_reject, 2)
        gt_bbox = box_cxcywh_to_xyxy(boxes[obj_id])
        img = cv2.rectangle(img, (int(w*gt_bbox[0]), int(h*gt_bbox[1])), (int(w*gt_bbox[2]), int(h*gt_bbox[3])), colors_accept, 2)
        each = points[obj_id]

        cv2.circle(img, (int(w*each[0]), int(h*each[1])), radius=3, color=colors_accept, thickness=-1)

        # file_name = 'obj-{:02}-label-{}-score-{:.4f}-query-{}-matched-{}.jpg'.format(i, label_text[int(label)], float(score), int(idxes[i]),)
        output = './visual/debug'
        file_name = 'img-{}-obj-{:02}.jpg'.format(int(img_id), int(obj_id))
        cv2.imwrite(os.path.join(output, file_name), img)

    # exit(0)

    pass


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, visual=False, visual_num=None, visual_order=None):
    # raise NotImplementedError("evaluation is not supported yet.")
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('mIoU', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    coco_evaluator = None
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    visual_cnt = 0
    # summary_query = []
    # fixed_query = [0, 60]

    for samples, points, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        points = [{k: v.to(device) for k, v in t.items()} for t in points]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    # for samples, targets in metric_logger.log_every(data_loader, 10, header):
    #     samples = samples.to(device)
    #     targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, points)

        # print('points ... ', points)
        # print(' ..... ', targets)

        # print('\n\n\n')
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(mIoU=loss_dict_reduced['mIoU'])

        if visual and visual_cnt < visual_num:
            visual_cnt+= 1

            for idx in range(len(targets)):
                visual_point(outputs['pred_boxes'][idx], points[idx], targets[idx])

        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)


        # results = postprocessors['bbox'](outputs, orig_target_sizes)


        # res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # if visual:
        #
        #     summary_query.append(
        #         visualize(res, targets, output_dir, visual_cnt, visual_order, visual_cnt < visual_num, fixed_query))
        #     visual_cnt += 1
        #
        #     if visual_cnt == 1000:
        #         summary_query = torch.cat(summary_query, dim=0)
        #         print('summary query .. ', summary_query.size())
        #         for q in range(len(fixed_query)):
        #             visual_delta(fixed_query[q], summary_query[:, :, q, :])
        #
        #         exit(0)

        # if coco_evaluator is not None:
        #     coco_evaluator.update(res)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator

@torch.no_grad()
def generate(model, data_loader, device, generated_anno_name, percent, coco_path):
    model.eval()
    #init ... read annotations from file ... (input)
    #output name :   root_dir / {generated_anno}.json
    import os
    import json
    print('percent .. ', percent, len(data_loader))
    if percent == 20:
        from datasets.annoted_img_ids import annoted_img_ids
    else:
        raise ('percent .. {} is not supported.'.format(percent))

    # from cvpack.dataset.annoted_img_ids import annoted_img_ids
    annoted_imgs = annoted_img_ids


    # root_dir = "/data/Datasets/COCO/annotations/"
    root_dir = os.path.join(coco_path, 'annotations')
    input_file = os.path.join(root_dir, 'instances_train2017.json')
    output_file = os.path.join(root_dir, '{}.json'.format(generated_anno_name))

    f = open(os.path.join(root_dir, input_file))  # coco dataset json file
    lines = f.readlines()
    f.close()

    data = lines[0]
    json_data = json.loads(data)
    annos = json_data['annotations']

    anno_id_mapping = {}

    n = len(annos)
    # print(' n .. ', n)
    for i in range(n):
        anno_id_mapping[annos[i]['id']] = i


    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # metric_logger.add_meter('mIoU', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'gen [{}] ... '.format(generated_anno_name)
    print_freq = 10
    # cnt = 0
    sum_iou = 0.
    iou_cnt = 0.

    for samples, points, targets in metric_logger.log_every(data_loader, print_freq, header):
        # cnt+=1
        # if cnt > 10:
        #     break

        samples = samples.to(device)
        points = [{k: v.to(device) for k, v in t.items()} for t in points]
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, points)

        for batch_idx in range(len(outputs['pred_boxes'])):
            outputs['pred_boxes'][batch_idx] = outputs['pred_boxes'][batch_idx].detach().to('cpu')

        # [     for each_output in outputs]

        for idx in range(len(targets)):
            target = targets[idx]
            boxes = target["boxes"]
            reg_target = target["reg_target"]

            #boxes ..
            img_id = int(target["image_id"])
            anno_ids = target["anno_ids"]
            ori_h, ori_w = target["orig_size"]

            # print('annoted img ids ', type(annoted_imgs))
            # print('img_id .. ', img_id, img_id in annoted_imgs, int(img_id) in annoted_imgs)
            if img_id in annoted_imgs:
                for i in range(len(anno_ids)):
                    each_anno_id = int(anno_ids[i])
                    # boxes[i]
                    j = anno_id_mapping[each_anno_id]
                    pseudo_bbox = annos[j]['bbox']
                    annos[j]['pseudo-bbox'] = pseudo_bbox
                    annos[j]['pseudo-score'] = 1.
                continue


            # visual_point(outputs['pred_boxes'][idx], points[idx], targets[idx])
            # print('anno_ids .. ', anno_ids, boxes, img_id)
            # print('outputs .. ', outputs['pred_boxes'][idx])


            for i in range(len(anno_ids)):
                j = anno_id_mapping[int(anno_ids[i])]

                pred = outputs['pred_boxes'][idx][i]

                # print('reg target', reg_target[i], 'box ', boxes[i], ' pred .. ', pred)

                cur_x , cur_y = points[idx]["points"][i]
                l, t, r, b = pred

                # pseudo_bbox = [
                #     cur_x - l,
                #     cur_y - t,
                #     l + r,
                #     t + b
                # ]

                pseudo_bbox = [
                    float(ori_w * (cur_x - l)),
                    float(ori_h * (cur_y - t)),
                    float(ori_w * (l + r)),
                    float(ori_h * (t + b))
                ]

                annos[j]['pseudo-bbox'] = pseudo_bbox
                    # print('pseudo-bbox', pseudo_bbox, 'gt-bbox', annos[j]['bbox'])
                iou, _ = box_iou(
                    box_xywh_to_xyxy(torch.tensor(pseudo_bbox).unsqueeze(0)),
                    box_xywh_to_xyxy(torch.tensor(annos[j]['bbox']).unsqueeze(0))
                )
                sum_iou += float(iou[0][0])
                iou_cnt += 1

                if torch.rand((1,)) < 0.01:
                    print('pseudo-bbox', pseudo_bbox, 'gt-bbox', annos[j]['bbox'], float(iou[0][0]), 'mIoU', sum_iou / iou_cnt)

                annos[j]['pseudo-score'] = 0.5

    print('print to ... ', output_file)
    json_data['annotations'] = annos
    f = open(output_file, 'w')  # target json file
    j = json.dumps(json_data)
    print(j, file=f)
    f.close()

    #read images from data_loader ..
    #for each image && points && targets  in   data_loader ...
        # pred it's .. points ... -> bboxes

        # if image_id in annoted_img ..   using ground truth


        # bboxes back to x,y,w,h

        # update annos

    # save anno

    #using their
