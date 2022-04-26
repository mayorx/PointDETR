# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, partial_training_data, is_training, dropout_points, percent, point_idx):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.partial_training_data = partial_training_data
        self.is_training = is_training

        # if self.is_training and self.partial_training_data:
        if percent == 20:
            from datasets.annoted_img_ids import annoted_img_ids
        else:
            raise('percent .. {} is not supported.'.format(percent))
        self.annoted_imgs = annoted_img_ids
        print('partial training data .. ', self.partial_training_data)
        self.dropout_points = dropout_points
        self.point_idx = point_idx

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        if self.partial_training_data and self.is_training:
            while image_id not in self.annoted_imgs:
                idx = int(torch.randint(0, len(self.ids), (1,)))
                image_id = self.ids[idx]

        img, target = super(CocoDetection, self).__getitem__(idx)
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target, self.is_training and self.dropout_points)
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # print('target ... ', target)
        # print(img.size())

        if self.is_training:
            points_supervision, target = generate_target_training(target)
        else:
            points_supervision, target = generate_target_evaluation(target, self.point_idx)
        # print('target ', target , ' ... ')
        return img, points_supervision, target

def generate_target_training(target, K=1):
    boxes = target['boxes']
    labels = target['labels']
    N = len(boxes)

    object_ids = torch.arange(N)

    eps = 0.01

    relative_x = torch.Tensor(N, K).uniform_(-0.5 + eps, 0.5 - eps)
    relative_y = torch.Tensor(N, K).uniform_(-0.5 + eps, 0.5 - eps)

    x = boxes[:, 0, None] + boxes[:, 2, None] * relative_x
    y = boxes[:, 1, None] + boxes[:, 3, None] * relative_y
    x, y = x.reshape(-1), y.reshape(-1)


    points = torch.stack([
        x, y
    ], dim=1)

    boxes = boxes[:, None, :].repeat(1, K, 1).flatten(0, 1) #Nx4 , cx, cy, w, h
    l = points[:, 0] - boxes[:, 0] + 0.5 * boxes[:, 2]
    t = points[:, 1] - boxes[:, 1] + 0.5 * boxes[:, 3]
    r = boxes[:, 0] + 0.5 * boxes[:, 2] - points[:, 0]
    b = boxes[:, 1] + 0.5 * boxes[:, 3] - points[:, 1]
    reg_target = torch.stack([l, t, r, b], dim=1)

    labels = labels[:, None].repeat(1, K).reshape(-1)
    object_ids = object_ids[:, None].repeat(1, K).reshape(-1)

    return {'labels': labels, 'object_ids': object_ids, 'points': points}, {'reg_target': reg_target, 'boxes': boxes}

def generate_target_evaluation(target, point_idx):
    K = 1

    boxes = target['boxes']
    labels = target['labels']
    N = len(boxes)

    object_ids = torch.arange(N)

    if point_idx < 0:
        points = boxes[:, :2]
    else:
        points = target['points'][:, point_idx, :] ##attention .. 0 is a magic numer xxxxxxxx

    boxes = boxes[:, None, :].repeat(1, K, 1).flatten(0, 1)
    labels = labels[:, None].repeat(1, K).reshape(-1)
    object_ids = object_ids[:, None].repeat(1, K).reshape(-1)

    l = points[:, 0] - boxes[:, 0] + 0.5 * boxes[:, 2]
    t = points[:, 1] - boxes[:, 1] + 0.5 * boxes[:, 3]
    r = boxes[:, 0] + 0.5 * boxes[:, 2] - points[:, 0]
    b = boxes[:, 1] + 0.5 * boxes[:, 3] - points[:, 1]
    reg_target = torch.stack([l, t, r, b], dim=1)
    reg_target.clamp_(0., 1.)

    return {'labels': labels, 'object_ids': object_ids, 'points': points}, {'boxes': boxes, 'reg_target': reg_target, 'orig_size': target['orig_size'], 'image_id': target['image_id'], 'anno_ids': target['anno_ids']}


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target, dropout_points):
        K = 10

        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        points = [obj["points"] for obj in anno]
        points = torch.as_tensor(points, dtype=torch.float32).reshape(-1, 10, 2) # x,y
        points[:, :, 0].clamp_(min=0, max=w) #..  x, y  or  y, x??
        points[:, :, 1].clamp_(min=0, max=h)

        ids = [obj["id"] for obj in anno]
        ids = torch.tensor(ids, dtype=torch.int64)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])

        keep_num = int(keep.sum())
        if dropout_points and keep_num > K:
            keep[keep.nonzero().view(-1)[torch.randperm(keep_num)[:keep_num - K]]] = False

        boxes = boxes[keep]
        points = points[keep]
        classes = classes[keep]
        ids = ids[keep]

        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["points"] = points
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        target["anno_ids"] = ids
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, data_augment=False, strong_aug=False, is_training=False, without_crop=False):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    if image_set == 'train':
        if is_training:
            scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

            if without_crop:
                multi_scales = T.RandomResize(scales, max_size=1333)
            else:
                multi_scales = T.RandomSelect(
                        T.RandomResize(scales, max_size=1333),
                        T.Compose([
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=1333),
                        ])
                    )

            if strong_aug:
                return T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomColorJitter(prob=0.8),
                    T.RandomGrayScale(prob=0.2),
                    T.RandomGaussianBlur(prob=0.5),
                    multi_scales,
                    normalize,
                ])

            if data_augment:
                return T.Compose([
                    T.RandomHorizontalFlip(),
                    multi_scales,
                    normalize,
                ])
            else:
                return T.Compose([
                    T.RandomHorizontalFlip(),
                    T.RandomResize([800], max_size=1333),
                    normalize,
                ])

    return T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize,
    ])


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017_with_points.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017_with_points.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set, args.data_augment, args.strong_aug, not (args.eval or args.generate_pseudo_bbox), without_crop=args.without_crop),
        return_masks=args.masks,
        partial_training_data=args.partial_training_data,
        is_training=not (args.eval or args.generate_pseudo_bbox or image_set == "val"),
        dropout_points=args.dropout_points,
        percent=args.percent_of_training_data,
        point_idx=args.generated_point_idx
    )
    return dataset
