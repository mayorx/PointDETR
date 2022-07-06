# PointDETR
pytorch implementation of the paper, points as queries: weakly semi-supervised object detection by points

    
##### Files
* [annotations with points, (train)](https://pan.baidu.com/s/1BMrEmZhZ356UKkfi6u0ylQ?pwd=jj2o)
* [annotations with points, (val)](https://pan.baidu.com/s/1hGBYMbUQu8svcWL_JooXxw?pwd=rcvl)
    * We annotate 10 points to test the robustness of the method. To reproduce the paper results, this repo use the first point of 10 points (idx: 0).
* pretrained [PointDETR-9x.pth](https://pan.baidu.com/s/1xMZVK67Tl57bN5GOaTLSFQ?pwd=bo7k) at 20%.
* 20% bbox + 80% pseudo-bbox annotation file, [PointDETR.json](https://pan.baidu.com/s/1EPXFptsugxLNdaQ3fLylLA?pwd=r5h8)


## Requirements
This work is tested under:
```
ubuntu 18.04
python 3.6.9
torch 1.5.1
cuda 10.1
```

## Installation
```
pip install -r requirements.txt
```
## Instructions

#### 0. Data Preparation
* COCO dataset ```./datasets/COCO``` 
* 20% image ids ```in ./datasets/annoted_img_ids.py && ./cvpods/datasets/annoted_img_ids.py``` 

#### 1. Train PointDETR by 20% bbox
* ```python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path ./datasets/COCO  --partial_training_data --output_dir ./ckpt-ps/point-detr-9x --epochs 108 --lr_drop 72 --data_augment --position_embedding sine --warm_up --multi_step_lr```

#### 2. Generate 80% pseudo-bbox 
* ```python3 main.py --coco_path ./datasets/COCO --generate_pseudo_bbox --generated_anno PointDETR --position_embedding sine --resume ./ckpt-ps/point-detr-9x/baseline-checkpoint0107.pth```

-------  Student Model -------
#### Install [cvpods](https://github.com/Megvii-BaseDetection/cvpods)

#### 3. Train the student model with 20% bbox + 80% pseudo-bbox
* ```cd ./cvpods/playground/detection/coco/fcos-20p-pointdetr```
* ``` pods_train --num-gpus 8 --dir . ```

#### 4. (optional) Train the student model with 20% bbox only. 
* ```cd ./cvpods/playground/detection/coco/fcos-20p-no_teacher```
* ``` pods_train --num-gpus 8 --dir . ```

## Citation
If this work helps your research / work, please consider citing:
```
@inproceedings{chen2021points,
  title={Points as queries: Weakly semi-supervised object detection by points},
  author={Chen, Liangyu and Yang, Tong and Zhang, Xiangyu and Zhang, Wei and Sun, Jian},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8823--8832},
  year={2021}
}
```

## Aknowledgement
This repo is built on the [cvpods](https://github.com/Megvii-BaseDetection/cvpods) and [DETR](https://github.com/facebookresearch/detr/)
