# PointDETR
pytorch implementation of the paper, points as queries: weakly semi-supervised object detection by points

##### Under development .. comming soon! including:
    * pretrained PointDETR at 20%  
    * 20% bbox + 80% pseudo-bbox annotation file (PointDETR.json)

### Requirements
This work is tested under:
```
ubuntu 18.04
python 3.6.9
torch 1.5.1
cuda 10.1
```

### Installation
```
pip install -r requirements.txt
```

### 0. Data Preparation
* COCO dataset ```./datasets/COCO``` 
* 20% image ids ```datasets```
* 20% bbox + 80% point annotation

### 1. Train PointDETR by 20% bbox
* ```python3 -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --coco_path ./datasets/COCO  --partial_training_data --output_dir ./ckpt-ps/point-detr-9x --epochs 108 --lr_drop 72 --data_augment --position_embedding sine --warm_up --multi_step_lr```

### 2. Generate 80% pseudo-bbox 
* ```python3 main.py --coco_path ./datasets/COCO --generate_pseudo_bbox --generated_anno PointDETR --position_embedding sine --resume ./ckpt-ps/point-detr-9x/baseline-checkpoint0107.pth```

-------  Student Model -------
### Install [cvpods](https://github.com/Megvii-BaseDetection/cvpods)

### 3. Train the student model with 20% bbox + 80% pseudo-bbox
* ```cd ./cvpods/playground/detection/coco/fcos-20p-pointdetr```
* ``` pods_train --num-gpus 8 --dir . ```

### 4. (optional) Train the student model with 20% bbox only. 
* ```cd ./cvpods/playground/detection/coco/fcos-20p-no_teacher```
* ``` pods_train --num-gpus 8 --dir . ```

### Citation
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

### Aknowledgement
This repo is built on the [cvpods](https://github.com/Megvii-BaseDetection/cvpods) and [DETR](https://github.com/facebookresearch/detr/)
