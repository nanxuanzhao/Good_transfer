## What Makes Instance Discrimination Good for Transfer Learning?



**What Makes Instance Discrimination Good for Transfer Learning?**  
Nanxuan Zhao*	Zhirong Wu*	Rynson W.H. Lau	Stephen Lin



## Pretrained Models

##### Different data augmentations for learning self-supervised and supervised representations (Table 1).

| Pretraining  | Pytorch Augmentation               | Download                                                     |
| ------------ | ---------------------------------- | ------------------------------------------------------------ |
| Unsupervised | \+ RandomHorizontalFlip(0.5)       | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/moco_v2_imagenet_aug_f.pth) |
|              | \+ RandomResizedCrop(224)          | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/moco_v2_imagenet_aug_fr.pth) |
|              | \+ ColorJitter(0.4, 0.4, 0.4, 0.1) | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/moco_v2_imagenet_aug_frc.pth) |
|              | \+ RandomGrayscale(p=0.2)          | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/moco_v2_imagenet_aug_frcg.pth) |
|              | \+ GaussianBlur(0.1, 0.2)          | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/moco_v2_imagenet.pth) |
| supervised   | \+ RandomHorizontalFlip(0.5)       | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/supervised_imagenet_aug_f.pth) |
|              | \+ RandomResizedCrop(224)          | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/supervised_imagenet_aug_fr.pth) |
|              | \+ ColorJitter(0.4, 0.4, 0.4, 0.1) | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/supervised_imagenet_aug_frc.pth) |
|              | \+ RandomGrayscale(p=0.2)          | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/supervised_imagenet_aug_frcg.pth) |
|              | \+ GaussianBlur(0.1, 0.2)          | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/supervised_imagenet.pth) |



##### Transfer performance with pretraining on various datasets (Table 2).

| Pretraining  | Pretraining Data | Download                                                     |
| ------------ | ---------------- | ------------------------------------------------------------ |
| Unsupervised | ImageNet         | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/moco_v2_imagenet.pth) |
|              | ImageNet-10%     | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/moco_v2_imagenet1over10.pth) |
|              | ImageNet-100     | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/moco_v2_imagenet100.pth) |
|              | Places           | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/moco_v2_palces.pth) |
|              | CelebA           | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/moco_v2_celeba.pth) |
|              | COCO             | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/moco_v2_coco.pth) |
|              | Synthia          | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/moco_v2_synthia.pth) |
| Supervised   | ImageNet         | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/supervised_synthia.pth) |
|              | ImageNet-10%     | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/supervised_imagenet1over10.pth) |
|              | ImageNet-100     | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/supervised_imagenet100.pth) |
|              | Places           | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/supervised_places.pth) |
|              | CelebA           | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/supervised_celeba.pth) |
|              | COCO             | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/supervised_coco.pth) |
|              | Synthia          | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/supervised_synthia.pth) |



##### Exemplar-based supervised pretraining (Table 3).

| Model       | Download                                                     |
| ----------- | ------------------------------------------------------------ |
| Exemplar v1 | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/exemplar_v1.pth) |
| Exemplar v2 | [model](https://frontiers.blob.core.windows.net/pretraining/checkpoints/pil_pretrained_models/good_transfer/exemplar_v2.pth) |



## Citation

If you use this work in your research, please cite:

```
@inproceedings{ZhaoICLR2021, 
    author = {Nanxuan Zhao and Zhirong Wu and Rynson W.H. Lau and Stephen Lin}, 
    title = {What Makes Instance Discrimination Good for Transfer Learning?}, 
    booktitle = {ICLR}, 
    year = {2021} 
}
```

