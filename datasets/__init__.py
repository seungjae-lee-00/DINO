# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .strad_dataset import StradDetection, make_strad_transforms

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build_coco(image_set, args)
    if args.dataset_file == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    if args.dataset_file == 'o365':
        from .o365 import build_o365_combine
        return build_o365_combine(image_set, args)
    if args.dataset_file == 'vanke':
        from .vanke import build_vanke
        return build_vanke(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')

def build_sv_ds(dataset_root, data_csv, strong_aug=False):
    transforms = make_strad_transforms('train', fix_size=False, strong_aug=True)
    dataset = StradDetection(data_csv, dataset_root, transforms=transforms) 
                            #  transforms=make_coco_transforms(image_set, fix_size=args.fix_size, strong_aug=strong_aug)
    return dataset

