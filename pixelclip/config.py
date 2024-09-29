# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_pixelclip_config(cfg):
    """
    Add config for PixelCLIP.
    """
    # data config
    # select the dataset mapper
    cfg.INPUT.STRONG_AUG = False
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"

    cfg.DATASETS.VAL_ALL = ("coco_2017_val_all_stuff_sem_seg",)

    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # pixelclip config
    cfg.MODEL.EPS = 100.0
    cfg.MODEL.DINO = False
    cfg.MODEL.MAX_MASKS = 512
    cfg.MODEL.CLIP_RESOLUTION = (320, 320)
    cfg.MODEL.MASK_RESOLUTION = (256, 256)
    cfg.MODEL.PROMPT_ENSEMBLE_TYPE = "single"
    cfg.MODEL.CLIP_PIXEL_MEAN = [122.7709383, 116.7460125, 104.09373615]
    cfg.MODEL.CLIP_PIXEL_STD = [68.5005327, 66.6321579, 70.3231630]
    
    cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON = "datasets/coco.json"
    cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON = "datasets/ade150.json"
    cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED = "ViT-B/16"
    cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH = 0
    cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE = "attention"
    
    cfg.SOLVER.CLIP_MULTIPLIER = 0.01
    cfg.SOLVER.PROMPT_MULTIPLIER = 10.0