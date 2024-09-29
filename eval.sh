#!/bin/sh

config=$1
gpus=$2
output=$3

if [ -z $config ]
then
    echo "No config file found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $gpus ]
then
    echo "Number of gpus not specified! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

if [ -z $output ]
then
    echo "No output directory found! Run with "sh eval.sh [CONFIG_FILE] [NUM_GPUS] [OUTPUT_DIR] [OPTS]""
    exit 0
fi

shift 3
opts=${@}

#COCO
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/coco.json" \
 DATASETS.TEST \(\"coco_2017_test_stuff_all_sem_seg\"\,\) \
 MODEL.PROMPT_ENSEMBLE_TYPE imagenet_select \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

#ADE20k-150
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/ade150.json" \
 DATASETS.TEST \(\"ade20k_150_test_sem_seg\"\,\) \
 MODEL.PROMPT_ENSEMBLE_TYPE imagenet_select \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

#Pascal Context 59
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON  "datasets/pc59.json" \
 DATASETS.TEST \(\"context_59_test_sem_seg\"\,\) \
 MODEL.PROMPT_ENSEMBLE_TYPE imagenet_select \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts
 
#Cityscapes
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/cityscapes.json" \
 DATASETS.TEST \(\"cityscapes_fine_sem_seg_val\"\,\) \
 MODEL.PROMPT_ENSEMBLE_TYPE imagenet_select \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

#Pascal VOC
python train_net.py --config $config \
 --num-gpus $gpus \
 --dist-url "auto" \
 --eval-only \
 OUTPUT_DIR $output/eval \
 MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/voc20.json" \
 DATASETS.TEST \(\"voc_2012_test_sem_seg\"\,\) \
 MODEL.PROMPT_ENSEMBLE_TYPE imagenet_select \
 MODEL.WEIGHTS $output/model_final.pth \
 $opts

cat $output/eval/log.txt | grep copypaste