# Prepare Datasets for PixelCLIP

A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
This document explains how to setup the builtin datasets so they can be used by the above APIs.
[Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`,
and how to add new datasets to them.

PixelCLIP has builtin support for a few datasets.
The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$SAM_DATASETS/            # SA-1B
  images/   
  annotations/

$DETECTRON2_DATASETS/
  coco/                   # COCO-Stuff
  ADEChallengeData2016/   # ADE20K-150
  VOCdevkit/ 
    VOC2010/              # PASCAL Context
    VOC2012/              # PASCAL VOC
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.


## Prepare data for [SA-1B](https://ai.meta.com/datasets/segment-anything/):

### Expected data structure

```
$SAM_DATASETS/
  images/
    sa_000020/
      image_list.da     # Generated by prepare_sa1b.py
      sa_223750.jpg
      ...
    sa_000021/
      image_list.da     # Generated by prepare_sa1b.py
      ...
    ...
  annotations/
    sa_000020/
      sa_223750.json
      ...
    sa_000021/
    ...
```
For training, we prepare 10% of SA-1B dataset, in which we train with 5%. This is corresponds to 100 tar files from [SA-1B](https://ai.meta.com/datasets/segment-anything-downloads/), where we provide a list of the subset we used in `sa1b.txt`. Download the tar files and extract as shown above, then run the following in `SAM_DATASETS` directory.

```
python prepare_sa1b.py
```

**❗️Note:** The images and annotations from SA-1B is provided in high resolution, where the shorter side is set to 1500. 
In practice, we resize the image and the corresponding annotation into half, resulting in the shorter side being 750. This significantly reduces the RLE decoding time for loading SA-1B mask annotations, which bottlenecks the dataloader.


## Prepare data for [COCO-Stuff](https://github.com/nightrome/cocostuff):

### Expected data structure

```
coco-stuff/
  annotations/
    train2017/
    val2017/
  images/
    train2017/
    val2017/
  # below are generated by prepare_coco_stuff.py
  annotations_detectron2/
    train2017/
    val2017/ 
```
Download the COCO (2017) images from https://cocodataset.org/

```bash
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
```

Download the COCO-Stuff annotation from https://github.com/nightrome/cocostuff.
```bash
wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
```
Unzip `train2017.zip`, `val2017.zip`, and `stuffthingmaps_trainval2017.zip`. Then put them to the correct location listed above. and generate the labels for training and testing.

```
python datasets/prepare_coco_stuff.py
```



## Prepare data for [ADE20K-150](http://sceneparsing.csail.mit.edu):

### Expected data structure 
```
ADEChallengeData2016/
  annotations/
    validation/
  images/
    validation/
  # below are generated by prepare_ade20k_150.py
  annotations_detectron2/
    validation/
```
Download the data of ADE20K-150 from http://sceneparsing.csail.mit.edu.
```
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
```
Unzip `ADEChallengeData2016.zip` and generate the labels for testing.
```
python datasets/prepare_ade20k_150.py
```

## Prepare data for [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit):


### Expected data structure 
```
VOCdevkit/
  VOC2012/
    Annotations/
    ImageSets/
    JPEGImages/
    SegmentationClass/
    SegmentationClassAug/ 
    SegmentationObject/
    # below are generated by prepare_voc.py
    annotations_detectron2
    annotations_detectron2_bg

```
Download the data of PASCAL VOC from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit.

We use SBD augmentated training data as SegmentationClassAug following [Deeplab](https://github.com/kazuto1011/deeplab-pytorch/blob/master/data/datasets/voc12/README.md).
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip
```
Unzip `VOCtrainval_11-May-2012.tar` and `SegmentationClassAug.zip`. Then put them to the correct location listed above and generate the labels for testing.
```
python datasets/prepare_voc.py
```


## Prepare data for [CityScapes](https://www.cityscapes-dataset.com/downloads/):


### Expected data structure 
```
cityscapes/
  gtFine/
    train/
      aachen/
        color.png, instanceIds.png, labelIds.png, polygons.json,
        labelTrainIds.png
      ...
    val/
    test/
  leftImg8bit/
    train/
    val/
    test/
```
Install cityscapes scripts by:
```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```
Note: to create labelTrainIds.png, first prepare the above structure, then run cityscapesescript with:

```
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```


## Prepare data for [PASCAL Context](https://www.cs.stanford.edu/~roozbeh/pascal-context/):


### Expected data structure 
```
VOCdevkit/
  VOC2010/
    Annotations/
    ImageSets/
    JPEGImages/
    SegmentationClass/
    SegmentationObject/
    trainval/
    labels.txt
    pascalcontext_val.txt
    trainval_merged.json
    # below are generated by prepare_pascal_context_59.py
    annotations_detectron2/
      pc59_val
```
Download the data of PASCAL VOC 2010 from https://www.cs.stanford.edu/~roozbeh/pascal-context/. 

```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
```
Download the annotation for [59](https://codalabuser.blob.core.windows.net/public/trainval_merged.json) classes.
```
wget https://codalabuser.blob.core.windows.net/public/trainval_merged.json
```
Unzip `VOCtrainval_03-May-2010.tar`. Then put them to the correct location listed above and generate the labels for testing.
```
python datasets/prepare_pascal_context_59.py
```