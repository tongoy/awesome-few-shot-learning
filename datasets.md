# Datasets



## Standard Few-shot Learning

### *mini*ImageNet
The [*mini*ImageNet]() dataset is a subset of ImageNet consisting of 100 classes with 600 images per class.
The standard splits include 64 classes for training, 16 classes for validation, and 20 classes for test.

[[paper](https://arxiv.org/abs/1606.04080)]
[[code]()]
[[splits](https://github.com/twitter-research/meta-learning-lstm/tree/master/data/miniImagenet)]
[[download](https://image-net.org/challenges/LSVRC/2015/2015-downloads.php)]



### *tiered*ImageNet
The *tiered*ImageNet dataset is a subset of ImageNet consisting of 608 classes (grouped into 34 high-level categories based on ImageNet hierarchy).
The standard splits include 351 classes (20 categories) for training, 97 classes (6 categories) for validation, and 160 classes (8 categories) for test.
The image numbers are 448,695, 124,261 and 206,209 for training, validation and test, respectively.

[[paper](https://arxiv.org/abs/1803.00676)]
[[code](https://github.com/renmengye/few-shot-ssl-public)]
[[splits](https://github.com/renmengye/few-shot-ssl-public/tree/master/fewshot/data/tiered_imagenet_split)]
[[download](https://image-net.org/challenges/LSVRC/2015/2015-downloads.php)]



### CUB-200-2011
The CUB-200-2011 dataset consists of 200 different classes of birds with a total of 11,788 images.
The standard splits include 100 base classes, 50 validation classes, and 50 novel classes.

[[paper](https://authors.library.caltech.edu/records/cvm3y-5hh21)]
[[code](https://www.vision.caltech.edu/datasets/cub_200_2011/)]
[[splits](https://github.com/bl0/negative-margin.few-shot/blob/master/data/CUB/write_CUB_filelist.py)]
[[download](https://data.caltech.edu/records/65de6-vp158)]



### CIFAR-FS
The CIFAR-FS dataset is a variant of CIFAR-100 consisting of 100 classes with 600 images per class.
The standard splits include 64 classes for training, 16 classes for validation, and 20 classes for test.

[[paper](https://arxiv.org/abs/1805.08136)]
[[code](https://github.com/bertinetto/r2d2)]
[[splits](https://drive.google.com/file/d/1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI/view)]
[[download](https://drive.google.com/file/d/1pTsCCMDj45kzFYgrnO67BWVbKs48Q3NI/view)]



## Cross-domain Few-shot Learning

### PlantVillage
The PlantVillage dataset consists of 54,306 images of plant leaves from 14 crop species with a spread of 38 classes (including 26 diseases and 12 healthy).
The dataset provides three different versions of the images, i.e., colored images, gray-scaled images and segmented images.
All images in the colored version, **i.e., 54,305 images of 38 classes**, are used for cross-domain few-shot learning.

[[paper](https://www.frontiersin.org/articles/10.3389/fpls.2016.01419/full)]
[[code](https://github.com/digitalepidemiologylab/plantvillage_deeplearning_paper_analysis)]
[[splits]()]
[[download1](https://www.kaggle.com/datasets/saroz014/plant-disease)]
[[download2](https://github.com/spMohanty/PlantVillage-Dataset)]



### EuroSAT
The EuroSAT dataset is based on Sentinel-2 satellite images covering 13 spectral bands and consisting out of 10 classes with in total 27,000 labeled and geo-referenced images.
The Sentinel-2 satellite images are openly and freely accessible provided in the Earth observation program Copernicus.
All colored images, **i.e., 27,000 images of 10 classes**, are used for cross-domain few-shot learning.

[[paper](https://arxiv.org/abs/1709.00029)]
[[code](https://github.com/phelber/EuroSAT)]
[[splits]()]
[[download](https://zenodo.org/records/7711810#.ZAm3k-zMKEA)]



### ISIC2018
The HAM10000 (“Human Against Machine with 10000 training images”) dataset consists of 10,015 dermatoscopic images which are released as a training set for academic machine learning purposes and are publicly available through the ISIC (International Skin Imaging Collaboration) archive.
All 10015 dermatoscopic images, **i.e., 10,015 images of 7 classes**, are used for cross-domain few-shot learning.

[[paper1](https://www.nature.com/articles/sdata2018161)]
[[paper2](https://arxiv.org/abs/1902.03368)]
[[code](https://challenge.isic-archive.com/)]
[[splits]()]
[[download](https://challenge.isic-archive.com/data/#2018)]



### ChestX-ray8
The ChestX-ray8 dataset is a chest X-ray database which comprises 108,948 frontal-view X-ray images of 32,717 unique patients with the text-mined eight disease image labels (where each image can have multi-labels), from the associated radiological reports using natural language processing.
Seven of the eight classes, **i.e., 25,848 images of 7 classes**, are used for cross-domain few-shot learning and each image has only one label.
Note that nowadays the ChestX-ray dataset comprises 112,120 frontal-view X-ray images of 30,805 unique patients with the text-mined fourteen disease image labels (where each image can have multi-labels), mined from the associated radiological reports using natural language processing.

[[paper](https://arxiv.org/abs/1705.02315)]
[[code](https://www.cc.nih.gov/drd/summers.html)]
[[splits]()]
[[download1](https://www.kaggle.com/datasets/nih-chest-xrays/data)]
[[download2](https://nihcc.app.box.com/v/ChestXray-NIHCC)]


