
<p align="center">
  <h1 align="center">A Voronoi Density based Locally Unique Classification Network for Fine-grained Mulitilabel Learning</h1>
  <p align="center">


   <br />
    <strong>Binghao Liu</strong></a>
    ·
    <strong>Qi Zhao</strong></a>
    ·
    <strong>Chunlei Wang</strong></a>
    ·
    <strong>Meng Li</strong></a>     
    ·
    <strong>Lijiang Chen</strong></a>
    <br />
<p align="center">
 </p>





## Introduction
This repo is the implementation of "A Voronoi Density based Locally Unique Classification Network for Fine-grained Mulitilabel Learning"

<p align="center">
  <img src="images/VoLUNet.png" width="720">
</p>

Voronoi Density based Superpixel Generation Module:

<p align="center">
  <img src="images/VDSM.png" width="360">
</p>

## Overview

+ `helper_functions/` includes helper functions and dataloader files
+ `loss_functions/` includes loss computation files
+ `models/` includes related model and module

## Usage

### Dataset

Please prepare related datasets: 

- TreeSatAI: [https://zenodo.org/records/6780578](https://zenodo.org/records/6780578)
- GeoLifeCLEF: [https://www.kaggle.com/competitions/geolifeclef-2024](https://www.kaggle.com/competitions/geolifeclef-2024)
- FathomNet: [https://www.kaggle.com/competitions/fathomnet-out-of-sample-detection](https://www.kaggle.com/competitions/fathomnet-out-of-sample-detection)

The data split files of these datasets can be found at [https://drive.google.com/drive/folders/1Bceu-wIOO4q3Q5wChkF8LmUc0A9YVgZq?usp=sharing](https://drive.google.com/drive/folders/1Bceu-wIOO4q3Q5wChkF8LmUc0A9YVgZq?usp=sharing)

### Train and Test

+ Use the following command for training with TResNet-L

  ```
  python train.py \
  --img_dir=path_of_images \
  --csv_path=path_of_csv_file \
  --backbone-name=tresnet_l \
  --image-size=224 \
  --model-path=path_of_pretrained_weights \
  --output-path=path_to_save_checkpoints \
  --subsampling \
  --sub-h=4 \
  --sub-w=4 \
  --min-superpixels=6
  ```

+ Use the following command for testing with TResNet-L

  ```
  python val.py \
  --img_dir=path_of_images \
  --csv_path=path_of_csv_file \
  --backbone-name=tresnet_l \
  --ckpt-path=path_of_model_weights
  ```

## Citation

If you have any question, please discuss with me by sending email to liubinghao@buaa.edu.cn

## References

The code is based on [ASL](https://github.com/Alibaba-MIIL/ASL) and [efficient-kan](https://github.com/Blealtan/efficient-kan). Thanks for their great works!
