# FedBCD: Federated Ultrasound Video and Image Joint Learning for Breast Cancer Diagnosis


## Introduction
The implementation of our paper [FedBCD: Federated Ultrasound Video and Image Joint Learning for Breast Cancer Diagnosis](https://ieeexplore.ieee.org/document/10848115).

## Abstract
Ultrasonography plays an essential role in breast cancer diagnosis. Current deep learning based studies train the models on either images or videos in a centralized learning manner, lacking consideration of joint benefits between two different modality models or the privacy issue of data centralization. In this study, we propose the first decentralized learning solution for joint learning with breast ultrasound video and image, called FedBCD. To enable the model to learn from images and videos simultaneously and seamlessly in client-level local training, we propose a Joint Ultrasound Video and Image Learning (JUVIL) model to bridge the dimension gap between video and image data by incorporating temporal and spatial adapters. The parameter-efficient design of JUVIL with trainable adapters and frozen backbone further reduces the computational cost and communication burden of federated learning, finally improving the overall efficiency. Moreover, considering conventional model-wise aggregation may lead to unstable federated training due to different modalities, data capacities in different clients, and different functionalities across layers. We further propose a Fisher information matrix (FIM) guided Layer-wise Aggregation method named FILA. By measuring layer-wise sensitivity with FIM, FILA assigns higher contributions to the clients with lower sensitivity, improving personalized performance during federated training. Extensive experiments on three image clients and one video client demonstrate the benefits of joint learning architecture, especially for the ones with small-scale data. FedBCD significantly outperforms nine federated learning methods on both video-based and image-based diagnoses, demonstrating the superiority and potential for clinical practice.

## Usage
### Installation
- Download the repository.
```
git clone https://github.com/tianpeng-deng/FedBCD.git
```
- Our code is based on [PFLlib](https://github.com/TsingZ0/PFLlib), the requirements should be installed correctly.

### Dataset Preparation
- All images and videos should be put in *dataset/data* folder. For each fold, the train and test set are partitioned according to benignVSmalignant.csv. Then put train.csv and test.csv into each client's folder as shown below.
```
    |-- dataset
        |-- data
        |    |-- BUSI
        |    |-- GDPH
        |    |-- SYSUCC
        |    |-- TDSC
        |-- fold1
        |    |-- BUSI
        |    |    |-- train.csv
        |    |    |-- test.csv
        |    |-- GDPH
        |    |    |-- train.csv
        |    |    |-- test.csv
        |    |-- SYSUCC
        |    |    |-- train.csv
        |    |    |-- test.csv
        |    |-- TDSC
        |    |    |-- train.csv
        |    |    |-- test.csv
        |-- fold2

```



### Run
-  To use this example code, you can train and test the FedBCD by simply using the command:

```
cd FedBCD
bash script/run_ultrasound.sh
```


## Citation
If you find the code useful, please consider citing our paper using the following BibTeX entry.
```
@ARTICLE{deng2025fedbcd,
  author={Deng, Tianpeng and Huang, Chunwang and Cai, Ming and Liu, Yu and Liu, Min and Lin, Jiatai and Shi, Zhenwei and Zhao, Bingchao and Huang, Jingqi and Liang, Changhong and Han, Guoqiang and Liu, Zaiyi and Wang, Ying and Han, Chu},
  journal={IEEE Transactions on Medical Imaging}, 
  title={FedBCD: Federated Ultrasound Video and Image Joint Learning for Breast Cancer Diagnosis}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Federated learning;breast cancer diagnosis;joint learning;layer-wise aggregation},
  doi={10.1109/TMI.2025.3532474}}
```
