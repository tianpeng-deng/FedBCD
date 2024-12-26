# FedBCD: Federated Ultrasound Video and Image Joint Learning for Breast Cancer Diagnosis


## Introduction
The implementation of our paper [FedBCD]. The paper is under review in IEEE Transactions on Medical Imaging



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
Our paper is under review in IEEE Transactions on Medical Imaging, the citation will be available after acceptance.
