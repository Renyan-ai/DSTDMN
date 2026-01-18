# DSTDMN
This is the pytorch implementation of DSTDMN. I hope these codes are helpful to you!

The architecture of the **Dynamic Spatio-Temporal Decoupling and Memory-Augmented Spatiotemporal Network (DSTDMN)** is illustrated below:

<p align="center">
  <img src="https://github.com/user-attachments/assets/a425559f-eee5-4a0c-ae06-ad76f4172bad" alt="DSTDMN Framework" width="100%">
  <br>
  <em>Figure: Overview of the DSTDMN Framework.</em>
</p>



## Datasets
The traffic datasets used in this project are publicly available. You can download the integrated package containing PEMS03, PEMS04, PEMS07, and PEMS08 from the following link:
- **Primary Source (Kaggle):** [https://www.kaggle.com/datasets/elmahy/pems-dataset](https://www.kaggle.com/datasets/elmahy/pems-dataset)


## Train Commands
It's easy to run! Here are some examples, and you can customize the model settings in train.py.
### PEMS08
```
python -u train.py --data PEMS08 > PEMS08.log &
```

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- PyTorch
- NumPy, Pandas, Scikit-learn
