# CoSEF-DBP

### ✨ Introduction

This repository contains the code and data for the paper titled "CoSEF-DBP: Convolution scope expanding fusion network for identifying DNA-binding proteins through bilingual representations." The paper introduces a novel end-to-end method for identifying DNA-binding proteins (DBPs) without relying on complex feature engineering. It innovatively enriches the semantics of amino acid sequences through the fusion of bilingual representations derived from distinct language models. We further designed a convolution scope expanding (CoSE) module to widen the receptive fields of convolution kernels, thereby forming protein-level CoSE representation sequences. These representations are subsequently integrated via BiLSTM in conjunction with a simplified capsule network, enhancing the hierarchical feature extraction capability. Extensive experiments confirm that our model surpasses existing baseline models across various benchmark datasets, notably achieving at least a 5.1% improvement in MCC value on the UniSwiss dataset.

This README provides an overview of the repository and instructions for running the code and using the data.

### ⚙️ Setup

To run the code in this repository, you'll need the following dependencies:

- Python 3.9
- PyTorch 2.2
- transformers

### 🤖 Download

Before training and testing, you need to download the [dataset](https://drive.google.com/file/d/1oKWI-R6XjHYP0Uq6LzoxyBC2UHzDWBxW/view?usp=sharing) and place it in the `./data` directory.

Before executing the code, you need to download the pre-trained model [esm2_t12_35M_UR50D](https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t12_35M_UR50D.pt) , [pubmedbert ](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext)and place them in the `./esm `and `./pubmedbert` directory.


### ⚡️ Running the Code

- Model Training:

```
  python model.py
```

- Model Testing:

  Download the [model](https://drive.google.com/file/d/1bdNX8P9mX2A0XI-7gtxZfDVMwXgnwxu7/view?usp=sharing) and place it in the `./save_dict` directory.

```
  python test.py
```
