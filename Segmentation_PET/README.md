# autoPET Segmentation

This project focuses on **Automated Lesion Segmentation in PET imaging**. It is developed as part of the European Union’s **Marie Skłodowska-Curie Doctoral Network Actions** (HORIZON-MSCA-2021-DN-01), under grant agreement No. 101073222.

## Repository Overview
This GitHub repository contains the following components:
* **`compound_loss.py`**: Implementation of the compound loss function.
* **`process.py`**: Scripts for the inference and segmentation process.
 * **`Biomarkers_change`** biomarkers extractions and changes
* **`requirements.txt`**: List of necessary dependencies and libraries.
* **Data Samples**: Examples of raw PET images and their corresponding segmented outputs.

## Associated Paper
**Title:** [Insert Full Paper Title Here]  
**Authors:** Tewele W. Tareke (1), Nérée Payan (1,2), Alexandre Cochet (1,2), Laurent Arnould (3), Benoit Presles (1), Jean-Marc Vrigneaud (1,2), Fabrice Meriaudeau (1), Alain Lalande (1).  
**Link:** [https://arxiv.org/abs/2502.04083](https://arxiv.org/abs/2502.04083)

## Bench mark nnU-Net Model Weights
The trained model weights are available via Google Drive:  
[Access Model Weights Here](https://drive.google.com/drive/u/2/folders/13OBjcDwZmER5vD73jyCqx-0Ie83ceJdA)


.......

Baseline model
↓
Fine tune on 12 follow up (Strategy A)
↓
Baseline model
↓
Active learning selection (7 cases)
↓
Fine tune (Strategy B)
↓
Compare Dice, IoU, HD95

## This may help to understand the project orchestration

│
├── main.py
├── config.py
│
├── losses/
│   └── combined_loss.py
│
├── trainers/
│   └── custom_trainers.py
│
├── preprocessing/
│   └── custom_preprocessing.py
│
├── finetuning/
│   └── active_finetune.py
│
└── utils/
    └── active_learning.py
---

## Contact
**Dr. Tewele Weletnsea Tareke** University of Burgundy, France  
**Email:** tewele-weletnsea.tareke@u-bourgogne.fr | tewetyy@gmail.com
