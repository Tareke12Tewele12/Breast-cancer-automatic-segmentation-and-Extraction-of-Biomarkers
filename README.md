
# Breast Cancer Automatic Segmentation and Biomarker Extraction

## Overview

This project presents a fully automated deep learning framework for breast cancer lesion segmentation and quantitative biomarker extraction from PET medical images. The system integrates advanced convolutional neural networks with radiomic feature analysis to support reproducible and objective tumor characterization.

The primary goal is to improve lesion detection accuracy in imbalanced datasets while enabling automated extraction of clinically relevant imaging biomarkers.

---

## Objectives

* Develop a robust deep learning model for PET-based tumor segmentation
* Address class imbalance using a combined Focal Tversky Loss and Binary Cross-Entropy loss
* Extract quantitative imaging biomarkers automatically from segmented regions
* Provide a reproducible pipeline for research and clinical evaluation

---

## Methodology

### 1. Segmentation Model

The segmentation framework is based on a 3D deep learning architecture trained on PET volumes.

To handle class imbalance and small lesion regions, a **combined weighted loss function** was implemented:

* Focal Tversky Loss (FTL)
* Binary Cross-Entropy (BCE)

The combined loss is defined as:

Loss = ε × FTL + (1 − ε) × BCE

Where:

* α and β control false negative and false positive penalties
* γ focuses learning on difficult-to-segment regions
* ε balances FTL and BCE contributions

This design improves sensitivity to small tumor regions and stabilizes optimization.

---

### 2. Biomarker Extraction

After segmentation, quantitative biomarkers are extracted automatically, including:

* SUV-based metrics

  * SUVmax
  * SUVmean
  * SUVpeak
* Metabolic Tumor Volume (MTV)
* Total Lesion Glycolysis (TLG)
* Shape and intensity statistics
* Texture features

These biomarkers can be used for:

* Prognostic modeling
* Treatment response evaluation
* Radiomic-based classification

---

## Project Structure

```
Segmentation_PET/
│
├── models/                 # Network architectures
├── losses/                 # Custom combined loss functions
├── training/               # Training scripts
├── inference/              # Inference and prediction scripts
├── biomarker_extraction/   # Radiomic feature extraction
├── nnUNet_results/         # Model outputs (excluded from Git)
└── utils/                  # Helper functions
```

Large checkpoint files are excluded from the repository. Pretrained models can be provided upon request.

---

## Technologies Used

* Python
* PyTorch
* nnUNet framework
* NumPy
* SimpleITK
* PyRadiomics
* Scikit-learn

---

## Key Features

* Fully automated 3D segmentation pipeline
* Class imbalance handling with advanced loss formulation
* Automated PET biomarker computation
* Modular design for research extension
* Compatible with NIfTI medical imaging format

---

## Installation

```bash
git clone https://github.com/Tareke12Tewele12/Breast-cancer-automatic-segmentation-and-Extraction-of-Biomarkers.git
cd Breast-cancer-automatic-segmentation-and-Extraction-of-Biomarkers
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python train.py
```

### Inference

```bash
python inference.py
```

### Biomarker Extraction

```bash
python extract_biomarkers.py
```

---

## Research Applications

This framework is suitable for:

* Medical image segmentation research
* Radiomics analysis
* Deep learning benchmarking
* Clinical decision support development

---

## Future Improvements

* Multi-modal PET/CT integration
* Cross-validation experiments
* Transformer-based segmentation models
* External dataset validation

---

## Author

Tewele Weletnsea Tareke
Medical Imaging and Deep Learning Researcher


