
# PETLesionSeg – PET (Baseline + Follow-up) Segmentation

## Overview

This project focuses on **automated lesion segmentation in PET imaging** and the **extraction of clinically relevant biomarkers** for breast cancer analysis. The framework is designed to handle longitudinal PET data, including both **baseline (BL)** and **follow-up (FU)** scans.

The work is developed within the context of the **Marie Skłodowska-Curie Doctoral Network (MSCA-DN)** under the European Union Horizon program (Grant Agreement No. 101073222).

The main objective is to build a robust and reproducible deep learning pipeline that improves lesion detection performance in imbalanced datasets and enables quantitative analysis of tumor progression.

---

## Key Contributions

* Automated 3D PET lesion segmentation using deep learning
* Handling of class imbalance using a compound loss function
* Integration of baseline and follow-up imaging for longitudinal analysis
* Extraction of quantitative biomarkers from segmented lesions
* Evaluation using clinically relevant metrics (Dice, IoU, HD95)
* Active learning strategy to improve model performance with limited data

---

## Repository Overview

This repository includes the following main components:

* **`compound_loss.py`**
  Implements a combined loss function based on:

  * Focal Tversky Loss (FTL)
  * Binary Cross-Entropy (BCE)
    This helps the model learn better from imbalanced lesion data.

* **`process.py`**
  Contains scripts for:

  * Model inference
  * Segmentation prediction
  * Post-processing of outputs

* **`Biomarkers_change/`**
  Includes scripts for:

  * Extraction of imaging biomarkers
  * Computation of changes between baseline and follow-up scans

* **`requirements.txt`**
  Lists all required Python dependencies for reproducibility.

* **Data Samples**
  Example PET images and corresponding segmentation outputs for demonstration purposes.

---

## Methodology

### 1. Segmentation Framework

The model is based on a deep learning segmentation architecture trained on PET volumes.

To address class imbalance and small lesion regions, a **compound loss function** is used:

Loss = ε × FTL + (1 − ε) × BCE

Where:

* FTL focuses on difficult-to-segment regions
* BCE stabilizes training
* ε balances the contribution of both losses

---

### 2. Training Strategy

The project follows a multi-stage training and refinement pipeline:

```
Baseline model
   ↓
Fine-tuning on 12 follow-up cases (Strategy A)
   ↓
Baseline model reused
   ↓
Active learning selection (7 informative cases)
   ↓
Fine-tuning (Strategy B)
   ↓
Performance comparison (Dice, IoU, HD95)
```

This approach improves generalization while minimizing annotation effort.

---

### 3. Biomarker Extraction

After segmentation, the framework extracts important quantitative biomarkers such as:

* SUVmax, SUVmean, SUVpeak
* Metabolic Tumor Volume (MTV)
* Total Lesion Glycolysis (TLG)
* Shape-based features
* Intensity and texture features

It also evaluates **biomarker changes between baseline and follow-up scans**, which is important for treatment monitoring.

---

## Project Structure

```
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
├── utils/
│   └── active_learning.py
```

This modular structure allows easy experimentation and extension.

---

## Model Weights

Pretrained nnU-Net model weights are available here:

[https://drive.google.com/drive/u/2/folders/13OBjcDwZmER5vD73jyCqx-0Ie83ceJdA](https://drive.google.com/drive/u/2/folders/13OBjcDwZmER5vD73jyCqx-0Ie83ceJdA)

These weights can be used directly for inference or fine-tuning.

---

## Associated Paper

**Title:** (Add your final paper title here)

**Authors:**
Tewele W. Tareke, Nérée Payan, Alexandre Cochet, Laurent Arnould, Benoit Presles, Jean-Marc Vrigneaud, Fabrice Meriaudeau, Alain Lalande

**Preprint:**
[https://arxiv.org/abs/2502.04083](https://arxiv.org/abs/2502.04083)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Tareke12Tewele12/Breast-cancer-automatic-segmentation-and-Extraction-of-Biomarkers.git
cd Breast-cancer-automatic-segmentation-and-Extraction-of-Biomarkers
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

```
evalutils==0.4.2
nibabel==5.1.0
pillow==9.4.0
numpy==1.24.4
scikit-image==0.19.3
scipy==1.10.1
click==8.1.5
batchgenerators==0.23
pandas==1.5.3
matplotlib==3.7.2
seaborn==0.12.2
```

---

## Evaluation Metrics

The model performance is evaluated using:

* Dice Similarity Coefficient
* Intersection over Union (IoU)
* Hausdorff Distance (HD95)

These metrics provide a comprehensive evaluation of segmentation quality.

---

## Applications

This framework can be used for:

* Breast cancer lesion segmentation
* Longitudinal PET analysis
* Radiomics research
* Treatment response assessment
* Clinical decision support systems

---

## Contact

Tewele Weletnsea Tareke (ph.D)
University of Burgundy, France

Email:
[tewele-weletnsea.tareke@u-bourgogne.fr](mailto:tewele-weletnsea.tareke@u-bourgogne.fr)
[tewetyy@gmail.com](mailto:tewetyy@gmail.com)

* Make a shorter version for recruiters

Just tell me.
