"""
==========================================================
Longitudinal PET Biomarker Extraction Script
Extracts SUVmax, MTV, TLG from 12 baseline and follow-up cases
Computes absolute and percentage change
Generates boxplots
Saves results to CSV
==========================================================
"""

import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon

# ============================
# 1. USER SETTINGS
# ============================

BASELINE_DIR = "baseline"
FOLLOWUP_DIR = "followup"
N_CASES = 12

OUTPUT_CSV = "biomarker_results.csv"
BOXPLOT_FILE = "biomarker_boxplot.png"

# ============================
# 2. BIOMARKER EXTRACTION FUNCTION
# ============================

def extract_biomarkers(pet_path, mask_path):
    """
    Extract SUVmax, MTV (mL), TLG
    """

    # Load NIfTI images
    pet_img = nib.load(pet_path)
    mask_img = nib.load(mask_path)

    pet = pet_img.get_fdata()
    mask = mask_img.get_fdata()

    # Ensure binary mask
    mask = (mask > 0).astype(np.uint8)

    tumor_voxels = pet[mask == 1]

    if len(tumor_voxels) == 0:
        return np.nan, np.nan, np.nan

    # SUVmax
    suv_max = np.max(tumor_voxels)

    # SUVmean
    suv_mean = np.mean(tumor_voxels)

    # Voxel volume calculation
    voxel_dims = pet_img.header.get_zooms()[:3]
    voxel_volume_mm3 = np.prod(voxel_dims)
    voxel_volume_ml = voxel_volume_mm3 / 1000.0

    # MTV
    mtv = np.sum(mask) * voxel_volume_ml

    # TLG
    tlg = suv_mean * mtv

    return suv_max, mtv, tlg


# ============================
# 3. PROCESS ALL CASES
# ============================

results = []

case_ids = [f"case{str(i).zfill(2)}" for i in range(1, N_CASES + 1)]

for case in case_ids:

    print(f"Processing {case}...")

    # Baseline paths
    base_pet = os.path.join(BASELINE_DIR, f"{case}_pet.nii.gz")
    base_mask = os.path.join(BASELINE_DIR, f"{case}_mask.nii.gz")

    # Follow-up paths
    follow_pet = os.path.join(FOLLOWUP_DIR, f"{case}_pet.nii.gz")
    follow_mask = os.path.join(FOLLOWUP_DIR, f"{case}_mask.nii.gz")

    # Extract baseline biomarkers
    suv_b, mtv_b, tlg_b = extract_biomarkers(base_pet, base_mask)

    # Extract follow-up biomarkers
    suv_f, mtv_f, tlg_f = extract_biomarkers(follow_pet, follow_mask)

    results.append({
        "Case": case,

        "SUVmax_baseline": suv_b,
        "MTV_baseline": mtv_b,
        "TLG_baseline": tlg_b,

        "SUVmax_followup": suv_f,
        "MTV_followup": mtv_f,
        "TLG_followup": tlg_f
    })

# Convert to DataFrame
df = pd.DataFrame(results)

# ============================
# 4. COMPUTE LONGITUDINAL CHANGES
# ============================

# Absolute change
df["Delta_SUVmax"] = df["SUVmax_followup"] - df["SUVmax_baseline"]
df["Delta_MTV"] = df["MTV_followup"] - df["MTV_baseline"]
df["Delta_TLG"] = df["TLG_followup"] - df["TLG_baseline"]

# Percentage change
df["Perc_Delta_SUVmax"] = 100 * df["Delta_SUVmax"] / df["SUVmax_baseline"]
df["Perc_Delta_MTV"] = 100 * df["Delta_MTV"] / df["MTV_baseline"]
df["Perc_Delta_TLG"] = 100 * df["Delta_TLG"] / df["TLG_baseline"]

# ============================
# 5. STATISTICAL TESTS (Optional)
# ============================

print("\nPaired t-test Results:")
print("SUVmax:", ttest_rel(df["SUVmax_baseline"], df["SUVmax_followup"]))
print("MTV:", ttest_rel(df["MTV_baseline"], df["MTV_followup"]))
print("TLG:", ttest_rel(df["TLG_baseline"], df["TLG_followup"]))

print("\nWilcoxon Test Results:")
print("SUVmax:", wilcoxon(df["SUVmax_baseline"], df["SUVmax_followup"]))
print("MTV:", wilcoxon(df["MTV_baseline"], df["MTV_followup"]))
print("TLG:", wilcoxon(df["TLG_baseline"], df["TLG_followup"]))

# ============================
# 6. BOXPLOT OF ABSOLUTE CHANGES
# ============================

melted = df.melt(
    value_vars=["Delta_SUVmax", "Delta_MTV", "Delta_TLG"],
    var_name="Biomarker",
    value_name="Absolute Change"
)

plt.figure(figsize=(8,6))
sns.boxplot(data=melted, x="Biomarker", y="Absolute Change")
sns.stripplot(data=melted, x="Biomarker", y="Absolute Change", color="black", alpha=0.6)
plt.title("Biomarker Changes (Baseline vs Follow-up, n=12)")
plt.tight_layout()
plt.savefig(BOXPLOT_FILE, dpi=300)
plt.show()

# ============================
# 7. SAVE RESULTS
# ============================

df.to_csv(OUTPUT_CSV, index=False)

print("\nFinished successfully.")
print(f"Results saved to: {OUTPUT_CSV}")
print(f"Boxplot saved to: {BOXPLOT_FILE}")