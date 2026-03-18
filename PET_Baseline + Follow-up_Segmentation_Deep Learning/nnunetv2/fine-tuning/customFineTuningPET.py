import os
import torch
import numpy as np
from copy import deepcopy
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.file_path_utilities import get_output_folder


# ==============================
# CONFIGURATION
# ==============================

DATASET_NAME = "Dataset000_FollowUp"
CONFIGURATION = "3d_fullres"
FOLD = 0

PRETRAINED_CHECKPOINT = "/path/to/baseline/checkpoint_final.pth"

FOLLOWUP_CASES = [
    "case_001", "case_002", "case_003", "case_004",
    "case_005", "case_006", "case_007", "case_008",
    "case_009", "case_010", "case_011", "case_012"
]

ACTIVE_POOL_CASES = [
    "case_013", "case_014", "case_015",
    "case_016", "case_017", "case_018", "case_019"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MC_SAMPLES = 8
ACTIVE_SELECTION_K = 3  # how many uncertain cases to add


# ==============================
# INITIALIZE TRAINER
# ==============================

trainer = nnUNetTrainer(
    plans_file=None,  # auto detected
    configuration=CONFIGURATION,
    fold=FOLD,
    dataset_json=None
)

trainer.initialize()
trainer.network.to(DEVICE)


# ==============================
# LOAD PRETRAINED MODEL
# ==============================

print("Loading pretrained weights...")
checkpoint = torch.load(PRETRAINED_CHECKPOINT, map_location=DEVICE)
trainer.network.load_state_dict(checkpoint['network_weights'])
print("Loaded successfully.")


# ==============================
# OPTIONAL: FREEZE ENCODER
# ==============================

for name, param in trainer.network.named_parameters():
    if "encoder" in name:
        param.requires_grad = False

for param_group in trainer.optimizer.param_groups:
    param_group['lr'] = 1e-4


# ==============================
# PHASE 1: FINE TUNE ON 12 CASES
# ==============================

print("Phase 1: Fine tuning on 12 follow-up cases")

trainer.dataset_tr = FOLLOWUP_CASES
trainer.run_training()

print("Phase 1 completed.")


# ==============================
# PHASE 2: ACTIVE LEARNING
# ==============================

print("Phase 2: Active learning uncertainty estimation")


def mc_dropout_prediction(model, input_tensor, n_samples=8):
    model.train()  # enable dropout
    preds = []

    with torch.no_grad():
        for _ in range(n_samples):
            output = torch.softmax(model(input_tensor), dim=1)
            preds.append(output)

    preds = torch.stack(preds)
    mean_pred = preds.mean(0)
    variance = preds.var(0)

    return mean_pred, variance


uncertainty_scores = {}

for case in ACTIVE_POOL_CASES:
    print(f"Evaluating uncertainty for {case}")

    # Replace this with actual nnU-Net dataloader call
    data = trainer.dataset_val.load_case(case)
    image = torch.tensor(data['data']).unsqueeze(0).to(DEVICE)

    _, variance = mc_dropout_prediction(trainer.network, image, MC_SAMPLES)

    score = variance.mean().item()
    uncertainty_scores[case] = score


# Select most uncertain cases
sorted_cases = sorted(uncertainty_scores.items(), key=lambda x: x[1], reverse=True)
selected_cases = [case for case, _ in sorted_cases[:ACTIVE_SELECTION_K]]

print("Selected cases for annotation:", selected_cases)


# ==============================
# PHASE 3: ADD SELECTED CASES
# ==============================

print("Phase 3: Continue fine tuning with selected active cases")

extended_training_set = FOLLOWUP_CASES + selected_cases
trainer.dataset_tr = extended_training_set

for param in trainer.network.parameters():
    param.requires_grad = True  # unfreeze everything for final tuning

for param_group in trainer.optimizer.param_groups:
    param_group['lr'] = 5e-5

trainer.run_training()

print("Pipeline completed successfully.")