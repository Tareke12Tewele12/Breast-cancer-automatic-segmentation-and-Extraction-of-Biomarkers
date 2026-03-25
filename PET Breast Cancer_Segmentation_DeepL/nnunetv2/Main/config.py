DATASET_NAME = "Dataset000_FollowUp"
CONFIGURATION = "3d_fullres"
FOLD = 5

PRETRAINED_CHECKPOINT = "/path/to/checkpoint_final.pth"

FOLLOWUP_CASES = [f"case_{i:03d}" for i in range(1, 13)]
ACTIVE_POOL_CASES = [f"case_{i:03d}" for i in range(13, 20)]