import torch
from trainers.custom_trainers import nnUNetTrainer_CustomLoss
from finetuning.active_finetune import run_active_finetuning
import config


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = nnUNetTrainer_CustomLoss(
        plans=None,
        configuration=config.CONFIGURATION,
        fold=config.FOLD,
        dataset_json=None,
        device=device
    )

    trainer.initialize()
    trainer.network.to(device)

    run_active_finetuning(
        trainer=trainer,
        pretrained_checkpoint=config.PRETRAINED_CHECKPOINT,
        followup_cases=config.FOLLOWUP_CASES,
        active_pool_cases=config.ACTIVE_POOL_CASES,
        device=device
    )


if __name__ == "__main__":
    main()