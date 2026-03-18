import torch
from utils.active_learning import mc_dropout_uncertainty


def run_active_finetuning(trainer,
                          pretrained_checkpoint,
                          followup_cases,
                          active_pool_cases,
                          device):

    print("Loading pretrained model...")
    checkpoint = torch.load(pretrained_checkpoint, map_location=device)
    trainer.network.load_state_dict(checkpoint['network_weights'])

    # Phase 1
    print("Fine tuning on follow-up cases")
    trainer.dataset_tr = followup_cases
    trainer.run_training()

    # Phase 2: Active Learning
    print("Estimating uncertainty")
    uncertainty_scores = {}

    for case in active_pool_cases:
        data = trainer.dataset_val.load_case(case)
        image = torch.tensor(data['data']).unsqueeze(0).to(device)

        score = mc_dropout_uncertainty(trainer.network, image)
        uncertainty_scores[case] = score

    sorted_cases = sorted(uncertainty_scores.items(),
                          key=lambda x: x[1],
                          reverse=True)

    selected = [case for case, _ in sorted_cases[:3]]
    print("Selected cases:", selected)

    # Phase 3
    trainer.dataset_tr = followup_cases + selected

    for param in trainer.network.parameters():
        param.requires_grad = True

    trainer.run_training()

    print("Active fine tuning completed.")