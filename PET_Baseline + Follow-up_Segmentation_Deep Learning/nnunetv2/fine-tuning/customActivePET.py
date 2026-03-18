import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from losses.combined_loss import CombinedWeightedLoss


class nnUNetTrainer_CustomLoss(nnUNetTrainer):

    def __init__(self, plans, configuration, fold, dataset_json,
                 unpack_dataset=True, device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json,
                         unpack_dataset, device)

        self.initial_lr = 2e-4
        self.num_epochs = 500

    def _build_loss(self):
        return CombinedWeightedLoss()
    


def mc_dropout_uncertainty(model, input_tensor, n_samples=8):
    model.train()
    predictions = []

    with torch.no_grad():
        for _ in range(n_samples):
            output = torch.softmax(model(input_tensor), dim=1)
            predictions.append(output)

    predictions = torch.stack(predictions)
    variance = predictions.var(dim=0)

    return variance.mean().item()
    