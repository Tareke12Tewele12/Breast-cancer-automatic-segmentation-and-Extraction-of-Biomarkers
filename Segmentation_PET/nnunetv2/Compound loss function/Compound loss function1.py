import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedWeightedLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, epsilon=0.5, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FN + self.beta * FP + self.smooth
        )

        ftl_loss = torch.pow((1 - tversky_index), self.gamma)
        bce_loss = F.binary_cross_entropy(inputs, targets)

        return self.epsilon * ftl_loss + (1 - self.epsilon) * bce_loss