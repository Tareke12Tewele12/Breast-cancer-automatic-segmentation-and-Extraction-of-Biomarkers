import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedWeightedLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, epsilon=0.5, smooth=1e-6):
        """
        Combined Focal Tversky Loss (FTL) and Binary Cross-Entropy (BCE).
        
        Args:
            alpha: Penalty for False Negatives (FN).
            beta: Penalty for False Positives (FP).
            gamma: Focusing parameter for FTL.
            epsilon: Weight factor for FTL (1-epsilon for BCE).
            smooth: Small constant to avoid division by zero.
        """
        super(CombinedWeightedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # 1. Calculate Tversky Index (TI)
        # TP: True Positives, FP: False Positives, FN: False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        
        tversky_index = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)  
        
        # 2. Calculate Focal Tversky Loss (FTL)
        # FTL = (1 - TI)^gamma
        ftl_loss = torch.pow((1 - tversky_index), self.gamma)
        
        # 3. Calculate Binary Cross-Entropy (BCE)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        # 4. Weighted Combined Loss
        # Loss = epsilon * FTL + (1 - epsilon) * BCE
        combined_loss = (self.epsilon * ftl_loss) + ((1 - self.epsilon) * bce_loss)
        
        return combined_loss