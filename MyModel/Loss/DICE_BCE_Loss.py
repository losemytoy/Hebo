from torch import nn
import torch

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
        Args:
            smooth: A float number to smooth loss, and avoid NaN error, default: 1
            p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
            predict: A tensor of shape [N, *]
            target: A tensor of shape same with predict
            reduction: Reduction method to apply, return mean over batch if 'mean',
                return sum if 'sum', return a tensor of shape [N,] if 'none'
        Returns:
            Loss tensor according to arg reduction
        Raise:
            Exception if unexpected reduction
        """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))



class DICE_BCE_Loss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        intersection = 2 * (logits * targets).sum() + self.smooth
        union = (logits + targets).sum() + self.smooth
        dice_loss = 1. - intersection / union

        loss = nn.BCELoss()
        bce_loss = loss(logits, targets)

        return dice_loss + bce_loss


def dice_coeff(logits, targets):
    intersection = 2 * (logits * targets).sum()
    union = (logits + targets).sum()
    if union == 0:
        return 1
    dice_coeff = intersection / union
    return dice_coeff.item()