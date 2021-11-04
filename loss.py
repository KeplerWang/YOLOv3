import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_class = 10
        self.lambda_no_obj = 20
        self.lambda_obj = 20
        self.lambda_box = 2

    def forward(self, predictions, target, anchors):
        obj = target[..., 0] == 1  
        no_obj = target[..., 0] == 0

        no_object_loss = F.binary_cross_entropy_with_logits(predictions[..., 0:1][no_obj], target[..., 0:1][no_obj])

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        object_loss = F.binary_cross_entropy_with_logits(predictions[..., 0:1][obj], target[..., 0:1][obj])

        predictions[..., 1:3] = predictions[..., 1:3].sigmoid()
        target[..., 3:5] = torch.log((1e-16 + target[..., 3:5] / anchors))
        box_loss = F.mse_loss(predictions[..., 1:5][obj], target[..., 1:5][obj])

        class_loss = F.cross_entropy(predictions[..., 5:][obj], target[..., 5][obj].long())

        return (
                self.lambda_box * box_loss
                + self.lambda_obj * object_loss
                + self.lambda_no_obj * no_object_loss
                + self.lambda_class * class_loss
        )

