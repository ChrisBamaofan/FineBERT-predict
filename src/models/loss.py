import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, positive_outputs, negative_outputs, weight_outputs, label_type_outputs,
                positive_labels, negative_labels, true_weights, true_label_types):
        positive_loss = self.bce_loss(positive_outputs, positive_labels).mean(dim=1)
        negative_loss = self.bce_loss(negative_outputs, negative_labels).mean(dim=1)
        label_loss = positive_loss + negative_loss
        weight_loss = self.mse_loss(weight_outputs, true_weights)
        label_type_loss = self.ce_loss(label_type_outputs, true_label_types)

        return label_loss.mean() + weight_loss + label_type_loss