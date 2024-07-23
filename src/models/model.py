import torch.nn as nn
from transformers import BertModel

class MultiLabelModel(nn.Module):
    def __init__(self, n_positive_classes, n_negative_classes, n_label_types, local_model_path):
        super(MultiLabelModel, self).__init__()
        self.bert = BertModel.from_pretrained(local_model_path)
        self.dropout = nn.Dropout(p=0.3)
        
        self.positive_classifier = nn.Linear(self.bert.config.hidden_size, n_positive_classes)
        self.negative_classifier = nn.Linear(self.bert.config.hidden_size, n_negative_classes)
        self.weight_predictor = nn.Linear(self.bert.config.hidden_size, 1)
        self.label_type_classifier = nn.Linear(self.bert.config.hidden_size, n_label_types)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        output = self.dropout(pooled_output)

        positive_output = self.positive_classifier(output)
        negative_output = self.negative_classifier(output)
        weight_output = self.weight_predictor(output).squeeze(-1)
        label_type_output = self.label_type_classifier(output)

        return positive_output, negative_output, weight_output, label_type_output
