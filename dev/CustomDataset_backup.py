import pandas as pd
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MultiLabelBinarizer

local_model_path = "D:\\workspace\\llm\\FinBERT_L-12_H-768_A-12_pytorch"

class CustomDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len, positive_mlb=None, negative_mlb=None, label_type_mlb=None, is_test=False):
        with open(file_path, 'r', encoding='utf-8') as file:
            self.data = file.readlines()

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_test = is_test

        if not is_test:
            self.positive_mlb = positive_mlb
            self.negative_mlb = negative_mlb
            self.label_type_mlb = label_type_mlb

            all_positive_labels = []
            all_negative_labels = []
            all_label_types = []
            for line in self.data:
                sentence, positive_labels, negative_labels, weight, label_type = self.parse_line(line)
                all_positive_labels.append(positive_labels)
                all_negative_labels.append(negative_labels)
                all_label_types.append([label_type])

            self.positive_mlb.fit(all_positive_labels)
            self.negative_mlb.fit(all_negative_labels)
            self.label_type_mlb.fit(all_label_types)
        else:
            self.positive_mlb = positive_mlb
            self.negative_mlb = negative_mlb
            self.label_type_mlb = label_type_mlb

    def parse_line(self, line):
        line = line.strip()
        parts = line.split('#')
        sentence = parts[0].strip()
        weight = int(parts[3].strip()) if len(parts) > 3 else 1  # 默认权重为1
        label_type = parts[4].strip() if len(parts) > 4 else ""
        if self.is_test:
            return sentence, [], [], weight, label_type
        positive_labels = parts[1].strip().split(',') if len(parts) > 1 and parts[1] else []
        negative_labels = parts[2].strip().split(',') if len(parts) > 2 and parts[2] else []
        return sentence, positive_labels, negative_labels, weight, label_type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        sentence, positive_labels, negative_labels, weight, label_type = self.parse_line(line)

        encoding = self.tokenizer.encode_plus(
            sentence,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'sentence': sentence,
            'weight': torch.tensor(weight, dtype=torch.float)
        }

        if not self.is_test:
            positive_labels = self.positive_mlb.transform([positive_labels])[0]
            negative_labels = self.negative_mlb.transform([negative_labels])[0]
            label_type = self.label_type_mlb.transform([[label_type]])[0]
            item['positive_labels'] = torch.tensor(positive_labels, dtype=torch.float)
            item['negative_labels'] = torch.tensor(negative_labels, dtype=torch.float)
            item['label_type'] = torch.tensor(label_type, dtype=torch.float)

        return item

    def get_label_names(self, binary_labels, mlb):
        return mlb.inverse_transform([binary_labels])[0]

positive_mlb = MultiLabelBinarizer()
negative_mlb = MultiLabelBinarizer()
label_type_mlb = MultiLabelBinarizer()
file_path = '../trainDataset/train.txt'

tokenizer = BertTokenizer.from_pretrained(local_model_path)

train_dataset = CustomDataset(file_path, tokenizer, max_len=128, positive_mlb=positive_mlb, negative_mlb=negative_mlb,
                              label_type_mlb=label_type_mlb, is_test=False)

loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

import torch.nn as nn
from transformers import BertModel
import torch

class MultiLabelModel(nn.Module):
    def __init__(self, n_positive_classes, n_negative_classes, n_label_types):
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

positive_classes = len(train_dataset.positive_mlb.classes_)
negative_classes = len(train_dataset.negative_mlb.classes_)
label_type_classes = len(train_dataset.label_type_mlb.classes_)
model = MultiLabelModel(n_positive_classes=positive_classes, n_negative_classes=negative_classes, n_label_types=label_type_classes)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

from transformers import AdamW, get_linear_schedule_with_warmup
import torch.optim as optim
import torch
import numpy as np

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

def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        positive_labels = d["positive_labels"].to(device)
        negative_labels = d["negative_labels"].to(device)
        true_weights = d["weight"].to(device)
        true_label_types = d["label_type"].to(device)

        positive_outputs, negative_outputs, weight_outputs, label_type_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = loss_fn(positive_outputs, negative_outputs, weight_outputs, label_type_outputs,
                       positive_labels, negative_labels, true_weights, true_label_types)

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            positive_labels = d["positive_labels"].to(device)
            negative_labels = d["negative_labels"].to(device)
            weights = d["weight"].to(device)
            label_types = d["label_type"].to(device)

            positive_outputs, negative_outputs, weight_outputs, label_type_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = loss_fn(positive_outputs, negative_outputs, weight_outputs, label_type_outputs,
                           positive_labels, negative_labels, weights, label_types)

            losses.append(loss.item())

    return np.mean(losses)

EPOCHS = 5
optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
loss_fn = CombinedLoss().to(device)

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 10)

    train_loss = train_epoch(
        model,
        loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(train_dataset)
    )

    print(f'Train loss {train_loss}')

def test_model(model, data_loader, device, dataset):
    model = model.eval()
    sentences = []
    positive_predictions = []
    negative_predictions = []
    weight_predictions = []
    label_type_predictions = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)

            positive_outputs, negative_outputs, weight_outputs, label_type_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            positive_preds = torch.sigmoid(positive_outputs).cpu().numpy()
            negative_preds = torch.sigmoid(negative_outputs).cpu().numpy()
            weight_preds = weight_outputs.cpu().numpy()
            label_type_preds = torch.argmax(label_type_outputs, dim=1).cpu().numpy()

            sentences.extend(d["sentence"])
            positive_predictions.extend(positive_preds)
            negative_predictions.extend(negative_preds)
            weight_predictions.extend(weight_preds)
            label_type_predictions.extend(label_type_preds)

    return sentences, positive_predictions, negative_predictions, weight_predictions, label_type_predictions

file_path_test = '../trainDataset/test.txt'
tokenizer = BertTokenizer.from_pretrained(local_model_path)
test_dataset = CustomDataset(file_path_test, tokenizer, max_len=128, positive_mlb=positive_mlb,
                             negative_mlb=negative_mlb, label_type_mlb=label_type_mlb, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Run the test model function
sentences, positive_predictions, negative_predictions, weight_predictions, label_type_predictions = test_model(model, test_loader, device, test_dataset)

# Convert predictions to binary format
threshold = 0.5
positive_binary_predictions = (np.array(positive_predictions) > threshold).astype(int)
negative_binary_predictions = (np.array(negative_predictions) > threshold).astype(int)

# Convert binary predictions to label names and format the output
results = []
for i in range(len(sentences)):
    positive_labels = test_dataset.positive_mlb.classes_[positive_binary_predictions[i].astype(bool)]
    negative_labels = test_dataset.negative_mlb.classes_[negative_binary_predictions[i].astype(bool)]
    predicted_weight = int(round(weight_predictions[i]))  # 四舍五入到最近的整数
    predicted_label_type = test_dataset.label_type_mlb.classes_[label_type_predictions[i]]
    results.append(
        f"{sentences[i]}#positive_labels:{','.join(positive_labels)}#negative_labels:{','.join(negative_labels)}#{predicted_weight}#{predicted_label_type}")

# Output the results
for result in results:
    print(result)

import os

# 创建保存模型的目录
os.makedirs('saved_models', exist_ok=True)

# 保存模型
torch.save(model.state_dict(), 'saved_models/model.pth')

# 保存tokenizer
tokenizer.save_pretrained('saved_models')

# 保存标签二值化器
import joblib

joblib.dump(positive_mlb, 'saved_models/positive_mlb.pkl')
joblib.dump(negative_mlb, 'saved_models/negative_mlb.pkl')
joblib.dump(label_type_mlb, 'saved_models/label_type_mlb.pkl')

import torch
from transformers import BertTokenizer
import joblib

def load_model():
    model = MultiLabelModel(n_positive_classes=len(positive_mlb.classes_),
                            n_negative_classes=len(negative_mlb.classes_),
                            n_label_types=len(label_type_mlb.classes_))
    model.load_state_dict(torch.load('saved_models/model.pth'))
    model.eval()
    return model

def predict(sentence, sentence_id):
    model = load_model()
    tokenizer = BertTokenizer.from_pretrained('saved_models')
    positive_mlb = joblib.load('saved_models/positive_mlb.pkl')
    negative_mlb = joblib.load('saved_models/negative_mlb.pkl')
    label_type_mlb = joblib.load('saved_models/label_type_mlb.pkl')

    encoding = tokenizer.encode_plus(
        sentence,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    with torch.no_grad():
        positive_outputs, negative_outputs, weight_output, label_type_output = model(input_ids, attention_mask)

    positive_preds = torch.sigmoid(positive_outputs).cpu().numpy()[0]
    negative_preds = torch.sigmoid(negative_outputs).cpu().numpy()[0]
    weight_pred = weight_output.cpu().numpy()[0]
    label_type_pred = torch.argmax(label_type_output).item()
    label_type = label_type_mlb.classes_[label_type_pred]

    threshold = 0.5
    positive_labels = positive_mlb.classes_[(positive_preds > threshold).astype(bool)]
    negative_labels = negative_mlb.classes_[(negative_preds > threshold).astype(bool)]

    return {
        'sentence': sentence,
        'positive_labels': ','.join(positive_labels),
        'negative_labels': ','.join(negative_labels),
        'label_type': label_type,
        'weight': int(round(weight_pred)),
        'sentence_id': sentence_id
    }


from flask import Flask, request, jsonify
import json

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    print('get data from request : '+data.get('sentence')+data.get('sentence_id'))
    sentence = data.get('sentence')
    sentence_id = data.get('sentence_id')
    result = predict(sentence, sentence_id)

    return json.dumps(result, ensure_ascii=False)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8008)
