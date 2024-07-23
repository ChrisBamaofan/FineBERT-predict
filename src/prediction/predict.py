import torch
from transformers import BertTokenizer
import joblib
from src.models.model import MultiLabelModel
import os

def load_model(model_path, positive_mlb, negative_mlb, label_type_mlb, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path {model_path} does not exist.")
    
    model = MultiLabelModel(n_positive_classes=len(positive_mlb.classes_),
                            n_negative_classes=len(negative_mlb.classes_),
                            n_label_types=len(label_type_mlb.classes_),
                            local_model_path = model_path)
    model.load_state_dict(torch.load(os.path.join(model_path, 'pytorch_model.bin')))
    model.to(device)
    model.eval()
    return model

def predict(sentence, sentence_id, model, tokenizer, positive_mlb, negative_mlb, label_type_mlb, device):
    # 对输入句子进行编码
    encoding = tokenizer.encode_plus(
        sentence,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )

    # 将输入张量移动到指定设备（如GPU）
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # 在指定设备上进行模型预测
    with torch.no_grad():
        positive_outputs, negative_outputs, weight_output, label_type_output = model(input_ids, attention_mask)

    # 将输出张量移动到CPU进行后处理
    positive_preds = torch.sigmoid(positive_outputs).cpu().numpy()[0]
    negative_preds = torch.sigmoid(negative_outputs).cpu().numpy()[0]
    weight_pred = weight_output.cpu().numpy()[0]
    label_type_pred = torch.argmax(label_type_output).item()
    label_type = label_type_mlb.classes_[label_type_pred]

    # 应用阈值进行标签选择
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