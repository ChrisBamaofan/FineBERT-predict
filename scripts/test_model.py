from transformers import BertTokenizer
from sklearn.preprocessing import MultiLabelBinarizer
from src.data.dataset import CustomDataset
from src.prediction.predict import test_model
from src.models.model import MultiLabelModel
from src.prediction.predict import load_model
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import joblib

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

if __name__ == '__main__':
    
    local_model_path = "D:/workspace/llm/FinBERT_L-12_H-768_A-12_pytorch"
    file_path_test = 'trainDataset/test.txt'

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    positive_mlb = joblib.load('saved_models/positive_mlb.pkl')
    negative_mlb = joblib.load('saved_models/negative_mlb.pkl')
    label_type_mlb = joblib.load('saved_models/label_type_mlb.pkl')
    model_path = local_model_path+'/saved_models'
    model = load_model(model_path, positive_mlb, negative_mlb, label_type_mlb,device)
    tokenizer = BertTokenizer.from_pretrained('saved_models')

    test_dataset = CustomDataset(file_path_test, tokenizer, max_len=128, positive_mlb=positive_mlb,negative_mlb=negative_mlb, label_type_mlb=label_type_mlb, is_test=True)
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