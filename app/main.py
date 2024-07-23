from flask import Flask, request, jsonify
import json
from src.prediction.predict import predict, load_model
from transformers import BertTokenizer
import joblib
import torch

app = Flask(__name__)

local_model_path = "D:/workspace/llm/FinBERT_L-12_H-768_A-12_pytorch"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
positive_mlb = joblib.load('saved_models/positive_mlb.pkl')
negative_mlb = joblib.load('saved_models/negative_mlb.pkl')
label_type_mlb = joblib.load('saved_models/label_type_mlb.pkl')
model_path = local_model_path+'/saved_models'
model = load_model(model_path, positive_mlb, negative_mlb, label_type_mlb,device)
tokenizer = BertTokenizer.from_pretrained('saved_models')


@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    print('get data from request : '+data.get('sentence')+data.get('sentence_id'))
    sentence = data.get('sentence')
    sentence_id = data.get('sentence_id')
    result = predict(sentence, sentence_id, model, tokenizer, positive_mlb, negative_mlb, label_type_mlb, device)

    return json.dumps(result, ensure_ascii=False)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8008)