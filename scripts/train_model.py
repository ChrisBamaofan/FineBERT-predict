import torch
from transformers import BertTokenizer, BertConfig

from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
from src.models.model import MultiLabelModel
from src.models.loss import CombinedLoss
from src.training.train import train_model
from transformers import AdamW, get_linear_schedule_with_warmup
from src.data.dataset import CustomDataset
import joblib

import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

if __name__ == '__main__':
    # 设置参数
    local_model_path = "D:\\workspace\\llm\\FinBERT_L-12_H-768_A-12_pytorch"
    file_path = 'trainDataset/train.txt'
    max_len = 128
    batch_size = 16
    epochs = 5
    learning_rate = 2e-5

    # 初始化 tokenizer 和 MultiLabelBinarizer
    # The MultiLabelBinarizer from the sklearn.preprocessing module is used to transform a list of labels into a binary format, which is suitable for machine learning algorithms. This is particularly useful for multi-label classification problems where each instance (e.g., a sentence) can be associated with multiple labels.
    tokenizer = BertTokenizer.from_pretrained(local_model_path)
    positive_mlb = MultiLabelBinarizer()
    negative_mlb = MultiLabelBinarizer()
    label_type_mlb = MultiLabelBinarizer()

    # 创建训练数据集和数据加载器
    train_dataset = CustomDataset(file_path, tokenizer, max_len=max_len,
                                  positive_mlb=positive_mlb, negative_mlb=negative_mlb,
                                  label_type_mlb=label_type_mlb, is_test=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 初始化模型
    positive_classes = len(train_dataset.positive_mlb.classes_)
    negative_classes = len(train_dataset.negative_mlb.classes_)
    label_type_classes = len(train_dataset.label_type_mlb.classes_)
    model = MultiLabelModel(n_positive_classes=positive_classes,
                            n_negative_classes=negative_classes,
                            n_label_types=label_type_classes,
                            local_model_path=local_model_path)

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # 初始化优化器、调度器和损失函数
    optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = CombinedLoss().to(device)

    # 训练模型
    trained_model = train_model(model, train_loader, loss_fn, optimizer, scheduler, device, epochs)

    # 保存模型和相关组件
    output_dir = 'saved_models'
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
    config = BertConfig.from_pretrained(local_model_path)
    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    joblib.dump(positive_mlb, 'saved_models/positive_mlb.pkl')
    joblib.dump(negative_mlb, 'saved_models/negative_mlb.pkl')
    joblib.dump(label_type_mlb, 'saved_models/label_type_mlb.pkl')

    print("Training completed and model saved.")