import torch
from torch.utils.data import Dataset


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