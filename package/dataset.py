import json
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter


class NERDataset(Dataset):
    def __init__(self, file_path, tokenizer, label_map, max_length=128):
        """
        初始化数据集

        Args:
            file_path (str): 数据文件路径
            tokenizer (transformers.PreTrainedTokenizer): 分词器
            label_map (dict): 标签到索引的映射
            max_length (int): 最大序列长度
        """
        self.data = self.load_jsonlines(file_path)
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.max_length = max_length

    def load_jsonlines(self, file_path):
        """
        读取jsonlines格式的数据

        Args:
            file_path (str): 数据文件路径

        Returns:
            list: 包含所有数据的列表
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取数据集中的一个样本

        Args:
            idx (int): 索引

        Returns:
            dict: 包含输入ID、标签ID、注意力掩码等信息的字典
        """
        item = self.data[idx]
        text = item['text']
        labels = item['labels']

        # 使用tokenizer对文本进行编码
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            is_split_into_words=True
        )

        # 将标签映射到索引
        label_ids = [self.label_map[label] for label in labels]
        label_ids = label_ids[:self.max_length] + [self.label_map["O"]] * (self.max_length - len(label_ids))

        # 转为tensor
        encoding['labels'] = torch.tensor(label_ids, dtype=torch.long)

        return {key: val.squeeze(0) for key, val in encoding.items()}

    def get_label_counts(self):
        """
        统计数据集中每个标签的出现次数

        Returns:
            dict: 每个标签及其出现次数的字典
        """
        label_counts = Counter()
        for item in self.data:
            for label in item['labels']:
                label_counts[self.label_map[label]] += 1
        return label_counts


# 示例用法
if __name__ == "__main__":
    from transformers import BertTokenizer

    # 示例标签映射
    label_map = {"O": 0, "B-EQU": 1, "I-EQU": 2, "B-TIME": 3, "I-TIME": 4, "B-ORG": 5, "I-ORG": 6}

    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    # 初始化数据集
    dataset = NERDataset(file_path='data/FNED/train.json', tokenizer=tokenizer, label_map=label_map)

    # 打印标签出现次数
    label_counts = dataset.get_label_counts()
    print(label_counts)
