import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from package.model import TextCNN
from package.dataset import NERDataset
from package.metrics import calculate_metrics, calculate_entity_level_metrics
import logging
from datetime import datetime
from collections import defaultdict

# 超参数
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3  # 调整学习率
MAX_LEN = 128

# 数据集路径
TRAIN_FILE_PATH = 'data/FNED/balanced_train.json'
VALID_FILE_PATH = 'data/FNED/balanced_valid.json'

# 资源路径
RESOURCE_PATH = 'resource/chinese_roberta_wwm_ext_pytorch'
TOKENIZER_PATH = os.path.join(RESOURCE_PATH, 'vocab.txt')
MODEL_PATH = os.path.join(RESOURCE_PATH, 'pytorch_model.bin')

# 标签映射
LABEL_MAP = {
    'O': 0,
    'B-PER': 1, 'I-PER': 2,
    'B-FAC': 3, 'I-FAC': 4,
    'B-ORG': 5, 'I-ORG': 6,
    'B-GPE': 7, 'I-GPE': 8,
    'B-LOC': 9, 'I-LOC': 10,
    'B-EVE': 11, 'I-EVE': 12,
    'B-EQU': 13, 'I-EQU': 14,
    'B-TIME': 15, 'I-TIME': 16
}

# 日志设置
log_dir = './log'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = os.path.join(log_dir, f'training_log_{timestamp}.log')
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 使用GPU（如果可用）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载分词器和预训练模型
tokenizer = BertTokenizer(vocab_file=TOKENIZER_PATH)
bert_model = BertModel.from_pretrained(RESOURCE_PATH)

# 创建数据集和数据加载器
train_dataset = NERDataset(file_path=TRAIN_FILE_PATH, tokenizer=tokenizer, label_map=LABEL_MAP, max_length=MAX_LEN)
valid_dataset = NERDataset(file_path=VALID_FILE_PATH, tokenizer=tokenizer, label_map=LABEL_MAP, max_length=MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 计算标签权重
label_counts = train_dataset.get_label_counts()
total_count = sum(label_counts.values())

# 使用 defaultdict 确保所有标签都有权重，即使没有出现在训练数据中
weights = defaultdict(lambda: total_count, {label: total_count / count for label, count in label_counts.items()})
weight_tensor = torch.tensor([weights[label] for label in range(len(LABEL_MAP))], dtype=torch.float).to(device)

# 打印标签权重，检查是否有异常
print("标签权重:")
for label, weight in weights.items():
    print(f"标签 {label}: 权重 {weight}")

# 在初始化TextCNN模型时添加参数
num_conv_layers = 3  # 增加卷积层的数量
num_channels = 256  # 每个卷积层的卷积核数量
kernel_sizes = [3, 4, 5]  # 卷积核的大小

# 初始化模型
model = TextCNN(vocab_size=len(tokenizer.vocab),
                embed_size=bert_model.config.hidden_size,
                num_classes=len(LABEL_MAP),
                kernel_sizes=kernel_sizes,
                num_channels=num_channels,
                num_conv_layers=num_conv_layers)  # 新增参数

# # 初始化TextCNN模型
# model = TextCNN(vocab_size=len(tokenizer.vocab), embed_size=bert_model.config.hidden_size, num_classes=len(LABEL_MAP),
#                 kernel_sizes=[3, 4, 5], num_channels=256)
model.embedding = bert_model.embeddings.word_embeddings  # 使用BERT的词嵌入

model.to(device)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss(weight=weight_tensor, ignore_index=LABEL_MAP['O'])  # 使用标签权重
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        inputs = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(inputs)

        # 将 outputs 和 labels 展平，并确保它们的形状匹配
        batch_size, seq_len, num_classes = outputs.shape
        outputs = outputs.view(batch_size * seq_len, num_classes)
        labels = labels.view(batch_size * seq_len)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(data_loader)


def evaluate(model, data_loader, criterion, device, label_map):
    model.eval()
    total_loss = 0
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(inputs)

            # 将 outputs 和 labels 展平，并确保它们的形状匹配
            batch_size, seq_len, num_classes = outputs.shape
            outputs = outputs.view(batch_size * seq_len, num_classes)
            labels = labels.view(batch_size * seq_len)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            true_labels.extend(labels.cpu().numpy().tolist())
            pred_labels.extend(preds.cpu().numpy().tolist())

    # 确保 true_labels 和 pred_labels 是列表形式
    assert isinstance(true_labels, list), f"true_labels is not a list, but {type(true_labels)}"
    assert isinstance(pred_labels, list), f"pred_labels is not a list, but {type(pred_labels)}"

    print(f"true_labels: {true_labels[:10]}")  # 打印前10个元素进行调试
    print(f"pred_labels: {pred_labels[:10]}")  # 打印前10个元素进行调试

    metrics = calculate_metrics(true_labels, pred_labels, label_map)
    entity_metrics = calculate_entity_level_metrics(true_labels, pred_labels, label_map, BATCH_SIZE)
    return total_loss / len(data_loader), metrics, entity_metrics


# 训练和验证
best_f1 = 0.0

for epoch in range(EPOCHS):
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
    valid_loss, metrics, entity_metrics = evaluate(model, valid_loader, criterion, device, LABEL_MAP)

    logging.info(f"Epoch {epoch + 1}/{EPOCHS}")
    logging.info(f"Train Loss: {train_loss:.4f}")
    logging.info(f"Valid Loss: {valid_loss:.4f}")
    logging.info(f"Metrics: {metrics}")
    logging.info(f"Entity Level Metrics: {entity_metrics}")

    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Valid Loss: {valid_loss:.4f}")
    print(f"Metrics: {metrics}")
    print(f"Entity Level Metrics: {entity_metrics}")

    # 保存最佳模型
    if entity_metrics['f1'] > best_f1:
        best_f1 = entity_metrics['f1']
        best_model_path = f'best_model_{timestamp}.bin'
        torch.save(model.state_dict(), best_model_path)
        logging.info(f"Best model saved with F1 score: {best_f1:.4f}")

print("训练完成！")
logging.info("训练完成！")
