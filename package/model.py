import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, kernel_sizes, num_channels, num_conv_layers):
        super(TextCNN, self).__init__()

        # 使用BERT嵌入层（如果需要）
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # 增加多个卷积层
        self.convs = nn.ModuleList()
        for _ in range(num_conv_layers):
            for kernel_size in kernel_sizes:
                # 每个卷积层使用不同的卷积核
                self.convs.append(nn.Conv2d(1, num_channels, (kernel_size, embed_size)))

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(kernel_sizes) * num_channels * num_conv_layers, num_classes)  # 修改全连接层输入大小

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, embed_size)
        x = x.unsqueeze(1)  # (batch_size, 1, seq_len, embed_size) - 加入通道维度

        # 进行卷积并池化
        conv_outputs = [
            torch.relu(conv(x)).squeeze(3) for conv in self.convs
        ]  # [(batch_size, num_channels, seq_len), ...]

        pooled_outputs = [
            torch.max_pool1d(output, output.size(2)).squeeze(2) for output in conv_outputs
        ]  # [(batch_size, num_channels), ...]

        cat = torch.cat(pooled_outputs, 1)  # (batch_size, len(kernel_sizes) * num_channels * num_conv_layers)
        cat = self.dropout(cat)
        logits = self.fc(cat)  # (batch_size, num_classes)

        seq_len = x.size(2)
        logits = logits.unsqueeze(1).repeat(1, seq_len, 1)  # (batch_size, seq_len, num_classes)
        return logits
