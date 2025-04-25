import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from dataset.A00_utils import print_predictions
from dataset.A10_Preprocessing import load_data

# 设置设备（优先使用 GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:',device)

# 加载 GloVe 词嵌入（与 TensorFlow 版本相同）
def load_glove_embeddings(glove_file, embedding_dim=100):
    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index


# 创建嵌入矩阵（与 TensorFlow 版本相同）
def create_embedding_matrix(word2idx, embeddings_index, embedding_dim=100):
    vocab_size = len(word2idx) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in word2idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[idx] = embedding_vector
    return embedding_matrix, vocab_size


# 自定义数据集类，用于处理序列数据
class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, documents, labels, word2idx, label2idx, max_len=50):
        self.documents = documents
        self.labels = labels
        self.word2idx = word2idx
        self.label2idx = label2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        # 将文档转换为词索引序列，并填充到 max_len
        tokens = [self.word2idx.get(word, 0) for word in self.documents[idx].split()]
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]
        else:
            tokens = tokens + [0] * (self.max_len - len(tokens))

        # 转换为张量
        tokens = torch.tensor(tokens, dtype=torch.long)
        label = torch.tensor(self.label2idx[self.labels[idx]], dtype=torch.long)
        return tokens, label


# RNN 模型定义
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, embedding_matrix):
        super(RNNClassifier, self).__init__()
        # 嵌入层，加载预训练的 GloVe 词嵌入
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False  # 不更新词嵌入

        # LSTM 层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, text):
        embedded = self.embedding(text)  # [batch_size, seq_len, embedding_dim]
        lstm_out, (hidden, cell) = self.lstm(embedded)  # hidden: [1, batch_size, hidden_dim]
        hidden = hidden.squeeze(0)  # [batch_size, hidden_dim]
        output = self.fc(hidden)  # [batch_size, output_dim]
        return output


# 训练函数
def train_rnn(documents, labels, embedding_dim=100, glove_file='glove.6B.100d.txt', epochs=5, batch_size=32):
    # 创建词汇表和标签映射
    word2idx = {word: idx + 1 for idx, word in enumerate(set(word for doc in documents for word in doc.split()))}
    embeddings_index = load_glove_embeddings(glove_file, embedding_dim)
    embedding_matrix, vocab_size = create_embedding_matrix(word2idx, embeddings_index, embedding_dim)
    label2idx = {label: idx for idx, label in enumerate(set(labels))}

    # 准备数据集
    dataset = NewsDataset(documents, labels, word2idx, label2idx)
    train_size = int(0.9 * len(dataset))  # 90% 训练，10% 验证
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    # 初始化模型
    hidden_dim = 128
    output_dim = len(set(labels))
    model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, embedding_matrix).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # 训练循环
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0, 0
        for batch_idx, (text, labels) in enumerate(train_loader):
            text, labels = text.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(text)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            train_correct += (preds == labels).sum().item()

        train_accuracy = train_correct / train_size

        # 验证
        model.eval()
        val_loss, val_correct = 0, 0
        with torch.no_grad():
            for text, labels in val_loader:
                text, labels = text.to(device), labels.to(device)
                output = model(text)
                loss = criterion(output, labels)
                val_loss += loss.item()
                preds = torch.argmax(output, dim=1)
                val_correct += (preds == labels).sum().item()

        val_accuracy = val_correct / val_size
        print(
            f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")

    return model, word2idx, label2idx


# 预测函数
def classify_rnn(document, model, word2idx, max_len=50):
    model.eval()
    tokens = [word2idx.get(word, 0) for word in document.split()]
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens = tokens + [0] * (max_len - len(tokens))

    tokens = torch.tensor([tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(tokens)
        pred = torch.argmax(output, dim=1).item()
    return pred


if __name__ == "__main__":
    # 加载数据
    data = load_data()
    train_texts, train_labels = data['train_texts'], data['train_labels']
    test_texts, test_labels = data['test_texts'], data['test_labels']

    # 训练模型
    model, word2idx, label2idx = train_rnn(train_texts, train_labels, glove_file= "glove.6B.100d.txt")

    # 预测
    rnn_predictions = [classify_rnn(doc, model, word2idx) for doc in test_texts]
    idx2label = {idx: label for label, idx in label2idx.items()}
    rnn_predictions = [idx2label[pred] for pred in rnn_predictions]

    # 打印预测结果
    print_predictions(rnn_predictions, test_labels, "RNN (Word Embeddings)")