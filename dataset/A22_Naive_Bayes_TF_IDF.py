from collections import defaultdict
import numpy as np

from dataset.A00_utils import print_predictions
from dataset.A10_Preprocessing import load_data


def compute_tfidf(documents):
    N = len(documents)  # 文档总数
    vocabulary = set(word for doc in documents for word in doc.split())  # 构建词汇表
    df = defaultdict(int)  # 文档频率
    for doc in documents:
        unique_words = set(doc.split())
        for word in unique_words:
            df[word] += 1  # 统计每个词出现的文档数

    tfidf_matrix = []
    for doc in documents:
        word_counts = defaultdict(int)
        words = doc.split()
        for word in words:
            word_counts[word] += 1  # 统计词频
        max_count = max(word_counts.values()) if word_counts else 1  # 用于归一化
        tfidf_doc = {}
        for word in word_counts:
            tf = word_counts[word] / max_count  # 归一化词频
            idf = np.log(N / (df[word] + 1))  # 逆文档频率，加平滑
            tfidf_doc[word] = tf * idf  # 计算 TF-IDF 值
        tfidf_matrix.append(tfidf_doc)
    return tfidf_matrix, vocabulary

def train_naive_bayes_tfidf(documents, labels):
    tfidf_matrix, vocabulary = compute_tfidf(documents)
    classes = set(labels)
    N = len(documents)
    prior = {}
    mean_tfidf = defaultdict(lambda: defaultdict(float))
    var_tfidf = defaultdict(lambda: defaultdict(float))

    for c in classes:
        indices = [i for i, label in enumerate(labels) if label == c]
        N_c = len(indices)
        prior[c] = N_c / N  # 计算先验概率 P(c)
        for word in vocabulary:
            # 计算类别 c 中每个词的 TF-IDF 值的均值和方差
            values = [tfidf_matrix[i].get(word, 0) for i in indices]
            mean_tfidf[word][c] = np.mean(values) if values else 0
            var_tfidf[word][c] = np.var(values) if values and len(values) > 1 else 1e-10  # 避免方差为 0

    return prior, mean_tfidf, var_tfidf, vocabulary

def classify_naive_bayes_tfidf(document, prior, mean_tfidf, var_tfidf, vocabulary):
    tfidf_doc, _ = compute_tfidf([document])
    tfidf = tfidf_doc[0]  # 获取文档的 TF-IDF 表示
    scores = {c: np.log(prior[c]) for c in prior}  # 初始化得分（对数形式）
    for word in tfidf:
        if word in vocabulary:
            tfidf_val = tfidf[word]
            for c in prior:
                mean = mean_tfidf[word][c]
                var = var_tfidf[word][c]
                if var > 1e-10:  # 避免除零错误
                    # 使用高斯概率密度函数计算条件概率
                    gaussian = -0.5 * np.log(2 * np.pi * var) - ((tfidf_val - mean) ** 2) / (2 * var)
                else:
                    gaussian = -np.inf  # 如果方差非常小，可以设为负无穷
                scores[c] += gaussian
    return max(scores, key=scores.get)


if __name__ == "__main__":
    data = load_data()
    train_texts, train_labels = data['train_texts'], data['train_labels']
    test_texts, test_labels = data['test_texts'], data['test_labels']

    # 训练模型
    prior_nb_tfidf, mean_tfidf_nb_tfidf, var_tfidf_nb_tfidf, vocab_nb_tfidf = train_naive_bayes_tfidf(train_texts, train_labels)
    # 对测试集进行预测
    nb_tfidf_predictions = [classify_naive_bayes_tfidf(doc, prior_nb_tfidf, mean_tfidf_nb_tfidf, var_tfidf_nb_tfidf, vocab_nb_tfidf) for doc in test_texts]
    # 打印前几个预测结果，确认代码运行
    print_predictions(nb_tfidf_predictions, test_labels, "Naive Bayes (TF-IDF)")
