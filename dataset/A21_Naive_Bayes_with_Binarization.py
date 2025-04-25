from collections import defaultdict
import numpy as np

from dataset.A00_utils import print_predictions
from dataset.A10_Preprocessing import load_data


def train_naive_bayes_binarized(documents, labels):
    # 构建词汇表：包含所有文档中出现过的唯一词
    vocabulary = set(word for doc in documents for word in set(doc.split()))
    N = len(documents)  # 文档总数
    classes = set(labels)  # 所有类别
    prior = {}  # 先验概率
    cond_prob = defaultdict(lambda: defaultdict(float))  # 条件概率

    for c in classes:
        # 提取属于类别 c 的文档
        docs_c = [doc for doc, label in zip(documents, labels) if label == c]
        N_c = len(docs_c)  # 类别 c 的文档数
        prior[c] = N_c / N  # 计算先验概率 P(c)
        word_counts = defaultdict(int)
        # 统计类别 c 中每个词出现的文档数（二值化）
        for doc in docs_c:
            for word in set(doc.split()):  # 只考虑词是否存在
                word_counts[word] += 1
        # 计算条件概率 P(w|c)，使用拉普拉斯平滑
        for word in vocabulary:
            cond_prob[word][c] = (word_counts[word] + 1) / (N_c + len(vocabulary))

    return prior, cond_prob, vocabulary

def classify_naive_bayes_binarized(document, prior, cond_prob, vocabulary):
    tokens = set(document.split())  # 提取文档的唯一词（二值化）
    scores = {c: np.log(prior[c]) for c in prior}  # 初始化得分（对数形式）
    # 计算每个类别的得分
    for word in tokens:
        if word in vocabulary:
            for c in prior:
                scores[c] += np.log(cond_prob[word][c])  # 累加条件概率的对数
    # 返回得分最高的类别
    return max(scores, key=scores.get)

if __name__ == "__main__":
    data = load_data()
    train_texts, train_labels = data['train_texts'], data['train_labels']
    test_texts, test_labels = data['test_texts'], data['test_labels']

    # 训练模型
    prior_nb_bin, cond_prob_nb_bin, vocab_nb_bin = train_naive_bayes_binarized(train_texts, train_labels)
    # 对测试集进行预测
    nb_bin_predictions = [classify_naive_bayes_binarized(doc, prior_nb_bin, cond_prob_nb_bin, vocab_nb_bin) for doc in test_texts]
    # 打印前几个预测结果，确认代码运行
    print_predictions(nb_bin_predictions, test_labels, "Naive Bayes (Binarization)")
