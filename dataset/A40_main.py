from dataset.A00_utils import print_predictions, compute_metrics
from dataset.A10_Preprocessing import load_data
from dataset.A21_Naive_Bayes_with_Binarization import train_naive_bayes_binarized, classify_naive_bayes_binarized
from dataset.A22_Naive_Bayes_TF_IDF import train_naive_bayes_tfidf, classify_naive_bayes_tfidf
from dataset.A23_RNN import train_rnn, classify_rnn
from dataset.A30_baselines import run_baselines


def run_experiments():
    # 加载数据
    data = load_data()
    train_texts, train_labels = data['train_texts'], data['train_labels']
    test_texts, test_labels = data['test_texts'], data['test_labels']
    labels = list(set(train_labels))

    # 朴素贝叶斯（二值化）
    prior_nb_bin, cond_prob_nb_bin, vocab_nb_bin = train_naive_bayes_binarized(train_texts, train_labels)
    nb_bin_predictions = [classify_naive_bayes_binarized(doc, prior_nb_bin, cond_prob_nb_bin, vocab_nb_bin) for doc in test_texts]
    print_predictions(nb_bin_predictions, test_labels, "Naive Bayes (Binarization)")

    # 朴素贝叶斯（TF-IDF）
    prior_nb_tfidf, mean_tfidf_nb_tfidf, var_tfidf_nb_tfidf, vocab_nb_tfidf = train_naive_bayes_tfidf(train_texts, train_labels)
    nb_tfidf_predictions = [classify_naive_bayes_tfidf(doc, prior_nb_tfidf, mean_tfidf_nb_tfidf, var_tfidf_nb_tfidf, vocab_nb_tfidf) for doc in test_texts]
    print_predictions(nb_tfidf_predictions, test_labels, "Naive Bayes (TF-IDF)")

    # RNN（词嵌入）
    rnn_model, rnn_word2idx, rnn_label2idx = train_rnn(train_texts, train_labels, glove_file='glove.6B.100d.txt')
    rnn_predictions = [classify_rnn(doc, rnn_model, rnn_word2idx) for doc in test_texts]
    idx2label = {idx: label for label, idx in rnn_label2idx.items()}
    rnn_predictions = [idx2label[pred] for pred in rnn_predictions]
    print_predictions(rnn_predictions, test_labels, "RNN (Word Embeddings)")

    # 基线方法
    baselines = run_baselines()
    majority_predictions = baselines['majority']
    random_predictions = baselines['random']
    lr_predictions = baselines['logistic_regression']
    print_predictions(majority_predictions, test_labels, "Majority Class Baseline")
    print_predictions(random_predictions, test_labels, "Random Baseline")
    print_predictions(lr_predictions, test_labels, "Logistic Regression (BoW)")

    # 评估所有方法
    methods = {
        'Naive Bayes (Binarization)': nb_bin_predictions,
        'Naive Bayes (TF-IDF)': nb_tfidf_predictions,
        'RNN (Word Embeddings)': rnn_predictions,
        'Majority Class Baseline': majority_predictions,
        'Random Baseline': random_predictions,
        'Logistic Regression (BoW)': lr_predictions
    }

    results = {}
    for method, predictions in methods.items():
        accuracy, macro_f1, weighted_f1 = compute_metrics(test_labels, predictions, labels)
        results[method] = (accuracy, macro_f1, weighted_f1)

    # 打印结果表格
    print("\n实验结果：")
    print("| Method | Accuracy | Macro-Averaged F1 | Weighted-Averaged F1 |")
    print("| --- | --- | --- | --- |")
    for method, (accuracy, macro_f1, weighted_f1) in results.items():
        print(f"| {method} | {accuracy:.3f} | {macro_f1:.3f} | {weighted_f1:.3f} |")

if __name__ == "__main__":
    run_experiments()