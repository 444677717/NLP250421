from collections import defaultdict
import numpy as np

def compute_metrics(y_true, y_pred, labels):
    accuracy = np.mean([1 if pred == true else 0 for pred, true in zip(y_pred, y_true)])
    class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            class_metrics[true]['tp'] += 1
        else:
            class_metrics[pred]['fp'] += 1
            class_metrics[true]['fn'] += 1

    precisions, recalls, f1s = [], [], []
    for label in labels:
        tp = class_metrics[label]['tp']
        fp = class_metrics[label]['fp']
        fn = class_metrics[label]['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_f1 = np.mean(f1s)
    label_counts = defaultdict(int)
    for true in y_true:
        label_counts[true] += 1
    weighted_f1 = sum(f1 * label_counts[label] for f1, label in zip(f1s, labels)) / len(y_true)

    return accuracy, macro_f1, weighted_f1


def print_predictions(predictions, test_labels, method_name, num_samples=5):
    print(f"\n{method_name} - 前 {num_samples} 个测试样本的预测结果：")
    for i in range(min(num_samples, len(predictions))):
        print(f"测试样本 {i + 1}: 预测类别 = {predictions[i]}, 真实类别 = {test_labels[i]}")