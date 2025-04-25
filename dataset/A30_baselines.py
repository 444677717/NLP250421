import random
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from dataset.A00_utils import print_predictions
from dataset.A10_Preprocessing import load_data


def run_baselines():
    data = load_data()
    train_texts, train_labels = data['train_texts'], data['train_labels']
    test_texts, test_labels = data['test_texts'], data['test_labels']

    # 多数类基线
    most_common_class = Counter(train_labels).most_common(1)[0][0]
    majority_predictions = [most_common_class for _ in test_texts]

    # 随机基线
    classes = list(set(train_labels))
    random_predictions = [random.choice(classes) for _ in test_texts]

    # 逻辑回归基线
    vectorizer = CountVectorizer()
    X_train_lr = vectorizer.fit_transform(train_texts)
    X_test_lr = vectorizer.transform(test_texts)
    lr_model = LogisticRegression(multi_class='ovr', max_iter=1000)
    lr_model.fit(X_train_lr, train_labels)
    lr_predictions = lr_model.predict(X_test_lr)

    return {
        'majority': majority_predictions,
        'random': random_predictions,
        'logistic_regression': lr_predictions,
        'test_labels': test_labels
    }

if __name__ == "__main__":
    predictions = run_baselines()
    print_predictions(predictions['majority'], predictions['test_labels'], "Majority Class Baseline")
    print_predictions(predictions['random'], predictions['test_labels'], "Random Baseline")
    print_predictions(predictions['logistic_regression'], predictions['test_labels'], "Logistic Regression (BoW)")