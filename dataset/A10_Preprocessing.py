import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from sklearn.model_selection import train_test_split

# 下载 NLTK 资源
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 预处理函数：分词、转小写、去除标点和停用词
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [t for t in tokens if t.strip() != '']
    return " ".join(tokens)

# 加载和预处理数据集
def load_data(dataset_path='News_Category_Dataset_v3_balanced.json'):
    data = pd.read_json(dataset_path, lines=True)
    data['text'] = data['headline'] + " " + data['short_description']
    data['text'] = data['text'].apply(preprocess_text)

    # 数据集划分：70% 训练，10% 验证，20% 测试
    train_data, temp_data = train_test_split(data, test_size=0.3, stratify=data['category'], random_state=42)
    dev_data, test_data = train_test_split(temp_data, test_size=0.667, stratify=temp_data['category'], random_state=42)

    # 提取文本和标签
    train_texts, train_labels = train_data['text'].tolist(), train_data['category'].tolist()
    dev_texts, dev_labels = dev_data['text'].tolist(), dev_data['category'].tolist()
    test_texts, test_labels = test_data['text'].tolist(), test_data['category'].tolist()
    # print(train_texts)
    # print(train_labels)
    # print(dev_texts)
    # print(dev_labels)
    # print(test_texts)
    # print(test_labels)
    print(f"训练集大小: {len(train_texts)} 篇文章")
    print(f"测试集大小: {len(test_texts)} 篇文章")

    return {
        'train_texts': train_texts,
        'train_labels': train_labels,
        'dev_texts': dev_texts,
        'dev_labels': dev_labels,
        'test_texts': test_texts,
        'test_labels': test_labels
    }

if __name__ == "__main__":
    data = load_data()