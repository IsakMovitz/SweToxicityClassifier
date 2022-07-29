
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from datasets import load_dataset,DatasetDict
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics


def load_split_data(jsonl_file,train_test_split):
 
    dataset = load_dataset('json', data_files= jsonl_file)['train']
    dataset = dataset.remove_columns(["id","thread_id","thread","keyword","starting_index","span_length"])
    dataset = dataset.rename_column("TOXIC", "label")

    train_test = dataset.train_test_split(test_size=train_test_split)
    train_test_dataset = DatasetDict({
        'train': train_test['train'],
        'test': train_test['test']
    })

    full_datasets = train_test_dataset

    return full_datasets


def load_data(jsonl_file):

    # Code for train,test,valid split
    dataset = load_dataset('json', data_files=jsonl_file)['train']

    dataset = dataset.remove_columns(["id","thread_id","thread","keyword","starting_index","span_length"])

    dataset = dataset.rename_column("TOXIC", "label")

    return dataset

train = load_data("./TRAIN_700.jsonl")
train_text_data = train['text']
labels = train['label']

test = load_data("./TEST_150.jsonl")
test_text_data = test['text']
test_labels = test['label']

# Pipeline and fitting model
"""
Models:
NAIVE BAYES
MultinomialNB(),¨

STOCHASTIC GRADIENT DESCENT
SGDClassifier(loss='hinge', penalty='l2',
                         alpha=1e-3, random_state=42,
                         max_iter=5, tol=None)),

MULTI-LAYER PERCEPTRON CLASSIFIER
MLPClassifier(solver='lbfgs', hidden_layer_sizes=50,
                                max_iter=150, shuffle=True, random_state=1,
                                activation=activation)
Tokenizers:
TfidfTransformer(),

"""
text_clf = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', SVC())])

text_clf.fit(train_text_data, labels)

# Evaluating model

predicted = text_clf.predict(test_text_data)
accuracy = np.mean(predicted == test_labels)
results = metrics.classification_report(test_labels, predicted)

print("-- Model --")
print(text_clf['clf'])
print(results)
