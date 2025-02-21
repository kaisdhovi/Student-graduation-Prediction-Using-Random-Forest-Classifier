import pandas as pd
import numpy as np
from collections import Counter
import random
from joblib import Parallel,delayed
random.seed(42)
np.random.seed(42)


def class_counts(rows):
    return Counter(np.array(rows)[:, -1])

def gini(rows):
    if len(rows) == 0:
        return 0
    labels = Counter(np.array(rows)[:, -1])
    probabilities = np.array(list(labels.values())) / len(rows)
    return 1 - np.sum(probabilities ** 2)


def info_gain(left, right, current_uncertainty):
    total_count = len(left) + len(right)
    if len(left) == 0 or len(right) == 0:
        return 0

    p = len(left) / total_count
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)

class Leaf:
    def __init__(self, rows):
        label_counts = class_counts(rows)
        total = sum(label_counts.values())
        self.predictions = {label: count / total for label, count in label_counts.items()}
class Decision_Node:
  def __init__(self,
               question,
               true_branch,
               false_branch):
    self.question = question
    self.true_branch = true_branch
    self.false_branch = false_branch
class QuestionSplit:
    def __init__(self, column, value, unique_values=None):
        self.column = column
        self.value = value

    def is_numeric(self, value):
        return isinstance(value, (int, float, np.number))

    def match(self, example):
        if isinstance(example[self.column], str):
            return example[self.column] == self.value
        else:
            val = example[self.column]
            if self.is_numeric(val) and self.is_numeric(self.value):
                return val >= self.value
            else:
                return str(val) == str(self.value)

    def __repr__(self):
        condition = "=="
        if self.is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (
            self.column, condition, str(self.value))
def partisi(rows, question):
    rows = np.array(rows)
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return np.array(true_rows), np.array(false_rows)
def find_best_split(rows, feature_indices,min_samples_split=2, min_samples_leaf=1):
    rows = np.array(rows)
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)

    if len(rows) < min_samples_split:
        return 0, None
    for col in feature_indices:
        values = set([row[col] for row in rows])
        for val in values:
            question = QuestionSplit(col, val)
            true_rows, false_rows = partisi(rows, question)
            if len(true_rows) < min_samples_leaf or len(false_rows) < min_samples_leaf:
                continue
            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain > best_gain:
                best_gain, best_question = gain, question


    return best_gain, best_question
def build_tree(rows, depth=0, max_depth=5, feature_indices=None, min_samples_split=2, min_samples_leaf=1):
    rows = np.array(rows)
    if feature_indices is None:
        feature_indices = list(range(rows.shape[1] - 1))

    gain, question = find_best_split(rows, feature_indices, min_samples_split, min_samples_leaf)

    if gain == 0 or depth == max_depth:
        return Leaf(rows)

    true_rows, false_rows = partisi(rows, question)

    true_branch = build_tree(true_rows, depth + 1, max_depth, feature_indices, min_samples_split, min_samples_leaf)
    false_branch = build_tree(false_rows, depth + 1, max_depth, feature_indices, min_samples_split, min_samples_leaf)

    return Decision_Node(question, true_branch, false_branch)

def classify(row, node):
    if isinstance(node, Leaf):
        return node.predictions
    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)
class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        data = np.column_stack((X, y))
        self.tree = build_tree(data, max_depth=self.max_depth,
                              min_samples_split=self.min_samples_split,
                              min_samples_leaf=self.min_samples_leaf)

    def predict(self, X):
        predictions = []
        # Convert the DataFrame to a NumPy array to iterate through the rows correctly.
        X = X.values # The change is on this line
        for row in X:
            prediction = max(classify(row, self.tree), key=classify(row, self.tree).get)
            predictions.append(prediction)

        return predictions
def bootstraping(data):
    n = len(data)
    bootstrap = [random.randint(0, n - 1) for _ in range(n)]
    bootstrap_set = set(bootstrap)  
    oob = [i for i in range(n) if i not in bootstrap_set]
    return bootstrap, oob


def random_fitur(fitur, n_fitur):
    if n_fitur > len(fitur):
        raise ValueError("n_fitur tidak boleh lebih besar dari jumlah fitur")
    return random.sample(fitur, n_fitur)
    




class RandomHutanClassifier:
    def __init__(self, n_estimator=10, max_depth=5, min_samples_split=2, min_samples_leaf=1, n_fitur='sqrt', n_jobs=- 1):
        self.n_estimator = n_estimator
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_fitur = n_fitur
        self.trees = []
        self.oob = []
        self.n_jobs = n_jobs

    def fit(self,X, y,batch_size= None):
        data = np.column_stack((X, y))
        self.trees = []
        self.oob = []
        if self.n_fitur is None:
            self.n_fitur = data.shape[1] - 1
        if self.n_fitur == 'log2':
            self.n_fitur = int(np.log2(data.shape[1]))
        if self.n_fitur == 'sqrt':
            self.n_fitur = int(np.sqrt(data.shape[1]))
        feature_indices = list(range(data.shape[1] - 1))

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self.build_tree)(data, feature_indices)
            for _ in range(self.n_estimator)
        )

        self.trees = [result[0] for result in results]
        print('pohon',len(self.trees))
        self.oob = [result[1] for result in results]

    def build_tree(self, data, feature_indices):
        bootstrap, oob_index = bootstraping(data)
        self.oob.append(oob_index)
        fitur_terpilih = random_fitur(feature_indices, self.n_fitur)
        tree = build_tree(data[bootstrap], max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, feature_indices=fitur_terpilih)
        return tree, oob_index

    def predict(self, X):
        predictions = []

        # Cek jika X adalah DataFrame
        if isinstance(X, pd.DataFrame):
            X = X.values  # Ambil X.values jika X adalah DataFrame
        elif isinstance(X, list):
            X = np.array(X)  # Ubah X menjadi array jika X adalah list
        elif isinstance(X, np.ndarray):
            pass  # Jika X sudah numpy array, biarkan seperti itu
        else:
            raise ValueError("Input X harus berupa DataFrame, list, atau numpy array")

        for row in X:
            prediction = []
            for tree in self.trees:
                prediction.append(max(classify(row, tree), key=classify(row, tree).get))
            predictions.append(max(set(prediction), key=prediction.count))

        return predictions


    def oob_score(self, data):

        # jumlah pohon == oob sample

        if len(self.trees) != len(self.oob):
            raise ValueError("jumlah pohon hrs sama dengan oob sample")

        # harus sesuai range
        num_samples = len(data)
        for oob_indices in self.oob:
            if any(idx >= num_samples or idx < 0 for idx in oob_indices):
                raise ValueError("jumlah oob sample hrs sama dgn dataset")

        oob_predictions = {i: [] for i in range(num_samples)}


        def process_tree(tree_idx, oob_indices):
            for idx in oob_indices:
                prediction = classify(data[idx], self.trees[tree_idx])
                predicted_class = max(prediction, key=prediction.get)

                # Simpan prediksi dri oob
                oob_predictions[idx].append(predicted_class)



        for i, oob_indices in enumerate(self.oob):
            process_tree(i, oob_indices)

        # hitung akurasi
        correct = 0
        total = 0

        if not oob_predictions:
            raise ValueError("tidak ada oob sample")

        for idx, preds in oob_predictions.items():
            if preds:
                majority_vote = Counter(preds).most_common(1)[0][0]
                if majority_vote == data[idx][-1]:
                    correct += 1
                total += 1
        print('total:',total)
        print('correct:' ,correct)
        return correct / total if total > 0 else 0

    def get_params(self, deep=True):
        return {
            'n_estimator': self.n_estimator,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'n_fitur': self.n_fitur,
            'n_jobs': self.n_jobs
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self