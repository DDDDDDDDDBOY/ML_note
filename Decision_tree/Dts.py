from abc import abstractclassmethod
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import math
from math import log

# data from textbook 5.1
def create_data():
    datasets = [['young', 'no', 'no', 'ordinary', 'no'],
               ['young', 'no', 'no', 'good', 'no'],
               ['young', 'yes', 'no', 'good', 'yes'],
               ['young', 'yes', 'yes', 'ordinary', 'yes'],
               ['young', 'no', 'no', 'ordinary', 'no'],
               ['middle', 'no', 'no', 'ordinary', 'no'],
               ['middle', 'no', 'no', 'good', 'no'],
               ['middle', 'yes', 'yes', 'good', 'yes'],
               ['middle', 'no', 'yes', 'verygood', 'yes'],
               ['middle', 'no', 'yes', 'verygood', 'yes'],
               ['old', 'no', 'yes', 'verygood', 'yes'],
               ['old', 'no', 'yes', 'good', 'yes'],
               ['old', 'yes', 'no', 'good', 'yes'],
               ['old', 'yes', 'no', 'verygood', 'yes'],
               ['old', 'no', 'no', 'ordinary', 'no'],
               ]
    labels = [u'age', u'job', u'house', u'credit', u'class']
    return datasets, labels
# empirical entropy
def calc_ent(datasets):
    data_length = len(datasets)
    label_count = {}
    for i in range(data_length):
        label = datasets[i][-1]
        if label not in label_count:
            label_count[label] = 0
        label_count[label] += 1
    ent = -sum([(p/data_length)*log(p/data_length, 2) for p in label_count.values()])
    return ent
# empircal conditional entropy
def cond_ent(datasets, axis=0):
    data_length = len(datasets)
    feature_sets = {}
    for i in range(data_length):
        feature = datasets[i][axis]
        if feature not in feature_sets:
            feature_sets[feature] = []
        feature_sets[feature].append(datasets[i])
    cond_ent = sum([(len(p)/data_length)*calc_ent(p) for p in feature_sets.values()])
    return cond_ent
#information_gain
def info_gain(ent,cond_ent):
    '''
    1.calculate H(D)
    2.calculate H(D|A)
    3.g(D,A) = H(D)-H(D|A)
    '''
    return ent - cond_ent
def info_gain_train(datasets):
    count = len(datasets[0]) - 1
    ent = calc_ent(datasets)
    best_feature = []
    for c in range(count):
        c_info_gain = info_gain(ent, cond_ent(datasets, axis=c))
        best_feature.append((c, c_info_gain))
        print('feature({}) - info_gain - {:.3f}'.format(labels[c], c_info_gain))
    # best feature
    best_ = max(best_feature, key=lambda x: x[-1])
    return print('feature({})max gain，root obtained'.format(labels[best_[0]]))
#define each node is a feature
class Node:
    def __init__(self, root=True, label=None, feature_name=None, feature=None):
        self.root = root
        self.label = label
        self.feature_name = feature_name
        self.feature = feature
        self.tree = {}
        self.result = {'label:': self.label, 'feature': self.feature, 'tree': self.tree}

    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self, val, node):
        self.tree[val] = node

    def predict(self, features):
        if self.root is True:
            return self.label
        return self.tree[features[self.feature]].predict(features)

class DTree:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self._tree = {}

    # empirical entropy
    @staticmethod
    def calc_ent(datasets):
        data_length = len(datasets)
        label_count = {}
        for i in range(data_length):
            label = datasets[i][-1]
            if label not in label_count:
                label_count[label] = 0
            label_count[label] += 1
        ent = -sum([(p/data_length)*log(p/data_length, 2) for p in label_count.values()])
        return ent

    # empirical conditional entropy
    def cond_ent(self, datasets, axis=0):
        data_length = len(datasets)
        feature_sets = {}
        for i in range(data_length):
            feature = datasets[i][axis]
            if feature not in feature_sets:
                feature_sets[feature] = []
            feature_sets[feature].append(datasets[i])
        cond_ent = sum([(len(p)/data_length)*self.calc_ent(p) for p in feature_sets.values()])
        return cond_ent

    # information gain
    @staticmethod
    def info_gain(ent, cond_ent):
        return ent - cond_ent

    def info_gain_train(self, datasets):
        count = len(datasets[0]) - 1
        ent = self.calc_ent(datasets)
        best_feature = []
        for c in range(count):
            c_info_gain = self.info_gain(ent, self.cond_ent(datasets, axis=c))
            best_feature.append((c, c_info_gain))
        #best features
        best_ = max(best_feature, key=lambda x: x[-1])
        return best_

    def train(self, train_data):
        """
        input:datasets D, features A, epsilon
        output:T
        """
        _, y_train, features = train_data.iloc[:, :-1], train_data.iloc[:, -1], train_data.columns[:-1]
        # if all the instance is the same class choose C_k for the label, return T
        if len(y_train.value_counts()) == 1:
            return Node(root=True,
                        label=y_train.iloc[0])

        # if A is a empty set
        if len(features) == 0:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        # decide max features
        max_feature, max_info_gain = self.info_gain_train(np.array(train_data))
        max_feature_name = features[max_feature]

        # check if less than threshold
        if max_info_gain < self.epsilon:
            return Node(root=True, label=y_train.value_counts().sort_values(ascending=False).index[0])

        # define subset
        node_tree = Node(root=False, feature_name=max_feature_name, feature=max_feature)

        feature_list = train_data[max_feature_name].value_counts().index
        for f in feature_list:
            sub_train_df = train_data.loc[train_data[max_feature_name] == f].drop([max_feature_name], axis=1)

            # recursive call to generate tree
            sub_tree = self.train(sub_train_df)
            node_tree.add_node(f, sub_tree)

        return node_tree

    def fit(self, train_data):
        self._tree = self.train(train_data)
        return self._tree

    def predict(self, X_test):
        return self._tree.predict(X_test)
datasets, labels = create_data()
data_df = pd.DataFrame(datasets, columns=labels)
dt = DTree()
tree = dt.fit(data_df)
print(dt.predict(['old', 'no', 'no', 'ordinary']))
print(dt.predict(['young', 'no', 'yes', 'ordinary']))
