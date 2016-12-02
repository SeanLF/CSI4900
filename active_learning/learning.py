from libact.base.dataset import Dataset
from libact.query_strategies import *
from libact.models import *

from active_learning.custom_labelers import WebLabeler, AutoLabeler
from active_learning.libact_models import *
from active_learning.models import Article, Label
from active_learning.learn import get_active_learning_strategy, train_active_learning

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

from scipy.sparse import vstack

import numpy
from numpy import argmax, argmin

from active_learning.utils import format_pusher_channel_name, get_pusher_client
from os import environ
import time


class Learn:
    '''
    The main class used for active learning.
    '''

    def __init__(self, **kwargs):
        '''
        Initializes the learner by:

        1. Getting a list of articles
        2. Getting a list of labels
        3. For each article:
            a. Adding the article id to a lookup table
            b. adding the article text to an array, X
            c. adding the article class label (id) to an array Y
        '''

        self.articles = Article.objects.filter(dataset_id=kwargs.pop('dataset_id', 1))
        self.label_ids = list(self.articles.values_list('class_label_id', flat=True).distinct())
        self.label_ids.sort()
        self.labels = {label.label: label.id for label in Label.objects.filter(id__in=self.label_ids)}
        self.lookup_table = []
        self.X = []
        self.y = []
        for article in self.articles:
            self.lookup_table.append(article.id)
            self.X.append(article.text)
            self.y.append(article.class_label_id)

    def classify_svm(self, X, y):
        '''
        Uses SVM to classify articles before calculating and printing the accuracy

        Parameters
        ----------
        X : array (of strings)
            An array of strings containing the text of each article
        y : array (of integers)
            An array of integers containing the class label id of each article
        '''

        text_clf = Pipeline([
          ('tfidf', TfidfVectorizer(stop_words='english')),
          ('chi2', SelectKBest(chi2, k=1000)),
          ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, n_iter=5, random_state=None)),
        ])
        scores = cross_val_score(text_clf, X, y, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        return scores

    def learn(self, **kwargs):
        '''
        Uses the specified learning strategy and then queries the oracle before printing the accuracy at which the articles are labeled

        Parameters
        ----------
        auto_label : (boolean) use auto-labeler or manual web labeler
        active_learning_strategy : (integer) the learning strategy to use
        num_queries : (integer) # queries for the oracle to label
        train_size : (0 <= number <= 1) fraction of the corpus to use in the labeled set
        '''

        auto_label = kwargs.pop('auto_label', False)
        active_learning_strategy = kwargs.pop('active_learning_strategy', 1)
        num_queries = kwargs.pop('num_queries', 50)
        train_size = kwargs.pop('train_size', 0.005)
        self_train = kwargs.pop('self_train', True)
        verbose = kwargs.pop('verbose', False)

        metrics = ['precision', 'recall', 'fbeta']
        results = self.setup_results(self.labels, metrics)

        # Extract features from documents in the form of TF-IDF
        X = numpy.array(TfidfVectorizer(stop_words='english').fit_transform(self.X).toarray())

        train_dataset, train_dataset_labeled, test_dataset, lookup_train, lookup_test, self.lookup_table = self.setup_datasets(X, self.y, self.lookup_table, train_size)

        labeler = AutoLabeler(train_dataset_labeled) if auto_label else WebLabeler(labels=self.labels, lookup_table=lookup_train)

        X_test, y_test = test_dataset.format_sklearn()
        options = {'intermediate_testing': True, 'intermediate_results': [], 'X_test': X_test}
        model, opt = train_active_learning(num_queries=num_queries, training_dataset=train_dataset, labeler=labeler, active_learning_strategy=active_learning_strategy, options=options)

        elapsed_time = opt['active_learning_training_time']
        strategy = opt['strategy']
        predictions = opt['intermediate_results']  # first prediction is before active learning, num_queries intermediate preductions (last one is final prediction)

        for y_pred in predictions:
            self.get_results(y_test, y_pred, results, metrics)

        if self_train:
            instances_to_label_per_loop = 20
            test_size = 0.25
            self.self_train(train_dataset, model, instances_to_label_per_loop, test_size)
            X_test, y_test, y_pred = self.test_and_get_results(model, test_dataset, results, metrics)

        conf_matrix = confusion_matrix(y_test, predictions[-1], labels=self.label_ids).tolist()
        self.send_results_to_display(num_queries, strategy, elapsed_time, results, conf_matrix, metrics)
        if verbose:
            print("Fin")

    def setup_results(self, labels, metrics):
        results = {}
        for label, label_id in labels.items():
            results[label_id] = {}
            for metric in metrics:
                results[label_id][metric] = []
            results[label_id]['label'] = label
        return results

    def setup_datasets(self, X, y, lookup_table, train_size):
        X_train, X_test, y_train, y_test, lookup_train, lookup_test \
            = train_test_split(X, y, lookup_table, train_size=0.7, random_state=None, stratify=y)
        X_train_labeled, X_train_unlabeled, y_train_labeled, y_train_unlabeled, lookup_train_labeled, lookup_train_unlabeled \
            = train_test_split(X_train, y_train, lookup_train, train_size=train_size, random_state=None, stratify=y_train)
        lookup_table = lookup_train_labeled + lookup_train_unlabeled + lookup_test
        lookup_train = lookup_train_labeled + lookup_train_unlabeled
        X_train = numpy.array(vstack([X_train_labeled, X_train_unlabeled]).toarray())
        y_train = y_train_labeled + y_train_unlabeled

        train_dataset = Dataset(X_train, y_train_labeled + [None] * len(y_train_unlabeled))
        train_dataset_labeled = Dataset(X_train, y_train)
        test_dataset = Dataset(X_test, y_test)
        return train_dataset, train_dataset_labeled, test_dataset, lookup_train, lookup_test, lookup_table

    def get_data(self, dataset):
        entries = dataset.get_entries()
        X, y = zip(*entries)
        return X, y

    def self_train(self, train_test_dataset, model, instances_to_label_per_loop, test_size):
        num_instances = len(train_test_dataset.data)
        num_loops = (train_test_dataset.len_unlabeled() - (test_size * num_instances))/instances_to_label_per_loop

        for self_train_loop_index in range(int(num_loops)):
            entries = train_test_dataset.get_unlabeled_entries()
            unlabeled_entry_ids, X_test = zip(*entries)
            confidence_scores = model.decision_function(X_test)
            entry_id = argmin(confidence_scores)
            if confidence_scores[entry_id] < 0:
                train_test_dataset.update(entry_id, self.label_ids[0])
            entry_id = argmax(confidence_scores)
            if confidence_scores[entry_id] > 0:
                train_test_dataset.update(entry_id, self.label_ids[1])
            model.fit(*(train_test_dataset.format_sklearn()))
            print(self_train_loop_index, '/', (int(num_loops)-1))

    def test_and_get_results(self, model, test_dataset, results, metrics):
        X_test, y_test = self.get_data(test_dataset)
        y_pred = model.predict(X_test)
        self.get_results(y_test, y_pred, results, metrics)
        return X_test, y_test, y_pred

    def get_results(self, y_true, y_pred, results_dict, metrics):
        # Obtain measures and append to their arrays
        p, r, fb, _ = precision_recall_fscore_support(y_true,  y_pred, labels=self.label_ids)
        for i, array in enumerate([p, r, fb]):
            for j, label_id in enumerate(self.label_ids):
                results_dict[label_id][metrics[i]].append(round(array[j], 5))

    def send_results_to_display(self, num_queries, strategy, time, results, confusion_matrix, metrics):
        pusher_client = get_pusher_client()
        channel_name = format_pusher_channel_name(environ['PRESENCE_CHANNEL_NAME'])
        pusher_data = {
            'strategy': strategy, 'time': time, 'results': results, 'confusion_matrix': confusion_matrix, 'metrics': metrics
        }
        pusher_client.trigger(channel_name, 'show_accuracy_over_queries', pusher_data)
