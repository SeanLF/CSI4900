from libact.base.dataset import Dataset
from libact.query_strategies import *
from libact.models import *
from active_learning.models import Article, Label
from libact.labelers.ideal_labeler import IdealLabeler
from active_learning.our_labeler import OurLabeler

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

from scipy.sparse import vstack
import numpy

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

        metrics = ['precision', 'recall', 'fbeta']
        results = self.setup_results(self.labels, metrics)

        # Extract features from documents in the form of TF-IDF
        X = numpy.array(TfidfVectorizer(stop_words='english').fit_transform(self.X).toarray())

        train_test_dataset, ground_truth_dataset, self.lookup_table = self.setup_datasets(X, self.y, self.lookup_table, train_size)

        if auto_label is False:
            labeler = OurLabeler(labels=self.labels)
        query_strategy, strategy = self.get_active_learning_strategy(active_learning_strategy, train_test_dataset)
        model = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-2, n_iter=5, random_state=None, learning_rate='optimal', class_weight='balanced')

        # Train on the labeled data and get initial accuracy
        model.fit(*(train_test_dataset.format_sklearn()))
        X_test, y_test = self.get_test_data(train_test_dataset, ground_truth_dataset)
        y_pred = model.predict(X_test)
        self.get_results(y_test, y_pred, results, metrics)

        elapsed_time = 0  # time duration spent taken querying and retraining

        # query the oracle / active learning part
        for query_number in range(1, num_queries + 1):
            start_time = time.time()
            query_id = query_strategy.make_query()
            label = ground_truth_dataset.data[query_id][1] if auto_label else labeler.label(self.lookup_table[query_id])

            # update the dataset with newly-labeled example then re-train
            train_test_dataset.update(query_id, label)
            model.fit(*(train_test_dataset.format_sklearn()))
            elapsed_time += time.time() - start_time

            # Test for intermediate and final metrics
            X_test, y_test = self.get_test_data(train_test_dataset, ground_truth_dataset)
            y_pred = model.predict(X_test)
            self.get_results(y_test, y_pred, results, metrics)

        conf_matrix = confusion_matrix(y_test, y_pred, labels=self.label_ids).tolist()
        self.send_results_to_display(num_queries, strategy, elapsed_time, results, conf_matrix, metrics)

    def setup_results(self, labels, metrics):
        results = {}
        for label, label_id in labels.items():
            results[label_id] = {}
            for metric in metrics:
                results[label_id][metric] = []
            results[label_id]['label'] = label
        return results

    def setup_datasets(self, X, y, lookup_table, train_size):
        X_train, X_test, y_train, y_test, lookup_train, lookup_test = train_test_split(X, y, lookup_table, train_size=train_size, random_state=None, stratify=y)

        lookup_table = lookup_train + lookup_test
        X = numpy.array(vstack([X_train, X_test]).toarray())
        y = y_train + y_test

        train_test_dataset = Dataset(X, y_train + [None] * len(y_test))
        ground_truth_dataset = Dataset(X, y)
        return train_test_dataset, ground_truth_dataset, lookup_table

    def get_active_learning_strategy(self, active_learning_strategy, dataset):
        '''
        Active learning strategies are:

        1. Active learning by learning
        2. Hint SVM
        3. Query by committee
        4. QUIRE
        5. Random sampling
        6. Uncertainty sampling
        7. Uncertainty sampling (smallest margin)
        8. Variance reduction
        '''

        # choose an active learning strategy
        if active_learning_strategy == 1:
            strategy = 'Active learning by learning'
            query_strategy = ActiveLearningByLearning(
                dataset,
                query_strategies=[
                    UncertaintySampling(dataset, model=LogisticRegression(C=1.)),
                    UncertaintySampling(dataset, model=LogisticRegression(C=.01))
                ],
                model=LogisticRegression(),
                T=1000)
        elif active_learning_strategy == 2:
            strategy = 'Hint SVM'
            query_strategy = HintSVM(dataset, Cl=0.01, p=0.8)
        elif active_learning_strategy == 3:
            strategy = 'Query by committee'
            query_strategy = QueryByCommittee(dataset, models=[LogisticRegression(C=1.0), LogisticRegression(C=0.1)])
        elif active_learning_strategy == 4:
            strategy = 'QUIRE'
            query_strategy = QUIRE(dataset)
        elif active_learning_strategy == 5:
            strategy = 'Random sampling'
            query_strategy = RandomSampling(dataset)
        elif active_learning_strategy == 6:
            strategy = 'Uncertainty sampling least confidence'
            query_strategy = UncertaintySampling(dataset, model=LogisticRegression(C=0.1))
        elif active_learning_strategy == 7:
            strategy = 'Uncertainty sampling smallest margin'
            query_strategy = UncertaintySampling(dataset, method='sm', model=LogisticRegression(C=0.1))
        else:
            strategy = 'Variance reduction'
            query_strategy = VarianceReduction(dataset)

        return query_strategy, strategy

    def get_test_data(self, train_test_dataset, ground_truth_dataset):
        entries = train_test_dataset.get_unlabeled_entries()
        unlabeled_entry_ids, X_test = zip(*entries)
        y_test = [ground_truth_dataset.data[entry_id][1] for entry_id in unlabeled_entry_ids]
        return X_test, y_test

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
