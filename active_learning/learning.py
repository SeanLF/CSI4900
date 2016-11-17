from django.urls import reverse
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
# from sklearn.linear_model import SGDClassifier
from .sgd_model import SGD_Model
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

from scipy.sparse import vstack
import numpy

from active_learning.utils import format_pusher_channel_name, get_pusher_client
from os import environ
import time


class Learn:
    '''
    The main class used for active learning.

    Included are the following active learning strategies (as part of the libact package):
    1. Active learning by learning
    2. Hint SVM
    3. Query by committee
    4. QUIRE
    5. Random sampling
    6. Uncertainty sampling
    7. Variance reduction
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

        self.articles = Article.objects.all()
        self.labels = {label.label: label.id for label in Label.objects.all()}
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
          ('vect', CountVectorizer()),
          ('tfidf', TfidfTransformer(stop_words='english')),
          # ('chi2', SelectKBest(chi2, k=1000)),
          ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, n_iter=5, random_state=42)),
        ])
        scores = cross_val_score(text_clf, X, y, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        return scores

    def learn(self, auto_label=True, active_learning_strategy=5, num_queries=50, train_size=0.005):
        '''
        Uses the specified learning strategy and then queries the oracle before printing the accuracy at which the articles are labeled

        Parameters
        ----------
        auto_label : boolean
            Boolean representing whether we want to use our own labeler or an ideal labeler
        active_learning_strategy : integer
            An integer representing the learning strategy to use
        num_queries : integer
            The number of queries for the oracle to label
        train_size : real number
            A real number, between 0 and 1, representing the percentage of the corpus to use in the labeled set
        '''
        scores = []

        X = numpy.array(TfidfVectorizer(stop_words='english').fit_transform(self.X).toarray())
        X_train, X_test, y_train, y_test, lookup_train, lookup_test = train_test_split(X, self.y, self.lookup_table, train_size=train_size, random_state=1, stratify=self.y)

        lookup_table = lookup_train + lookup_test
        X_train_test = numpy.array(vstack([X_train, X_test]).toarray())
        y_train_hidden_test = y_train + [None]*len(y_test)

        dataset = Dataset(numpy.array(vstack([X_train, X_test]).toarray()), y_train + y_train_hidden_test)
        test_set = Dataset(X_test, y_test)

        if auto_label is False:
            labeler = OurLabeler(labels=self.labels)
        else:
            labeler = IdealLabeler(dataset=Dataset(X_train_test, y_train + y_test))

        # stochastic gradient descent classifier
        model = SGD_Model(loss='hinge', penalty='l2', alpha=1e-4, n_iter=5, random_state=42)

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
            strategy = 'Uncertainty sampling'
            query_strategy = UncertaintySampling(dataset, model=LogisticRegression(C=0.1))
        else:
            strategy = 'Variance reduction'
            query_strategy = VarianceReduction(dataset)

        startTime = time.time()

        # Train classifier on a few instances, and see how accurate it is
        model.train(dataset)
        accuracy = model.score(test_set)
        labeled = str(dataset.len_labeled()) + '/' + str(dataset.len_unlabeled())
        scores.append(accuracy)
        self.show_score(accuracy, labeled)

        # query the oracle / active learning part
        for _ in range(num_queries):
            query_id = query_strategy.make_query()  # let the specified QueryStrategy suggest a data to query
            if auto_label is False:
                url = 'http://localhost:8000' + reverse('active_learning:detail', args=[lookup_table[query_id]])
                lbl = labeler.label({'url': url, 'id': lookup_table[query_id]})  # query the label of the example at query_id
            else:
                lbl = labeler.label(dataset.data[query_id][0])
            dataset.update(query_id, lbl)  # update the dataset with newly-labeled example
            model.train(dataset)  # train model with newly-updated Dataset
            accuracy = model.score(test_set)
            labeled = str(dataset.len_labeled()) + '/' + str(dataset.len_unlabeled())
            scores.append(accuracy)
            self.show_score(accuracy, labeled)

        elapsedtime = (time.time() - startTime)

        self.show_accuracy_plot(num_queries, scores, strategy, elapsedtime)

    def show_score(self, accuracy, labeled_string):
        pusher_client = get_pusher_client()
        channel_name = format_pusher_channel_name(environ['PRESENCE_CHANNEL_NAME'])

        pusher_client.trigger(channel_name, 'model_scored', {'accuracy': accuracy, 'labeled/unlabeled': labeled_string})
        print("Accuracy: ", accuracy, '\tLabeled: ', labeled_string)

    def show_accuracy_plot(self, num_queries, scores, strategy, time):
        pusher_client = get_pusher_client()
        channel_name = format_pusher_channel_name(environ['PRESENCE_CHANNEL_NAME'])
        labels = list(range(1, num_queries + 1))
        labels.insert(0, 'pre')
        pusher_client.trigger(channel_name, 'show_accuracy_over_queries', {'scores': scores, 'labels': labels, 'strategy': strategy, 'time': time})
