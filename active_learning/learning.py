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
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

from scipy.sparse import vstack

import numpy


class Learn:
    def __init__(self, **kwargs):
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
        text_clf = Pipeline([
          ('vect', CountVectorizer()),
          ('tfidf', TfidfTransformer(stop_words='english')),
          # ('chi2', SelectKBest(chi2, k=1000)),
          ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
        ])
        scores = cross_val_score(text_clf, X, y, cv=10)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        return scores

    def learn(self, auto_label=True, active_learning_strategy=5, num_queries=50, train_size=0.005):
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

        # choose an active learning strategy
        if active_learning_strategy == 1:
            query_strategy = ActiveLearningByLearning(
                dataset,
                query_strategies=[
                    UncertaintySampling(dataset, model=LogisticRegression(C=1.)),
                    UncertaintySampling(dataset, model=LogisticRegression(C=.01)),
                    HintSVM(dataset)
                ],
                model=LogisticRegression())
        elif active_learning_strategy == 2:
            query_strategy = HintSVM(dataset, Cl=0.01, p=0.8)
        elif active_learning_strategy == 3:
            query_strategy = QueryByCommittee(dataset, models=[LogisticRegression(C=1.0), LogisticRegression(C=0.1)])
        elif active_learning_strategy == 4:
            query_strategy = QUIRE(dataset)
        elif active_learning_strategy == 5:
            query_strategy = RandomSampling(dataset)
        elif active_learning_strategy == 6:
            query_strategy = UncertaintySampling(dataset, model=LogisticRegression(C=0.1))
        else:
            query_strategy = VarianceReduction(dataset)

        model = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)

        model.fit(*(dataset.format_sklearn()))
        print("Accuracy: %0.2f" % (model.score(X_test, y_test)))

        # query the oracle / active learning part
        for _ in range(num_queries):
            query_id = query_strategy.make_query()  # let the specified QueryStrategy suggest a data to query
            if auto_label is False:
                url = 'http://localhost:8000' + reverse('active_learning:detail', args=[lookup_table[query_id]])
                lbl = labeler.label({'url': url, 'id': lookup_table[query_id]})  # query the label of the example at query_id
            else:
                lbl = labeler.label(dataset.data[query_id][0])
            dataset.update(query_id, lbl)  # update the dataset with newly-labeled example
            model.fit(*(dataset.format_sklearn()))  # train model with newly-updated Dataset
            # Quickly print score
            a = Article.objects.get(id=lookup_table[query_id])
            print("Accuracy: %0.2f" % (model.score(X_test, y_test)), '\tLabel:', a.class_label.label, '\tTitle:', a.title)
