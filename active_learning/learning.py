from django.urls import reverse
from libact.base.dataset import Dataset
from libact.query_strategies import *
from libact.models import *
from libact.models import Perceptron
from active_learning.models import Article, Label
from libact.labelers.ideal_labeler import IdealLabeler
from active_learning.our_labeler import OurLabeler
from active_learning.fetch_data import tokenize_and_stem, tf_idf_matrix, feature_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import numpy


class Learn:
    def __init__(self, **kwargs):
        self.articles = Article.objects.all()

    def learn(self, train_percentage=0.1, num_queries=10, use_chi=False, num_features=1000, active_learning_strategy=4, web_oracle=True):
        lookup_table = []
        X_train = []
        y_train = []

        # initialize arrays
        for article in self.articles:
            lookup_table.append(article.id)
            X_train.append(article.text)
            y_train.append(article.class_label_id)

        # Use scikit-learn to generate a matrix[document][feature] = tf-idf for feature
        tf_idf = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words='english')
        X_train = tf_idf.fit_transform(X_train)

        # Use 𝛘² to select the k best features
        if use_chi is True:
            ch2 = SelectKBest(chi2, k=num_features)
            X_train = ch2.fit_transform(X_train, y_train)
            all_feature_names = tf_idf.get_feature_names()
            feature_names = [all_feature_names[i] for i in ch2.get_support(indices=True)]

        # new_instance = tf_idf.transform([string])
        # new_instance = use ch2.transform(new_instance)

        labels = {label.label: label.id for label in Label.objects.all()}

        # TODO: shuffling messes with the lookup table
        # HACK: shuffle data for random sampling
        X_train = numpy.array(X_train.toarray())
        zipped = []
        for index in range(0, len(y_train)):
            zipped.append([X_train[index], y_train[index]])
        numpy.random.shuffle(zipped)
        X_train, y_train = zip(*zipped)
        # HACK: end
        X_test = X_train
        y_test = y_train
        test_set = Dataset(X_test, y_test)

        # split data
        num_labeled_to_take = int(len(y_train) * train_percentage)
        temp = y_train[:num_labeled_to_take]
        spliced_y_train = list(temp) + [None]*(len(y_train) - num_labeled_to_take)
        dataset = Dataset(X_train, spliced_y_train)

        if web_oracle is True:
            labeler = OurLabeler(labels=labels)
        else:
            labeler = IdealLabeler(dataset=Dataset(X_train, y_train))

        # choose an active learning strategy
        if active_learning_strategy == 1:
            query_strategy = ActiveLearningByLearning(dataset, query_strategies=[UncertaintySampling(dataset, model=LogisticRegression(C=1.)), UncertaintySampling(dataset, model=LogisticRegression(C=.01)), HintSVM(dataset)], model=LogisticRegression())
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

        model = SVM()

        # query the oracle / active learning part
        for _ in range(num_queries):
            query_id = query_strategy.make_query()  # let the specified QueryStrategy suggest a data to query
            url = 'http://localhost:8000' + reverse('active_learning:detail', args=[lookup_table[query_id]])
            # lbl = labeler.label({'url': url, 'id': query_id})  # query the label of the example at query_id
            lbl = labeler.label(dataset.data[query_id][0])
            dataset.update(query_id, lbl)  # update the dataset with newly-labeled example
            model.train(dataset)  # train model with newly-updated Dataset
            # Quickly print score
            print('score: ', model.score(test_set))
