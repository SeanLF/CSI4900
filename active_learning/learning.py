from django.urls import reverse
from libact.base.dataset import Dataset
from libact.query_strategies import QUIRE
from libact.models import SVM
from libact.models import Perceptron
from active_learning.models import Article
from active_learning.our_labeler import OurLabeler
from active_learning.fetch_data import tokenize_and_stem, tf_idf_matrix, feature_selection
import os

import pusher


class Learn:
    def __init__(self, **kwargs):
        self.pusher_client = pusher.Pusher(
          app_id=os.environ['PUSHER_APP_ID'],
          key=os.environ['PUSHER_KEY'],
          secret=os.environ['PUSHER_SECRET'],
          ssl=True
        )

    articles = Article.objects.all()

    def learn(self):
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
        tf_idf_matrix = tf_idf.fit_transform(X_train)

        # Use 𝛘² to select the k best features
        ch2 = SelectKBest(chi2, k=1000)
        X_train = ch2.fit_transform(tf_idf_matrix, y_train)
        all_feature_names = tf_idf.get_feature_names()
        feature_names = [all_feature_names[i] for i in ch2.get_support(indices=True)]

        # new_instance = tf_idf.transform([string])
        # new_instance = use ch2.transform(new_instance)

        dataset = Dataset(X_train, y_train)
        labeler = OurLabeler(label_name=['yes', 'maybe', 'no'], pusher_client=self.pusher_client)
        query_strategy = QUIRE(dataset)
        model = SVM()

        for _ in range(15):
            for _ in range(3):  # loop through the number of queries
                query_id = query_strategy.make_query()  # let the specified QueryStrategy suggest a data to query
                url = 'http://localhost:8000' + reverse('active_learning:detail', args=[lookup_table[query_id]])
                lbl = labeler.label({'url': url, 'id': query_id})  # query the label of the example at query_id
                dataset.update(query_id, lbl)  # update the dataset with newly-labeled example
            model.train(dataset)  # train model with newly-updated Dataset
