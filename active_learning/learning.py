from django.urls import reverse
from libact.base.dataset import Dataset
from libact.query_strategies import QUIRE
from libact.models import SVM
from libact.models import Perceptron
from active_learning.models import Article
from active_learning.our_labeler import OurLabeler
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
        X = []
        Y = []

        for article in self.articles:
            lookup_table.append(article['id'])
            X.append([0, 0, 0, 0])
            Y.append(article['security_breach'])

        dataset = Dataset(X, Y)
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
