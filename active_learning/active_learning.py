from libact.base.dataset import Dataset
from libact.query_strategies import QUIRE
from libact.models import LogisticRegression
from libact.models import Perceptron
from libact.base.interfaces import Labeler
from active_learning.models import Article

articles = Article.objects.all().values('cyber_security_occurences', 'hack_occurences', 'ip_occurences', 'breach_occurences')
articles = list(articles)

X = list(map(lambda x: list(x.values()), articles))
Y = [None]*len(X)
Y[0] = 'no'
Y[1] = 'yes'
Y[2] = 'maybe'

dataset = Dataset(X, Y)
labeler = OurLabeler()
query_strategy = QUIRE(dataset)
model = Perceptron()

for _ in range(100):  # loop through the number of queries
    query_id = query_strategy.make_query()  # let the specified QueryStrategy suggest a data to query
    lbl = labeler.label(dataset.data[query_id][0])  # query the label of the example at query_id
    dataset.update(query_id, lbl)  # update the dataset with newly-labeled example
    model.train(dataset)  # train model with newly-updated Dataset


class OurLabeler(Labeler):
    def label(self, feature):

        banner = "Enter the associated label with the article: "

        lbl = input(banner)

        while (self.label_name is not None) and (lbl not in self.label_name):
            print('Invalid label, please re-enter the associated label.')
            lbl = input(banner)

        return self.label_name.index(lbl)
