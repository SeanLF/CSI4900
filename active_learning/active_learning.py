from django.urls import reverse
from libact.base.dataset import Dataset
from libact.query_strategies import QUIRE
from libact.models import LogisticRegression
from libact.models import Perceptron
from active_learning.models import Article
from active_learning.our_labeler import OurLabeler

articles = Article.objects.all().values('cyber_security_occurences', 'hack_occurences', 'ip_occurences', 'breach_occurences', 'id', 'security_breach')

lookup_table = []
X = []
Y = []

for article in articles:
    lookup_table.append(article['id'])
    X.append([float(article['breach_occurences']), float(article['cyber_security_occurences']),
              float(article['hack_occurences']), float(article['ip_occurences'])])
    Y.append(article['security_breach'])

dataset = Dataset(X, Y)
labeler = OurLabeler(label_name=['yes', 'maybe', 'no'])
query_strategy = QUIRE(dataset)
model = Perceptron()

for _ in range(100):  # loop through the number of queries
    query_id = query_strategy.make_query()  # let the specified QueryStrategy suggest a data to query
    url = 'http://localhost:8000' + reverse('active_learning:detail', args=[lookup_table[query_id]])
    lbl = labeler.label(url)  # query the label of the example at query_id
    dataset.update(query_id, lbl)  # update the dataset with newly-labeled example
    model.train(dataset)  # train model with newly-updated Dataset
