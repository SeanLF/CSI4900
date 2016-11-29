from sklearn.naive_bayes import MultinomialNB

from libact.base.interfaces import Model


class MNB_Model(Model):
    """
    An interface for sklearn's MultinomialNB classifier.
    """
    def __init__(self, *args, **kwargs):
        self.model = MultinomialNB(*args, **kwargs)

    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)