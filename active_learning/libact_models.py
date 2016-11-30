from sklearn.naive_bayes import MultinomialNB
from libact.base.interfaces import Model
import numpy as np
from sklearn.linear_model import SGDClassifier
from libact.base.interfaces import ContinuousModel


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
        return self.model.score(*(testing_dataset.format_sklearn() + args), **kwargs)


class SGD_Model(ContinuousModel):
    """
    An interface for sklearn's SGDClassifier.
    """
    def __init__(self, *args, **kwargs):
        self.model = SGDClassifier(*args, **kwargs)

    def train(self, dataset, *args, **kwargs):
        return self.model.fit(*(dataset.format_sklearn() + args), **kwargs)

    def predict(self, feature, *args, **kwargs):
        return self.model.predict(feature, *args, **kwargs)

    def score(self, testing_dataset, *args, **kwargs):
        return self.model.score(*(testing_dataset.format_sklearn() + args),
                                **kwargs)

    def predict_real(self, feature, *args, **kwargs):
        dvalue = self.model.decision_function(feature, *args, **kwargs)
        if len(np.shape(dvalue)) == 1:  # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            return dvalue
