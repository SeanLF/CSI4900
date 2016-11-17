"""SGD_Model
An interface for scikit-learn's SGDClassifier
SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, n_iter=5, random_state=42)
"""
import logging
LOGGER = logging.getLogger(__name__)

import numpy as np
from sklearn.linear_model import SGDClassifier

from libact.base.interfaces import ContinuousModel


class SGD_Model(ContinuousModel):

    """C-Support Vector Machine Classifier
    References
    ----------
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
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
        if self.model.decision_function_shape != 'ovr':
            LOGGER.warn("Model support only 'ovr' for predict_real.")

        dvalue = self.model.decision_function(feature, *args, **kwargs)
        if len(np.shape(dvalue)) == 1:  # n_classes == 2
            return np.vstack((-dvalue, dvalue)).T
        else:
            return dvalue
