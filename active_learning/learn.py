import numpy

from scipy.sparse import vstack

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

from libact.base.dataset import Dataset
from libact.query_strategies import *
from libact.models import *

from active_learning.models import Article, Label, DataSet
from active_learning.libact_models import SGD_Model
from active_learning.custom_labelers import AutoLabeler, WebLabeler

import time


def train_active_learning(**kwargs):
    # Params
    num_queries = kwargs.pop('num_queries', None)
    training_dataset = kwargs.pop('training_dataset', None)
    labeler = kwargs.pop('labeler', None)
    active_learning_strategy = kwargs.pop('active_learning_strategy', 1)
    options = kwargs.pop('options', {'intermediate_testing': False})
    # Setup active-learning strategy
    query_strategy, options['strategy'] = get_active_learning_strategy(active_learning_strategy, training_dataset, max_queries=num_queries)
    # Create model
    model = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-2, n_iter=1000, random_state=None, learning_rate='optimal', class_weight='balanced')
    # train on labeled instances
    model.fit(*(training_dataset.format_sklearn()))
    # test if we want intermediate training results
    if options['intermediate_testing'] is True:
        options['intermediate_results'].append(model.predict(options['X_test']))
    # active learning
    options['active_learning_training_time'] = 0
    for query_num in range(num_queries):
        start_time = time.time()
        # query the active learning strategy
        query_id = query_strategy.make_query()
        # use the oracle to label the query
        label = labeler.label(query_id)
        # update dataset with label then retrain the model
        training_dataset.update(query_id, label)
        model.fit(*(training_dataset.format_sklearn()))
        options['active_learning_training_time'] += time.time() - start_time
        if options['intermediate_testing'] is True:
            options['intermediate_results'].append(model.predict(options['X_test']))
    return model, options


def cross_validate_active_learning(num_folds, X, y, labeled_instances, num_queries, **kwargs):
    active_learning_strategy = kwargs.pop('active_learning_strategy', 1)
    results = []
    # make K folds
    skf = StratifiedKFold(n_splits=num_folds)
    # for each fold
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # hide all but labeled_instances in training set
        X_train, X_train_unlabeled, y_train, y_train_oracle = train_test_split(X_train, y_train, train_size=labeled_instances, random_state=None, stratify=y_train)
        # join X to obtain semi-labeled dataset
        X_train = numpy.array(vstack([X_train, X_train_unlabeled]).toarray())
        # fill in y_train with missing labels for active learning on semi-labeled dataset
        y_train = y_train.tolist()
        y_train_with_missing = y_train + [None] * len(y_train_oracle)
        # build dataset object for libact library
        training_dataset = Dataset(X_train, y_train_with_missing)
        oracle_training_dataset = Dataset(X_train, y_train + y_train_oracle.tolist())
        # do active_learning and get trained model
        trained_model, _ = train_active_learning(num_queries=num_queries, training_dataset=training_dataset, labeler=AutoLabeler(oracle_training_dataset), active_learning_strategy=active_learning_strategy)
        # test model
        predicted = trained_model.predict(X_test)
        results.append({'y_true': y_test, 'y_pred': predicted})
    return results


def preprocess(X_strings):
    return numpy.array(TfidfVectorizer(stop_words='english').fit_transform(X_strings).toarray())


def learn(dataset_id, **kwargs):
    num_queries = kwargs.pop('num_queries', 20)
    num_folds = kwargs.pop('num_folds', 10)
    labeled_instances = kwargs.pop('labeled_instances', 5)
    active_learning_strategy = kwargs.pop('active_learning_strategy', 1)
    # Build dataset
    articles = Article.objects.filter(dataset_id=dataset_id)
    label_ids = list(articles.values_list('class_label_id', flat=True).distinct())
    labels = {label.label: label.id for label in Label.objects.filter(id__in=label_ids)}
    lookup_table = []
    X_strings = []
    y = []
    for article in articles:
        lookup_table.append(article.id)
        X_strings.append(article.text)
        y.append(article.class_label_id)
    X = preprocess(X_strings)
    y = numpy.array(y)
    return cross_validate_active_learning(num_folds, X, y, labeled_instances, num_queries, active_learning_strategy=active_learning_strategy)


def get_active_learning_strategy(active_learning_strategy, dataset, **kwargs):
    '''
    Active learning strategies are:

    1. Active learning by learning
    2. Active learning by learning (with QBC)
    3. Query by committee
    4. Uncertainty sampling
    5. Random sampling
    6. QUIRE
    7. Uncertainty sampling (smallest margin)
    8. Hint SVM
    9. Variance reduction
    '''

    max_queries = kwargs.pop('max_queries', 100)
    if max_queries == 0:
        max_queries = 100

    # choose an active learning strategy
    if active_learning_strategy == 1:
        strategy = 'Active learning by learning'
        query_strategy = ActiveLearningByLearning(
            dataset,
            query_strategies=[
                UncertaintySampling(dataset, model=SGD_Model(loss='hinge', penalty='l2', alpha=1e-2, n_iter=1000, random_state=None, learning_rate='optimal', class_weight='balanced')),
                UncertaintySampling(dataset, model=SGD_Model(loss='hinge', penalty='l2', alpha=1e-3, n_iter=1000, random_state=None, learning_rate='optimal', class_weight='balanced')),
            ],
            model=SGD_Model(loss='hinge', penalty='l2', alpha=1e-2, n_iter=1000, random_state=None, learning_rate='optimal', class_weight='balanced'),
            T=max_queries)
    elif active_learning_strategy == 2:
        strategy = 'Active learning by learning (with QBC)'
        query_strategy = ActiveLearningByLearning(
            dataset,
            query_strategies=[
                UncertaintySampling(dataset, model=SGD_Model(loss='hinge', penalty='l2', alpha=1e-2, n_iter=1000, random_state=None, learning_rate='optimal', class_weight='balanced')),
                UncertaintySampling(dataset, model=SGD_Model(loss='hinge', penalty='l2', alpha=1e-3, n_iter=1000, random_state=None, learning_rate='optimal', class_weight='balanced')),
                QueryByCommittee(dataset, models=[SGD_Model(alpha=1e-2, n_iter=1000), SGD_Model(alpha=1e-3, n_iter=1000)]),
            ],
            model=SGD_Model(loss='hinge', penalty='l2', alpha=1e-2, n_iter=1000, random_state=None, learning_rate='optimal', class_weight='balanced'),
            T=1000)
    elif active_learning_strategy == 3:
        strategy = 'Query by committee'
        query_strategy = QueryByCommittee(dataset, models=[SGD_Model(alpha=1e-2, n_iter=1000), SGD_Model(alpha=1e-3, n_iter=1000)])
    elif active_learning_strategy == 4:
        strategy = 'Uncertainty sampling least confidence'
        query_strategy = UncertaintySampling(dataset, model=SVM(kernel='linear', decision_function_shape='ovr'))
    elif active_learning_strategy == 5:
        strategy = 'Random sampling'
        query_strategy = RandomSampling(dataset)
    elif active_learning_strategy == 6:
        strategy = 'QUIRE'
        query_strategy = QUIRE(dataset)
    elif active_learning_strategy == 7:
        strategy = 'Uncertainty sampling smallest margin'
        query_strategy = UncertaintySampling(dataset, method='sm', model=SVM(kernel='linear', decision_function_shape='ovr'))
    elif active_learning_strategy == 8:
        strategy = 'Hint SVM'
        query_strategy = HintSVM(dataset, Cl=0.01, p=0.8)
    else:
        strategy = 'Variance reduction'
        query_strategy = VarianceReduction(dataset)

    return query_strategy, strategy
