from libact.base.interfaces import Labeler
from django.urls import reverse
import pusherclient
from active_learning.utils import format_pusher_channel_name, get_pusher_client
from os import environ
from time import sleep
from threading import Semaphore


class WebLabeler(Labeler):
    '''
    Handles interaction between oracle and the process of labeling an article
    '''

    def __init__(self, **kwargs):
        '''
        Initializes the labeler by:

        1. Retrieving the list of labels from the given argument
        2. Getting the pusher client
        3. Getting the channel name for the pusher
        4. Handling the client's connection

        Parameters
        ----------
        labels : list of labels
        '''

        self.labels = kwargs.pop('labels', None)
        self.lookup_table = kwargs.pop('lookup_table', [])
        self.pusher_client = get_pusher_client()
        self.private_channel_name = kwargs.pop('private_channel_name', None)
        self.lbl = None
        self.semaphore = Semaphore(0)

        global pusher
        # listen for response from client, then disconnect
        pusher = pusherclient.Pusher(environ['PUSHER_KEY'], True, environ['PUSHER_SECRET'], {'user_id': 'active_learning_labeler'})
        pusher.connection.bind('pusher:connection_established', self.connect_handler)
        pusher.connect()

    def label(self, query_id):
        '''
        Return the label produced by the oracle
        '''

        article_id = self.lookup_table[query_id]
        pusher_data = {'id': article_id, 'url': environ['HOST'] + reverse('active_learning:detail', args=[article_id])}
        self.pusher_client.trigger(self.private_channel_name, 'request_label', pusher_data)

        # wait for the oracle to provide a label
        self.semaphore.acquire()

        label = self.labels[self.lbl]
        self.lbl = None
        return label

    def connect_handler(self, data):
        '''
        We can't subscribe until we've connected, so we use a callback handler to subscribe when able
        '''
        channel = pusher.subscribe(self.private_channel_name)
        channel.bind('client-label', self.callback)

    def callback(self, data):
        import json
        self.semaphore.release()
        self.lbl = json.loads(data)['label']


class AutoLabeler(Labeler):
    """
    Uses the ground truth to label instances
    """

    def __init__(self, oracle_dataset):
        self.oracle_dataset = oracle_dataset

    def label(self, query_id):
        """
        Return the label from the oracle dataset (automatic label)
        """
        return self.oracle_dataset.data[query_id][1]
