from libact.base.interfaces import Labeler
import pusherclient
from .utils import format_pusher_channel_name, get_pusher_client
from os import environ


class OurLabeler(Labeler):
    lbl = None

    def __init__(self, **kwargs):
        self.labels = kwargs.pop('labels', None)
        self.pusher_client = get_pusher_client()
        self.channel_name = format_pusher_channel_name(environ['PRESENCE_CHANNEL_NAME'])

        global pusher
        # listen for response from client, then disconnect
        pusher = pusherclient.Pusher(environ['PUSHER_KEY'], True, environ['PUSHER_SECRET'], {'user_id': 'active_learning_labeler'})
        pusher.connection.bind('pusher:connection_established', self.connect_handler)
        pusher.connect()

    # Feature is a dictionary with url and id as keys
    def label(self, feature):
        self.pusher_client.trigger(self.channel_name, 'request_label', feature)

        while (self.labels is not None) and (self.lbl not in self.labels.keys()):
            from time import sleep
            sleep(1)

        label = self.labels[self.lbl]
        self.lbl = None
        return label

        # We can't subscribe until we've connected, so we use a callback handler
        # to subscribe when able
    def connect_handler(self, data):
        channel = pusher.subscribe(self.channel_name)
        channel.bind('client-label', self.callback)

    def callback(self, data):
        import json
        self.lbl = json.loads(data)['label']
