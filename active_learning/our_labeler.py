from libact.base.interfaces import Labeler
import pusherclient


class OurLabeler(Labeler):
    lbl = None

    def __init__(self, **kwargs):
        self.label_name = kwargs.pop('label_name', None)
        self.pusher_client = kwargs.pop('pusher_client', None)
        global pusher
        # listen for response from client, then disconnect
        # HACK: shouldn't get key and secret like this
        pusher = pusherclient.Pusher(self.pusher_client._pusher_client.key, True, self.pusher_client._pusher_client.secret, {'user_id': 'learner'})
        pusher.connection.bind('pusher:connection_established', self.connect_handler)
        pusher.connect()

    # Feature is a dictionary with url and id as keys
    def label(self, feature):
        self.pusher_client.trigger('presence-channel', 'request_label', feature)

        while (self.label_name is not None) and (self.lbl not in self.label_name):
            from time import sleep
            sleep(1)

        lbl = self.lbl
        self.lbl = None
        return self.label_name.index(lbl)

        # We can't subscribe until we've connected, so we use a callback handler
        # to subscribe when able
    def connect_handler(self, data):
        channel = pusher.subscribe('presence-channel')
        channel.bind('client-label', self.callback)

    def callback(self, data):
        import json
        self.lbl = json.loads(data)['label']
