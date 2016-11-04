from os import environ
import pusher


def format_pusher_channel_name(channel_name):
    environment = environ['ENVIRONMENT']

    if environment == 'dev':
        user = environ['USER']
        channel_name += '-dev-' + user

    return channel_name


def get_pusher_client():
    return pusher.Pusher(
      app_id=environ['PUSHER_APP_ID'],
      key=environ['PUSHER_KEY'],
      secret=environ['PUSHER_SECRET'],
      ssl=True
    )
