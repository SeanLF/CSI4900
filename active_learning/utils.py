from os import environ
import pusher


def format_pusher_channel_name(channel_name):
    '''
    Formats a pusher channel name by:
    - Setting the user's username to the environment username
    - Setting the channel name with a prefix of '-dev' and the user's username

    Returns the channel name
    '''

    environment = environ['ENVIRONMENT']

    if environment == 'dev':
        user = environ['USER']
        channel_name += '-dev-' + user

    return channel_name


def get_pusher_client():
    '''
    Returns the pusher client using environment variables for the pusher's attributes
    '''

    return pusher.Pusher(
      app_id=environ['PUSHER_APP_ID'],
      key=environ['PUSHER_KEY'],
      secret=environ['PUSHER_SECRET'],
      ssl=True
    )
