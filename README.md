# Honours project

## Setup
After pulling the repository, you need to ensure that you have python3 (and pip3) installed.

### Dependencies
[*libact*](https://github.com/ntucllab/libact#basic-dependencies) and [*newspaper*](https://github.com/codelucas/newspaper/#get-it-now) require specific libraries to be installed. The commands to install them will depend on your choice of OS. Check their GitHub readme for details.

  To install dependencies, run
  ```bash
  pip3 install -r requirements.txt
  ```

### Database
If you don't have a database yet, or you have pending migrations, run
```bash
python3 manage.py migrate
```

### Environment
Create a `.env` file in the project root folder.
```.env
ENVIRONMENT=dev
PUSHER_APP_ID=_________________
PUSHER_KEY=_________________
PUSHER_SECRET=_________________
BING_NEWS_SEARCH_API_KEY=_________________
PRESENCE_CHANNEL_NAME=presence-channel
```
Replace the underscores with your environment secrets. Contact a team member if you don't have access to Pusher or the Bing News API.

## Running
To run the server, use
```
python3 manage.py runserver
```

To start the learning process, open a browser tab and go to `/`. Click on the LEARN button to start learning.

---

## Acquiring data
### Bing dataset
Make a GET request to `/get_articles` with the following params:
- `search_query`: one or more words to get links from the Bing News Search API (if more than one word, join them with a `+`, ex: `search_query=data+breach`)
- `max_results`: an integer for the maximum number of articles you want to fetch (ex: `max_results=1000`)
- `label`: you assume that all articles fetched from Bing News are relevant to your label. This should be a word (ex: `label=yes`)

Full example: `/get_articles?search_query=data+breach&max_results=300&label=yes`

### Four University dataset
To import the Four University dataset
- Download the dataset from this unofficial [source](https://raw.githubusercontent.com/daniel-cloudspace/WekaPres/master/datasets/webkb-data.gtar.gz)
- Extract it to the top level directory of this repository (`/webkb`)
- Make a GET request to `/load_four_university`
