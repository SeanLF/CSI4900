# for acquire data
import os
import requests
# for download_and_extract_text_from_url
import newspaper
# for tokenize_and_stem
import re
from nltk.stem.porter import PorterStemmer
# for tf_idf_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
# for feature_selection
from sklearn.feature_selection import SelectKBest, chi2
import random
from bs4 import BeautifulSoup


def bing_news_search(query, max_results=100, offset=0):
    '''
    Use Bing API to query Bing news and return a JSON response

    Parameters
    ----------
    query : string
        The query term to be used
    max_results : integer
        The number of results expected per page
    offset : integer
        The number of articles to skip over from the start
    '''

    url = 'https://api.cognitive.microsoft.com/bing/v5.0/news/search'
    # query string parameters
    payload = {'q': query, 'count': str(max_results), 'safeSearch': 'Off', 'mkt': 'en-ca', 'offset': str(offset), 'freshness': 'Day'}
    # custom headers
    headers = {'Ocp-Apim-Subscription-Key': os.environ['BING_NEWS_SEARCH_API_KEY']}
    # make GET request
    r = requests.get(url, params=payload, headers=headers)
    # get JSON response
    return r.json()


def get_links(search_query, max_results):
    '''
    Loop through the returned results and add the urls to a list

    Parameters
    ----------
    search_query : string
        The search query term to be used
    max_results : integer
        The final expected number of items to fetch
    '''

    urls = []
    while len(urls) < max_results:
        json = bing_news_search(search_query, max_results, len(urls))
        # If the url array is still empty then find the minimum between the given max_results and the returned estimate
        if urls == []:
            max_results = min(max_results, json['totalEstimatedMatches'])
        urls += [search_result['url'] for search_result in json['value']]
    return urls


def download_articles(articles):
    '''
    Loop through the returned results and add the urls to a list

    Parameters
    ----------
    articles : list of Article
        The list of Articles that were found
    '''

    urls = [a.url for a in articles]
    failed_articles = []
    filled_requests = newspaper.network.multithread_request(urls)
    # Note that the responses are returned in original order
    for index, req in enumerate(filled_requests):
        html = newspaper.network.get_html(req.url, response=req.resp)
        articles[index].set_html(html)
        if not req.resp:
            failed_articles.append(articles[index])
    articles = [a for a in articles if a.html]


def clean_string(string):
    return string.lower().replace('\n', ' ')


def tokenize_and_stem(string):
    '''
    Return the given string after splitting it up, word by word, then getting the stem of each work

    Getting the stem of words allows for easier identification of a word in all of its forms
    '''

    stems = []
    stemmer = PorterStemmer()

    # Tokenize the string using a regular expression matching words containing at least 2 alphabetical characters
    regex = r"\b[a-z]{2,}\b"
    tokens = re.findall(regex, string)

    # Stem each token (could ignore stopwords)
    for token in tokens:
        stems.append(stemmer.stem(token))

    # return the stemmed tokens
    return stems


def tf_idf_matrix(raw_documents, tokenizing_method):
    '''
    Uses the raw documents with a specific tokenizing method in order to create and return a vectorizer and matrix

    Parameters
    ----------
    raw_documents : list (of strings)
        A list of the text contained in the documents
    tokenizing_method : method
        The tokenizer method to be used in the vectorizer
    '''

    # Use scikit-learn to generate a matrix[document][feature] = tf-idf for feature
    tfidf = TfidfVectorizer(tokenizer=tokenizing_method, stop_words='english')
    tfs = tfidf.fit_transform(raw_documents)

    # return the vectorizer and matrix
    return {'matrix': tfs, 'vectorizer': tfidf}


def feature_selection(tf_idf_matrix, document_classes_for_matrix, max_features, feature_names):
    '''
    Uses a tf idf matrix as well as the given document classes for the matrix in order to extract an array of features to use

    Parameters
    ----------
    tf_idf_matrix : array
        An n x n array of samples and features (training set)
    document_classes_for_matrix : array
        An array of n samples (target values)
    max_features : integer
        The number of top features to select
    feature_names : array
        An array of features to use
    '''

    # Use 𝛘² or Pearson's product moment coefficient to select the k best features
    ch2 = SelectKBest(chi2, k=max_features)
    X_train = ch2.fit_transform(tf_idf_matrix, document_classes_for_matrix)

    feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]

    # return array of features to use
    return feature_names


def load_four_university(yes_label_id, no_label_id, root_dir='webkb'):
    """
    Imports the four university dataset for future insertion to the DB
    Warning: only run once!
    """

    if not os.path.isdir(root_dir):
        print('Could not find directory ' + root_dir +
              '. Extract webkb-data.gtar.gz or provide correct location.')
        return

    articles = []

    for folder, subs, files in os.walk(root_dir):
        for filename in files:
            if not('www' in filename) or ('misc' in folder) or ('other' in folder):
                continue
            file_location = os.path.join(folder, filename)

            with open(file_location, 'r', errors='replace') as src:
                html = src.read()
                html = html.lower()

                if html.find('<html>') > 0:
                    html = html[html.find('<html>'):]
                elif html.find('<') > 0:
                    html = html[html.find('<'):]

                soup = BeautifulSoup(html, 'html.parser')
                text = soup.get_text()
                text = clean_string(text)
                # TODO: find a search query
                # search_query = SearchQuery.objects.get_or_create(search_query='')[0]

                if 'course' in folder:
                    label_id = yes_label_id
                else:
                    label_id = no_label_id

                if soup.title is not None:
                    title = soup.title.string
                else:
                    title = 'No Title ' + str(random.randrange(0, 10000))

                articles.append([{'url': file_location, 'title': title, 'text': text}, {'class_label_id': label_id}])  # , 'query_id': search_query_id}])

    return articles
