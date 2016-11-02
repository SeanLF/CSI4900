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


def bing_news_search(query, max_results=100, offset=0):
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
    urls = []
    while len(urls) < max_results:
        json = bing_news_search(search_query, max_results, len(urls))
        # Don't call more than needed
        if urls == []:
            max_results = min(max_results, json['totalEstimatedMatches'])
        urls += [search_result['url'] for search_result in json['value']]
    return urls


def download_articles(articles):
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
    # Format the string to be case insensitive (lower case)
    # Remove special formatting characters (ex: new line character)
    # return the cleaned string
    return string.lower().replace('\n', ' ')


def tokenize_and_stem(string):
    stems = []
    stemmer = PorterStemmer()
    # from nltk.corpus import stopwords
    # stopwords = stopwords.words('english') + ['the', 'to', 'and', 'of', 'a', 'is', 'that', 'in']

    # Tokenize the string using a regular expression matching words containing at least 2 characters
    regex = r"\b[a-z]{2,}\b"
    tokens = re.findall(regex, string)

    # Stem each token (could ignore stopwords)
    for token in tokens:
        # if token not in self.stopwords:
        stems.append(stemmer.stem(token))

    # return the stemmed tokens
    return stems


def tf_idf_matrix(raw_documents, tokenizing_method):
    # Use scikit-learn to generate a matrix[document][feature] = tf-idf for feature
    tfidf = TfidfVectorizer(tokenizer=tokenizing_method, stop_words='english')
    tfs = tfidf.fit_transform(raw_documents)

    # return the rizer and matrix
    return {'matrix': tfs, 'vectorizer': tfidf}


def feature_selection(tf_idf_matrix, document_classes_for_matrix, max_features, feature_names):
    # Use 𝛘² or Pearson's product moment coefficient to select the k best features
    ch2 = SelectKBest(chi2, k=max_features)
    X_train = ch2.fit_transform(tf_idf_matrix, document_classes_for_matrix)

    feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]

    # return array of features to use
    return feature_names
