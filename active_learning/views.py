from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse
from django.db import IntegrityError
from django.views.decorators.clickjacking import xframe_options_exempt
from numpy import unique
from requests import head
import newspaper
import after_response
from os import environ
from .models import Article, Label, SearchQuery, DataSet
from .fetch_data import clean_string, download_articles, get_links, load_four_university
from .learning import Learn
from .utils import format_pusher_channel_name, get_pusher_client


@xframe_options_exempt
def index(request):
    '''
    Renders the list of articles on the index.html page
    '''

    datasets = []
    for ds in DataSet.objects.all():
        articles = Article.objects.filter(dataset_id=ds.id).order_by("class_label_id")
        datasets.append({'dataset_id': ds.id, 'name': ds.name, 'articles': articles, 'count': len(articles)})
    context = {'datasets': datasets}
    return render(request, 'active_learning/index.html', context)


@xframe_options_exempt
def detail(request, article_id):
    '''
    Renders an articles details on the details.html page
    '''

    article = get_object_or_404(Article, pk=article_id)
    return render(request, 'active_learning/detail.html', {'article': article})


def iframe(request):
    '''
    Renders the iframe in the popup iframe.html page
    '''

    presence_channel_name = format_pusher_channel_name(environ['PRESENCE_CHANNEL_NAME'])
    PUSHER_KEY = environ['PUSHER_KEY']
    return render(request, 'active_learning/iframe.html', {'presence_channel_name': presence_channel_name, 'PUSHER_KEY': PUSHER_KEY})


def learn(request):
    '''
    Starts the active learning process which requires another tab to be loaded at the active_learning
    root (iframe) page for the oracle to label articles
    '''

    dataset_id = request.GET['dataset']
    auto_label = request.GET['auto_label']

    do_learn.after_response(dataset_id, auto_label == u'true')
    return HttpResponse(status=200)


@after_response.enable
def do_learn(dataset_id, auto_label):
    '''
    Begins the active learning process (asynchronously)
    '''
    Learn(dataset_id=dataset_id).learn(auto_label=auto_label, self_train=False, active_learning_strategy=1, num_queries=20, train_size=0.01)


def get_articles(request):
    '''
    Fetch articles from Bing
    '''

    search_query = request.GET['search_query']
    max_results = int(request.GET['max_results'])
    label = request.GET['label']
    dataset_name = request.GET['dataset_name']
    # find or create
    dataset_id = DataSet.objects.get_or_create(name=dataset_name)[0].id
    search_query = SearchQuery.objects.get_or_create(search_query=search_query)[0]
    label = Label.objects.get_or_create(label=label)[0]

    get_articles_task.after_response(search_query, max_results, label, dataset_id)

    return HttpResponse(status=200)


@after_response.enable
def get_articles_task(search_query, max_results, label, dataset_id):
    '''
    Get the articles from Bing before saving the article details to the database
    '''

    pusher_client = get_pusher_client()
    presence_channel_name = format_pusher_channel_name(environ['PRESENCE_CHANNEL_NAME'])

    # get links from Bing
    links = get_links(search_query.search_query, max_results)

    # previously fetched URLs
    db_urls = [article.url for article in Article.objects.all()]

    # get unique links (resolve redirect URL)
    temp_links = []
    for link in links:
        temp = head(link)
        # make sure there is no 404 error (page not found)
        if temp.status_code != 404:
            temp_links.append(temp.headers['Location'])
    links = list(set(unique(temp_links)) - set(db_urls))

    pusher_client.trigger(presence_channel_name, 'get_articles', 'got unique links from bing')

    articles = [newspaper.Article(url=links[i]) for i in range(len(links))]
    download_articles(articles)

    for article in articles:
        try:
            article.parse()
        except newspaper.article.ArticleException:
            continue

        text = clean_string(article.text)
        if text == '':
            continue

        a = Article(url=article.url, title=article.title, text=text)
        a.class_label_id = label.id
        a.query_id = search_query.id
        a.dataset_id = dataset_id
        try:
            a.save()  # save to DB
            pusher_client.trigger(presence_channel_name, 'get_articles', 'got article from link')
        except IntegrityError as e:
            pusher_client.trigger(presence_channel_name, 'get_articles', {'errorMessage': str(e), 'article': {'url': a.url, 'title': a.title}})
            continue

    pusher_client.trigger(presence_channel_name, 'get_articles', 'finished getting articles')


def load_four_university_dataset(request):
    load_four_university_task.after_response()
    return HttpResponse(status=200)


@after_response.enable
def load_four_university_task():
    pusher_client = get_pusher_client()
    presence_channel_name = format_pusher_channel_name(environ['PRESENCE_CHANNEL_NAME'])

    dataset_id = DataSet.objects.get_or_create(name='Four University')[0].id
    yes_label_id = Label.objects.get_or_create(label='yes')[0].id
    no_label_id = Label.objects.get_or_create(label='no')[0].id
    search_query_id = SearchQuery.objects.get_or_create(search_query='')[0].id

    pusher_client.trigger(presence_channel_name, 'four_university', 'loading dataset')
    articles = load_four_university(yes_label_id, no_label_id)
    pusher_client.trigger(presence_channel_name, 'four_university', 'finished reading dataset')

    for article in articles:
        db_article = Article(**article[0])
        db_article.query_id = search_query_id  # article[1]['query_id']
        db_article.class_label_id = article[1]['class_label_id']
        db_article.dataset_id = dataset_id
        try:
            db_article.save()
        except IntegrityError as e:
            pusher_client.trigger(presence_channel_name, 'four_university', {'errorMessage': str(e), 'article': {'url': db_article.url, 'title': db_article.title}})
            continue

    pusher_client.trigger(presence_channel_name, 'four_university', 'imported dataset into DB')


@csrf_exempt
def auth(request):
    '''
    Authenticates the current client as a pusher client
    '''

    pusher_client = get_pusher_client()

    auth = pusher_client.authenticate(
        channel=request.POST['channel_name'],
        socket_id=request.POST['socket_id'],
        custom_data={'user_id': 'user'}
    )
    return JsonResponse(auth)
