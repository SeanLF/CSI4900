from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse
from django.db import IntegrityError
from django.views.decorators.clickjacking import xframe_options_exempt
from .models import Article, Label, SearchQuery
from .fetch_data import clean_string, download_articles, get_links
from .learning import Learn
from numpy import unique
from requests import head
import newspaper
import pusher
import os
import after_response


def get_pusher_client():
    return pusher.Pusher(
      app_id=os.environ['PUSHER_APP_ID'],
      key=os.environ['PUSHER_KEY'],
      secret=os.environ['PUSHER_SECRET'],
      ssl=True
    )


@xframe_options_exempt
def index(request):
    articles = Article.objects.all().order_by("query")
    context = {'articles': articles, 'count': len(articles)}
    return render(request, 'active_learning/index.html', context)


@xframe_options_exempt
def detail(request, article_id):
    article = get_object_or_404(Article, pk=article_id)
    return render(request, 'active_learning/detail.html', {'article': article})


def iframe(request):
    return render(request, 'active_learning/iframe.html', {})


def learn(request):
    learn = Learn()
    learn.learn()

    return HttpResponse(status=200)


def get_articles(request):
    search_query = request.GET['search_query']
    max_results = int(request.GET['max_results'])
    label = request.GET['label']
    # find or create
    search_query = SearchQuery.objects.get_or_create(search_query=search_query)[0]
    label = Label.objects.get_or_create(label=label)[0]

    get_articles_task.after_response(search_query, max_results, label)

    return HttpResponse(status=200)


@after_response.enable
def get_articles_task(search_query, max_results, label):
    pusher_client = get_pusher_client()

    # get unique links
    links = get_links(search_query.search_query, max_results)
    links = unique([head(link).headers['Location'] for link in links])
    pusher_client.trigger('presence-channel', 'get_articles', 'got links from bing')

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
        try:
            a.save()  # save to DB
            pusher_client.trigger('presence-channel', 'get_articles', 'got article from link')
        except IntegrityError as e:
            pusher_client.trigger('presence-channel', 'get_articles', str(e))
            continue

    pusher_client.trigger('presence-channel', 'get_articles', 'finished getting articles')


@csrf_exempt
def auth(request):
    pusher_client = get_pusher_client()

    auth = pusher_client.authenticate(
        channel=request.POST['channel_name'],
        socket_id=request.POST['socket_id'],
        custom_data={'user_id': 'user'}
    )
    return JsonResponse(auth)
