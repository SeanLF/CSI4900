from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse
from django.db import IntegrityError
from django.views.decorators.clickjacking import xframe_options_exempt
from .models import Article, Label, SearchQuery
from .fetch_data import *
from .learning import Learn
import newspaper.settings
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
    context = {'articles': articles}
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

    # urls is an array of strings
    links = get_links(search_query, max_results)
    articles = [get_article_from_url(links[i]) for i in range(len(links))]

    pusher_client.trigger('presence-channel', 'get_articles', 'got links from bing and fetched articles')

    for article in articles:
        text = clean_string(article.text)
        a = Article(url=article.url, title=article.title, text=text)
        a.label_id = label.id
        a.query_id = search_query.id
        try:
            a.save()  # save to DB
        except IntegrityError:
            pusher_client.trigger('presence-channel', 'get_articles', 'detected article with duplicate title')
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
