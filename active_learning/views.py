from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from .models import Article
from .fetch_data import *
from .learning import Learn
import newspaper.settings
import pusher
import os
from django.views.decorators.clickjacking import xframe_options_exempt


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

    return HttpResponse(status_code=200)


@csrf_exempt
def auth(request):
    pusher_client = pusher.Pusher(
      app_id=os.environ['PUSHER_APP_ID'],
      key=os.environ['PUSHER_KEY'],
      secret=os.environ['PUSHER_SECRET'],
      ssl=True
    )

    auth = pusher_client.authenticate(
        channel=request.POST['channel_name'],
        socket_id=request.POST['socket_id'],
        custom_data={'user_id': 'user'}
    )
    return JsonResponse(auth)
