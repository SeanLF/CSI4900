from django.shortcuts import render, get_object_or_404
from .models import Article
from .fetch_data import FetchData


def index(request):
    if Article.objects.count() == 0:
        fd = FetchData()
        links_per_query = fd.getGoogleNewsURLs()
        fd.processAndSaveDataFromLinks(links_per_query)

    articles = Article.objects.all().order_by("query")

    context = {'articles': articles}
    return render(request, 'active_learning/index.html', context)


def detail(request, article_id):
    article = get_object_or_404(Article, pk=article_id)
    return render(request, 'active_learning/detail.html', {'article': article})
