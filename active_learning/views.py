from django.shortcuts import render, get_object_or_404, redirect
from .models import Article
from .fetch_data import FetchData
import newspaper.settings


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


def term_frequencies(request):
    articles = Article.objects.all()
    fd = FetchData()

    for article in articles:
        fd.findWordFrequencies(article)
        article.save()

    return render(request, 'active_learning/term_frequencies.html', {})
