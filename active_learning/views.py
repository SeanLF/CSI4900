from django.shortcuts import render, get_object_or_404, redirect
from .models import Article
from .fetch_data import FetchData
import re
import newspaper.settings
from collections import defaultdict


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
    numberOfDocuments = len(articles)

    with open(newspaper.settings.NLP_STOPWORDS_EN, 'r') as f:
        stopwords = set([w.strip() for w in f.readlines()])

    for article in articles:
        # take care of case sensitivity
        text = article.text.lower()

        article.cyber_security_occurences = len(re.findall(r'\bcyber\s?security\b', text))
        article.hack_occurences = len(re.findall(r'\bhack\b', text))
        article.ip_occurences = len(re.findall(r'\bip\b', text))
        article.breach_occurences = len(re.findall(r'\bbreach\b', text))
        article.save()

    return render(request, 'active_learning/term_frequencies.html', {})
