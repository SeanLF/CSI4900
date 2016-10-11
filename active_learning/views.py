from django.shortcuts import render, get_object_or_404
from .models import Article


def index(request):
    articles = Article.objects.all()
    context = {'articles': articles}
    return render(request, 'active_learning/index.html', context)


def detail(request, article_id):
    article = get_object_or_404(Article, pk=article_id)
    return render(request, 'active_learning/detail.html', {'article': article})
