from django.conf.urls import url

from . import views

app_name = 'active_learning'
urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^(?P<article_id>[0-9]+)/$', views.detail, name='detail'),
]
