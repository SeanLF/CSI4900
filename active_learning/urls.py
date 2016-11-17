from django.conf.urls import url

from . import views

app_name = 'active_learning'
urlpatterns = [
    url(r'^$', views.iframe, name='iframe'),
    url(r'^article$', views.index, name='index'),
    url(r'^article/(?P<article_id>[0-9]+)/$', views.detail, name='detail'),
    url(r'^learn$', views.learn),
    url(r'^pusher/auth', views.auth),
    url(r'^get_articles', views.get_articles),
    url(r'^load_four_university', views.load_four_university_dataset),
]
