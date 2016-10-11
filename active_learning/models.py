from django.db import models


class Article(models.Model):
    query = models.CharField(max_length=20)
    url = models.TextField()
    title = models.TextField()
    text = models.TextField()
    keywords = models.TextField()
    security_breach = models.NullBooleanField()
