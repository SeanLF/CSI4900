from django.db import models


class Article(models.Model):
    query = models.CharField(max_length=20)
    url = models.TextField()
    date_accessed = models.DateField(auto_now_add=True)
    title = models.TextField()
    text = models.TextField()
    nlp_keywords = models.TextField()

    # mining attributes
    # cyber security, hack, ip, breach
    cyber_security_occurences = models.PositiveSmallIntegerField(default=0)
    hack_occurences = models.PositiveSmallIntegerField(default=0)
    ip_occurences = models.PositiveSmallIntegerField(default=0)
    breach_occurences = models.PositiveSmallIntegerField(default=0)

    # class attribute
    security_breach = models.PositiveSmallIntegerField(null=True)
