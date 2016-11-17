from django.db import models


class Article(models.Model):
    query = models.ForeignKey('SearchQuery', on_delete=models.CASCADE)
    dataset = models.ForeignKey('DataSet', on_delete=models.CASCADE)
    url = models.TextField()
    date_accessed = models.DateField(auto_now_add=True)
    title = models.TextField(unique=True)
    text = models.TextField()
    # class attribute
    class_label = models.ForeignKey('Label', on_delete=models.CASCADE, null=True)


class Label(models.Model):
    label = models.CharField(max_length=20)


class SearchQuery(models.Model):
    search_query = models.CharField(max_length=100)


class DataSet(models.Model):
    name = models.CharField(max_length=50)
