# -*- coding: utf-8 -*-
# Generated by Django 1.10.2 on 2016-10-13 19:54
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('active_learning', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='article',
            name='security_breach',
            field=models.CharField(max_length=5, null=True),
        ),
    ]
