# -*- coding: utf-8 -*-
# Generated by Django 1.11.2 on 2018-03-22 02:36
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ImageTransferWeb', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='diytask',
            name='flag',
            field=models.BooleanField(default=False),
        ),
    ]