# -*- coding: utf-8 -*-
# Generated by Django 1.11.2 on 2018-03-22 02:34
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='DIYTask',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('email', models.EmailField(max_length=100)),
                ('diy_user', models.CharField(max_length=255)),
                ('diy_style', models.CharField(max_length=255)),
            ],
            options={
                'ordering': ['name'],
            },
        ),
    ]
