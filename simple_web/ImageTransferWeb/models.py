# -*- coding: utf-8 -*-
from django.db import models


# Create your models here.
class DIYTask(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField(max_length=100)
    diy_user = models.CharField(max_length=255)
    diy_style = models.CharField(max_length=255)
    diy_style_detail = models.IntegerField(default=1)
    flag = models.BooleanField(default=False)

    def __unicode__(self):
        return self.email

    class Meta:
        ordering = ['name']
