"""GraduationPro URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/1.11/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  url(r'^$', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  url(r'^$', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.conf.urls import url, include
    2. Add a URL to urlpatterns:  url(r'^blog/', include('blog.urls'))
"""
# -*- coding: utf-8 -*-
from django.conf.urls import url
from django.contrib import admin
from ImageTransferWeb import views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^ImageTransfer/home/$', views.home),
    url(r'^ImageTransfer/upload_online/$', views.upload_online),
    url(r'^ImageTransfer/upload_share/$', views.upload_share),
    url(r'^ImageTransfer/upload_diy/$', views.upload_diy),
    url(r'^ImageTransfer/save_diy_info/$', views.save_diy_info),
    url(r'^ImageTransfer/transfer_style/$', views.transfer_style),
]
