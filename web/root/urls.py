from django.contrib import admin
from django.urls import path, include

import generatorViewer.views


urlpatterns = [
    path("admin/", admin.site.urls),
    path("", generatorViewer.views.index),
    path("next_batch", generatorViewer.views.next_batch),
    path("__reload__/", include("django_browser_reload.urls"))
]
