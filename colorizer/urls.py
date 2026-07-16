from django.urls import path

from . import views

app_name = "colorizer"

urlpatterns = [
    path("", views.index, name="index"),
    path("colorize/", views.colorize_image, name="colorize"),
]
