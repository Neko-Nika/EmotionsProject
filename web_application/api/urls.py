from django.urls import path

from . import views

urlpatterns = [
    path("create_account", views.create_account, name="create_account"),
    path('login_account', views.login_account, name="login_account"),
]