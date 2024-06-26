from django.urls import path, include

from . import views

urlpatterns = [
    path('', views.index, name='index'),

    path('login/', views.login, name="login"),
    path("register/", views.register, name="register"),
    path("signout/", views.signout, name="signout"),

    path("video/", views.video, name="video"),
    path("cameras/", views.cameras, name="cameras"),
    path("camera/<int:id>", views.camera, name="camera"),
    path('live_stream/<int:id>', views.live_stream, name="live_stream"),
]