from django.urls import path

from . import views

urlpatterns = [
    path("create_account", views.create_account, name="create_account"),
    path('login_account', views.login_account, name="login_account"),

    path('upload_video', views.upload_video, name="upload_video"),
    path('check_connection_available', views.check_connection_available, name="check_connection_available"),
    path('add_camera', views.add_camera, name="add_camera"),
    path('delete_camera', views.delete_camera, name="delete_camera"),
]