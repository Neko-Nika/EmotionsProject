from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class UploadedVideo(models.Model):
    video = models.FileField(upload_to="uploaded_videos/", blank=False, null=False)


class Camera(models.Model):
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=200, unique=False, blank=False, null=False)
    link = models.CharField(max_length=200, unique=False, blank=False, null=False)
    pid = models.IntegerField(default=None, null=True)
