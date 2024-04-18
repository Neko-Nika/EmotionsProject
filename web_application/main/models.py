from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class UploadedVideo(models.Model):
    video = models.FileField(upload_to="uploaded_videos/", blank=False, null=False)
