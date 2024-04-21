from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.views.decorators import gzip
from django.contrib.auth import logout
from django.http import StreamingHttpResponse

from main.models import *
import main.stream as ms


@login_required(login_url="/login")
def index(request):
    return render(request, 'instructions.html')


def login(request):
    return render(request, 'login.html')
                  

def register(request):
    return render(request, "register.html")


@login_required(login_url="/login")
def signout(request):
    logout(request)
    return redirect("login")


@login_required(login_url="/login")
def video(request):
    return render(request, "video.html")


@login_required(login_url="/login")
def cameras(request):
    cameras = request.user.camera_set.all()
    if cameras:
        cameras = cameras.order_by("id")[::-1]
    else:
        cameras = []
    context = {
        "cameras": cameras
    }

    return render(request, "cameras.html", context)


@login_required(login_url="/login")
def camera(request, id):
    context = {
        "camera": request.user.camera_set.get(id=id),
    }

    return render(request, "camera.html", context)


@login_required
@gzip.gzip_page
def live_stream(request, id):
    camera = request.user.camera_set.get(id=id)
    stream = ms.VideoCamera(camera)
    return StreamingHttpResponse(ms.gen(stream), content_type="multipart/x-mixed-replace;boundary=frame")
