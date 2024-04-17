from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import logout


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
