from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required


@login_required(login_url="/login")
def index(request):
    return render(request, 'main.html')


def login(request):
    return render(request, 'login.html')
                  

def register(request):
    return render(request, "register.html")
