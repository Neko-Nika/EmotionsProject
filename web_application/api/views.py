from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.contrib.auth import authenticate, login
from django.contrib.auth.decorators import login_required
import os
import json
from pathlib import Path
from datetime import datetime
from main.models import *
from emotions.creds import PYTHON_PATH


@csrf_exempt
def login_account(request):
    if request.method == "POST" and request.content_type == 'application/json':
        data = json.loads(request.body)
        
        password = data['password']
        email = data['email']

        try:
            user = authenticate(email=email, username=email, password=password)
            if user is not None:
                login(request, user)
            else:
                raise Exception("Неверный Email или пароль")
        except Exception as exc:
            response_data = {
                'success': False,
                'message': str(exc)
            }

            return JsonResponse(response_data) 

        response_data = {
            'success': True,
            'message': 'Пользователь успешно авторизовался'
        }

        return JsonResponse(response_data)
    else:

        response_data = {
            'success': False,
            'message': 'Only POST requests are allowed'
        }
        
        return JsonResponse(response_data, status=405)


@csrf_exempt
def create_account(request):
    if request.method == "POST" and request.content_type == 'application/json':
        data = json.loads(request.body)
        
        password = data['password']
        repeat_password = data['password2']
        email = data['email']

        try:
            if password != repeat_password:
                raise Exception("Пароли не совпадают")
            if User.objects.filter(email=email).first() is not None:
                raise Exception("Пользователь с таким Email уже зарегестрирован")
        except Exception as exc:
            response_data = {
                'success': False,
                'message': str(exc)
            }

            return JsonResponse(response_data) 

        new_user = User.objects.create_user(email=email, username=email, password=password)
        new_user.save()

        login(request, new_user)

        response_data = {
            'success': True,
            'message': 'Аккаунт успешно создан'
        }

        return JsonResponse(response_data)
    else:

        response_data = {
            'success': False,
            'message': 'Only POST requests are allowed'
        }
        
        return JsonResponse(response_data, status=405)


@csrf_exempt
def upload_video(request):
    if request.method == "POST":
        try:
            video_file = request.FILES["video"]
            uploaded = UploadedVideo.objects.create(video=video_file)
            uploaded.save()
            
            path = uploaded.video.path
            os.system(f"{PYTHON_PATH} api/detection.py {path}")
            uploaded.delete()

            core_path = core_path = Path(path)

            response_data = {
                'success': True,
                'output_filename': f"{core_path.stem}_output{core_path.suffix}",
                'message': 'Обработка успешно завершена'
            }
            return JsonResponse(response_data)
        
        except Exception as exc:
            response_data = {
                'success': False,
                'message': exc
            }

            return JsonResponse(response_data)
    else:

        response_data = {
            'success': False,
            'message': 'Only POST requests are allowed'
        }
        
        return JsonResponse(response_data, status=405)
