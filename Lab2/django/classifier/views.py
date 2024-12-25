from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .predictor import FruitPredictor
from PIL import Image
import json

def home(request):
    return render(request, 'classifier/home.html')

@csrf_exempt
def predict(request):
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            predictor = FruitPredictor()  # Create instance when needed
            image = Image.open(request.FILES['image'])
            result = predictor.predict(image)
            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=400)
    return JsonResponse({'error': 'No image provided'}, status=400)
