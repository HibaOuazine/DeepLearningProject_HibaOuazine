from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI()

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monter les fichiers statiques
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration des templates
templates = Jinja2Templates(directory="templates")

class FruitPredictor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, 'fruit_classification_model.h5')
        self.class_names = ["apple", "avocado", "banana", "cherry", "kiwi", "mango", "orange", "pineapple", "strawberries", "watermelon"]
        self.model = None  # Initialize later
        self.ensure_model_exists()

    def ensure_model_exists(self):
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # If model doesn't exist in fastapi_app/model, try to copy from django/classifier/model
        if not os.path.exists(self.model_path):
            django_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 
                                           'django', 'classifier', 'model', 'fruit_classification_model.h5')
            if os.path.exists(django_model_path):
                import shutil
                shutil.copy2(django_model_path, self.model_path)
        
        # Now try to load the model
        try:
            self.model = tf.keras.models.load_model(self.model_path)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.model = None

    def preprocess_image(self, image, target_size=(32, 32)):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize(target_size)
        image_array = np.array(image, dtype=np.float32)
        image_array = image_array.reshape((1, 32, 32, 3))
        return image_array

    def predict(self, image):
        try:
            preprocessed_image = self.preprocess_image(image)
            predictions = self.model.predict(preprocessed_image)
            predictions = tf.nn.softmax(predictions[0]).numpy()
            
            class_index = np.argmax(predictions)
            confidence = float(predictions[class_index])
            
            return {
                'class_name': self.class_names[class_index],
                'confidence': confidence,
                'all_probabilities': [
                    {'class': name, 'probability': float(prob)}
                    for name, prob in zip(self.class_names, predictions)
                ]
            }
        except Exception as e:
            return {'error': str(e)}

predictor = FruitPredictor()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print(f"Received prediction request for file: {file.filename}")
    try:
        # Lire l'image
        contents = await file.read()
        print("File read successfully")
        image = Image.open(io.BytesIO(contents))
        print("Image opened successfully")
        
        # Faire la prédiction
        result = predictor.predict(image)
        print(f"Prediction result: {result}")
        return result
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="debug")