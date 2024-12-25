import tensorflow as tf
import numpy as np
from PIL import Image
import os
from pathlib import Path

class FruitPredictor:
    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, 'fruit_classification_model.h5')
        print(f"Model path: {self.model_path}")
        print(f"Model file exists: {os.path.exists(self.model_path)}")
        self.class_names = ["apple","avocado","banana","cherry","kiwi","mango", "orange","pineapple", "strawberries","watermelon"]
        self.model = self.load_model()

    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at: {self.model_path}")
            print("Loading model...")
            model = tf.keras.models.load_model(self.model_path)
            print("Model loaded successfully!")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def preprocess_image(self, image, target_size=(32, 32)):
        try:
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = image.resize(target_size)
            image_array = np.array(image, dtype=np.float32)
            image_array = image_array / 255.0
            image_array = image_array.reshape((1, 32, 32, 3))
            return image_array
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            raise RuntimeError(f"Failed to preprocess image: {str(e)}")

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
            print(f"Error during prediction: {str(e)}")
            raise RuntimeError(f"Failed to make prediction: {str(e)}")
