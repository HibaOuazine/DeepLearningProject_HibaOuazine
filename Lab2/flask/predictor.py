import tensorflow as tf
import numpy as np
from PIL import Image

class FruitPredictor:
    def __init__(self):
        self.model_path = 'fruit_classification_model.h5'
        self.class_names = ["apple","avocado","banana","cherry","kiwi","mango", "orange","pineapple", "strawberries","watermelon"]
        self.model = self.load_model()

    def load_model(self):
        try:
            print("\n=== Loading Model ===")
            print("Loading model from:", self.model_path)
            model = tf.keras.models.load_model(self.model_path)
            print("\n=== Model Architecture ===")
            model.summary()
            print("\n=== Model Configuration ===")
            print("Model input shape:", model.input_shape)
            print("Model output shape:", model.output_shape)
            print("Number of layers:", len(model.layers))
            print("First layer type:", type(model.layers[0]).__name__)
            print("Last layer type:", type(model.layers[-1]).__name__)
            print("=== End Model Info ===\n")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise e

    def preprocess_image(self, image, target_size=(32, 32)):  
        try:
            print(f"Original image size: {image.size}, mode: {image.mode}")
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize the image
            image = image.resize(target_size)
            print(f"Resized image size: {image.size}")
            
            # Convert to numpy array and normalize
            image_array = np.array(image, dtype=np.float32)
            print(f"Array shape before processing: {image_array.shape}")
            
            # The model has a rescaling layer, so we don't need to normalize here
            # Just ensure the values are float32
            image_array = image_array.astype(np.float32)
            
            # Add batch dimension if not present
            if len(image_array.shape) == 3:
                image_array = np.expand_dims(image_array, axis=0)
            
            print(f"Final preprocessed shape: {image_array.shape}")
            print(f"Value range: min={np.min(image_array)}, max={np.max(image_array)}")
            return image_array
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            raise e

    def predict(self, image):
        try:
            if not isinstance(image, Image.Image):
                return {'error': f'Invalid image type. Expected PIL.Image.Image, got {type(image)}'}

            print(f"Input image details - Size: {image.size}, Mode: {image.mode}")
            
            print("Starting prediction process...")
            print("Preprocessing image...")
            preprocessed_image = self.preprocess_image(image)
            print(f"Preprocessed image shape: {preprocessed_image.shape}")
            
            if preprocessed_image.shape != (1, 32, 32, 3):
                return {'error': f'Invalid image shape after preprocessing. Expected (1, 32, 32, 3), got {preprocessed_image.shape}'}
            
            print("Running model prediction...")
            predictions = self.model.predict(preprocessed_image)
            print(f"Raw predictions shape: {predictions.shape}")
            
            # Get the predicted class index and probability
            predicted_class_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class_index])
            predicted_class = self.class_names[predicted_class_index]
            
            # Format probabilities for all classes
            all_probabilities = []
            for i, prob in enumerate(predictions[0]):
                all_probabilities.append({
                    'class': self.class_names[i],
                    'probability': float(prob)
                })
            
            print(f"Prediction complete - Class: {predicted_class}, Confidence: {confidence:.2f}")
            
            return {
                'class_name': predicted_class,
                'confidence': confidence,
                'all_probabilities': all_probabilities
            }
            
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            print(error_msg)
            return {'error': error_msg}
