from flask import Flask, render_template, request, jsonify
import os
from predictor import FruitPredictor
from PIL import Image
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
predictor = FruitPredictor()

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request")
    print("Files in request:", request.files)
    
    if 'image' not in request.files:
        print("Error: 'image' not found in request.files")
        return jsonify({'error': 'No image provided in the request'}), 400
    
    file = request.files['image']
    print(f"Received file with filename: {file.filename}")
    
    if file.filename == '':
        print("Error: Empty filename")
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read image directly from request
        image_bytes = file.read()
        print(f"Read {len(image_bytes)} bytes from the image file")
        
        image = Image.open(io.BytesIO(image_bytes))
        print(f"Successfully opened image: size={image.size}, mode={image.mode}")
        
        # Get prediction
        result = predictor.predict(image)
        print(f"Prediction result: {result}")
        
        if 'error' in result:
            print(f"Prediction error: {result['error']}")
            return jsonify(result), 400
        return jsonify(result)
    
    except Exception as e:
        error_msg = f"Server error during prediction: {str(e)}"
        print(error_msg)
        return jsonify({'error': error_msg}), 500

if __name__ == '__main__':
    app.run(debug=True)
