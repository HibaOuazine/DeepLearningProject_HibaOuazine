import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import io
import pandas as pd

# Set page config
st.set_page_config(
    page_title="Fruit Classification",
    page_icon="🍎",
    layout="centered"
)

# Custom CSS to improve the app appearance
st.markdown("""
    <style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Define class names and their emojis
CLASS_NAMES = {
    "apple": "🍎",
    "avocado": "🥑",
    "banana": "🍌",
    "cherry": "🍒",
    "kiwi": "🥝",
    "mango": "🥭",
    "orange": "🍊",
    "pineapple": "🍍",
    "strawberries": "🍓",
    "watermelon": "🍉"
}

@st.cache_resource
def load_classification_model():
    """Load and cache the model"""
    return load_model('fruit_classification_model.h5')

def preprocess_image(img):
    """Preprocess the image for model prediction"""
    # Resize image to 224x224 (assuming this is what your model expects)
    img = img.convert("RGB") 
    img = img.resize((224, 224))
    # Convert to array and expand dimensions
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    # Normalize the image
    img_array = img_array / 255.0
    return img_array

def get_fruit_details(fruit_name):
    """Return nutritional information for each fruit"""
    details = {
        "apple": "Rich in fiber and antioxidants. One medium apple contains about 95 calories.",
        "avocado": "High in healthy fats and potassium. Contains about 240 calories per fruit.",
        "banana": "Excellent source of potassium and vitamin B6. Contains about 105 calories per medium banana.",
        "cherry": "Rich in antioxidants and anti-inflammatory compounds. About 50 calories per cup.",
        "kiwi": "High in vitamin C and fiber. One medium kiwi contains about 42 calories.",
        "mango": "Rich in vitamins A and C. One cup of mango contains about 99 calories.",
        "orange": "Excellent source of vitamin C. One medium orange contains about 62 calories.",
        "pineapple": "Contains bromelain and vitamin C. One cup has about 82 calories.",
        "strawberries": "High in vitamin C and antioxidants. One cup contains about 49 calories.",
        "watermelon": "Rich in lycopene and very hydrating. One cup contains about 46 calories."
    }
    return details.get(fruit_name, "Nutritional information not available.")

def main():
    st.title("🍎 Fruit Classification App")
    st.write("Upload an image of a fruit to classify it!")

    # Load the model
    try:
        model = load_classification_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image_bytes = uploaded_file.read()
            img = Image.open(io.BytesIO(image_bytes))
            st.image(img, caption='Uploaded Image', use_column_width=True)

            # Add a prediction button
            if st.button('Classify Fruit'):
                with st.spinner('Analyzing image...'):
                    # Preprocess the image
                    processed_img = preprocess_image(img)

                    # Print the image shape for debugging
                    print("Image shape after preprocessing:", processed_img.shape)
                    
                    # Make prediction
                    prediction = model.predict(processed_img)

                    # Convert logits to probabilities using softmax
                    probabilities = tf.nn.softmax(prediction).numpy()  # Apply softmax and convert to numpy array
                    
                    # Get predicted class and confidence
                    class_names = list(CLASS_NAMES.keys())
                    predicted_class = class_names[np.argmax(prediction)]
                    confidence = float(np.max(prediction)) * 100

                    # Display results with emoji
                    st.markdown(f"""
                    <div class="prediction-box" style="background-color: #f0f2f6;">
                        <h2>{CLASS_NAMES[predicted_class]} Predicted: {predicted_class.title()}</h2>
                        <h3>Confidence: {confidence:.2f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)

                    # Display nutritional information
                    st.subheader("Nutritional Information")
                    st.info(get_fruit_details(predicted_class))

                    # Display probability distribution
                    st.subheader("Probability Distribution")
                    prob_df = pd.DataFrame({
                        'Fruit': [f"{CLASS_NAMES[name]} {name.title()}" for name in class_names],
                        'Probability': prediction[0] * 100
                    })
                    prob_df = prob_df.sort_values('Probability', ascending=True)
                    st.bar_chart(prob_df.set_index('Fruit'))

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.write("Please try uploading a different image.")

    # Add information about supported fruits
    with st.expander("Supported Fruits"):
        st.write("This model can classify the following fruits:")
        cols = st.columns(2)
        for i, (fruit, emoji) in enumerate(CLASS_NAMES.items()):
            col_idx = i % 2
            cols[col_idx].write(f"{emoji} {fruit.title()}")

    # Add instructions
    with st.expander("How to Use"):
        st.write("""
        1. Upload an image of a fruit using the file uploader above
        2. Make sure the image is clear and well-lit
        3. Click the 'Classify Fruit' button
        4. View the prediction results, confidence score, and nutritional information
        
        Tips for best results:
        - Use well-lit photos
        - Center the fruit in the image
        - Avoid multiple fruits in one image
        - Use a clear background when possible
        """)

if __name__ == '__main__':
    main()