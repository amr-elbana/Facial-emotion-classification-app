import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the pre-trained model
model = load_model('model.keras')

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Custom CSS for styling
st.markdown(
    """
    <style>
    .stApp {
        background-color:rgb(0, 0, 0);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stFileUploader>div>div>div>div {
        color: #4CAF50;
    }
    .stMarkdown h1 {
        color: #4CAF50;
    }
    .stProgress>div>div>div {
        background-color: #4CAF50;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app layout
st.title("🎭 Emotion Classification App")
st.markdown("Upload an image of a face, and the app will predict the emotion.")
st.markdown("---")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.subheader("Uploaded Image")
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    with st.spinner("Processing image..."):
        img = img.convert("L")  # Convert to grayscale
        img = img.resize((48, 48))  # Resize to 48x48
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    with st.spinner("Predicting emotion..."):
        prediction = model.predict(img_array)
        predicted_emotion = emotion_labels[np.argmax(prediction)]

    # Display prediction with a progress bar
    st.subheader("Prediction Result")
    st.write(f"**Predicted Emotion:** {predicted_emotion}")
    
    st.markdown("---")
    st.success("✅ Prediction complete!")