import streamlit as st
try:
    import cv2
except ImportError:
    st.error("OpenCV (cv2) is not installed. Please run: pip install opencv-python-headless")
    st.stop()
import numpy as np
from PIL import Image
import io
import torch
from ultralytics import YOLO

def main():
    # Set page title and description
    st.title("Image Processing and Computer Vision App")
    st.write("Upload an image to convert it to grayscale or detect objects")
    
    # File uploader for image input
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    # Processing options
    processing_option = st.radio(
        "Select Processing Option",
        ["Grayscale Conversion", "Object Detection"]
    )
    
    if uploaded_file is not None:
        try:
            # Read the image
            image = Image.open(uploaded_file)
            
            # Convert PIL Image to OpenCV format (numpy array)
            img_array = np.array(image)
            
            # Display original image
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Process the image based on selected option
            if processing_option == "Grayscale Conversion":
                processed_img = process_grayscale(img_array)
                st.subheader("Grayscale Image")
                st.image(processed_img, caption="Grayscale Image", use_column_width=True)
                
                # Download button for processed image
                if st.button("Download Grayscale Image"):
                    download_image(processed_img, "grayscale_image.jpg")
                
            else:  # Object Detection
                result_img = detect_objects(img_array)
                st.subheader("Object Detection Result")
                st.image(result_img, caption="Detected Objects", use_column_width=True)
                
                # Download button for processed image
                if st.button("Download Detection Result"):
                    download_image(result_img, "detection_result.jpg")
                
        except Exception as e:
            st.error(f"Error processing image: {e}")

def process_grayscale(image):
    """Convert image to grayscale"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image

def detect_objects(image):
    """Detect objects in the image using YOLO"""
    # Load YOLO model
    model = YOLO("yolov8n.pt")
    
    # Run inference
    results = model(image)
    
    # Get the result with annotations
    annotated_img = results[0].plot()
    
    return annotated_img

def download_image(image, filename):
    """Allow users to download the processed image"""
    # Convert grayscale to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(image)
    
    # Save to BytesIO object
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    buf.seek(0)
    
    # Provide download button
    st.download_button(
        label="Download Image",
        data=buf,
        file_name=filename,
        mime="image/jpeg"
    )

if __name__ == "__main__":
    main()