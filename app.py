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
        ["Grayscale Conversion", "Object Detection", "Gender Detection"]
    )
    
    # Add model selection and confidence threshold for object detection
    if processing_option == "Object Detection":
        col1, col2 = st.columns(2)
        with col1:
            model_option = st.selectbox(
                "Select YOLO Model",
                ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                index=0
            )
        with col2:
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=0.9,
                value=0.25,
                step=0.05
            )
    
    if uploaded_file is not None:
        try:
            # Read the image
            image = Image.open(uploaded_file)
            
            # Convert image to RGB if it has an alpha channel
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            # Convert PIL Image to OpenCV format (numpy array)
            img_array = np.array(image)
            
            # Display original image
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # Process the image based on selected option
            if processing_option == "Grayscale Conversion":
                processed_img = process_grayscale(img_array)
                st.subheader("Grayscale Image")
                st.image(processed_img, caption="Grayscale Image", use_container_width=True)
                
                # Download button for processed image
                if st.button("Download Grayscale Image"):
                    download_image(processed_img, "grayscale_image.jpg")
                
            elif processing_option == "Object Detection":
                result_img, detection_results = detect_objects(img_array, model_option, confidence_threshold)
                st.subheader("Object Detection Result")
                st.image(result_img, caption="Detected Objects", use_container_width=True)
                
                # Display detection results
                if detection_results:
                    st.subheader("Detected Objects")
                    for i, (cls, conf) in enumerate(detection_results):
                        st.write(f"{i+1}. {cls} (Confidence: {conf:.2f})")
                else:
                    st.info("No objects detected with the current confidence threshold.")
                
                # Download button for processed image
                if st.button("Download Detection Result"):
                    download_image(result_img, "detection_result.jpg")
            
            # In the main function, update the gender detection results display
            else:  # Gender Detection
                result_img, gender_results = detect_gender(img_array)
                st.subheader("Gender and Emotion Detection Result")
                st.image(result_img, caption="Detected Persons", use_container_width=True)
                
                # Display gender and emotion detection results
                if gender_results:
                    st.subheader("Detected Persons")
                    for i, (gender, emotion, conf, bbox) in enumerate(gender_results):
                        st.write(f"{i+1}. {gender} - Emotion: {emotion} (Confidence: {conf:.2f})")
                else:
                    st.info("No persons detected in the image.")
                
                # Download button for processed image
                if st.button("Download Gender Detection Result"):
                    download_image(result_img, "gender_detection_result.jpg")
                
        except Exception as e:
            st.error(f"Error processing image: {e}")
            # Add more detailed error information for debugging
            import traceback
            st.error(traceback.format_exc())

def process_grayscale(image):
    """Convert image to grayscale"""
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image

def detect_objects(image, model_name="yolov8n.pt", conf_threshold=0.25):
    """Detect objects in the image using YOLO"""
    # Load YOLO model
    model = YOLO(model_name)
    
    # Run inference with lower confidence threshold to detect more objects
    results = model(image, conf=conf_threshold)
    
    # Get the result with annotations
    annotated_img = results[0].plot()
    
    # Extract detection results for display
    detection_results = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = result.names[int(box.cls)]
            conf = float(box.conf)
            detection_results.append((cls, conf))
    
    return annotated_img, detection_results

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

# Fix the typo in the function definition (remove "ein")
def detect_gender(image):
    """Detect persons and their gender in the image"""
    # Load face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load YOLO model for person detection
    model = YOLO("yolov8n.pt")
    
    # Make a copy of the image for drawing
    img_copy = image.copy()
    
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    # Use YOLO to detect persons
    results = model(image, classes=[0])  # Class 0 is person in COCO dataset
    
    # Extract gender detection results
    gender_results = []
    
    # Process YOLO results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            
            # Determine gender based on face detection within person bounding box
            gender = "Unknown"
            emotion = "Neutral"
            max_face_area = 0
            
            for (fx, fy, fw, fh) in faces:
                # Check if face is within this person bounding box
                if (fx > x1 and fy > y1 and fx + fw < x2 and fy + fh < y2):
                    face_area = fw * fh
                    if face_area > max_face_area:
                        max_face_area = face_area
                        
                        # Simple gender classification based on face width-to-height ratio
                        ratio = fw / fh
                        if ratio > 0.78:  # Arbitrary threshold
                            gender = "Male"
                        else:
                            gender = "Female"
                        
                        # Extract the face ROI for emotion analysis
                        face_roi = gray[fy:fy+fh, fx:fx+fw]
                        
                        # Simple emotion detection based on pixel intensity variance
                        # This is a very simplified approach - in a real app you'd use a trained model
                        if face_roi.size > 0:
                            variance = np.var(face_roi)
                            if variance > 2000:
                                emotion = "Happy/Excited"
                            elif variance > 1000:
                                emotion = "Confident"
                            elif variance < 500:
                                emotion = "Sad/Tired"
                            else:
                                emotion = "Neutral"
            
            # Draw bounding box with gender and emotion label
            color = (0, 255, 0) if gender == "Male" else (255, 0, 0)
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_copy, f"{gender} ({emotion}) {conf:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            gender_results.append((gender, emotion, conf, (x1, y1, x2, y2)))
    
    return img_copy, gender_results

if __name__ == "__main__":
    main()