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
            
            # Gender Detection option (this was incorrectly indented)
            elif processing_option == "Gender Detection":
                result_img, gender_results = detect_gender(img_array)
                st.subheader("Person Analysis Result")
                st.image(result_img, caption="Detected Persons", use_container_width=True)
                
                # Display comprehensive detection results
                if gender_results:
                    st.subheader("Detected Persons")
                    
                    # Create a table for better visualization
                    data = []
                    for i, (gender, emotion, age_group, activity, conf, _) in enumerate(gender_results):
                        data.append({
                            "Person": i+1,
                            "Gender": gender,
                            "Age Group": age_group,
                            "Emotion": emotion,
                            "Activity": activity,
                            "Confidence": f"{conf:.2f}"
                        })
                    
                    st.table(data)
                    
                    # Show detailed analysis
                    st.subheader("Detailed Analysis")
                    for i, (gender, emotion, age_group, activity, conf, _) in enumerate(gender_results):
                        with st.expander(f"Person {i+1} Details"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Gender:** {gender}")
                                st.write(f"**Age Group:** {age_group}")
                            with col2:
                                st.write(f"**Emotion:** {emotion}")
                                st.write(f"**Confidence:** {conf:.2f}")
                            
                            st.write(f"**Detected Activity:** {activity}")
                            
                            # Add interpretation
                            st.write("**Interpretation:**")
                            if emotion == "Happy/Excited":
                                st.write("This person appears to be in a positive emotional state.")
                            elif emotion == "Sad/Tired":
                                st.write("This person may be experiencing fatigue or negative emotions.")
                            elif emotion == "Confident":
                                st.write("This person appears self-assured and composed.")
                            elif emotion == "Surprised":
                                st.write("This person appears to be reacting to something unexpected.")
                            
                            # Add business context interpretation
                            if "Business Meeting" in activity:
                                st.write("**Business Context:** This person appears to be engaged in a formal business discussion.")
                            elif "Handshake" in activity:
                                st.write("**Business Context:** This person may be concluding a business deal or greeting someone formally.")
                            elif "Presentation" in activity:
                                st.write("**Business Context:** This person appears to be presenting information in a professional setting.")
                            elif "Negotiation" in activity:
                                st.write("**Business Context:** This person seems to be in a negotiation process.")
                else:
                    st.info("No persons detected in the image.")
                
                # Download button for processed image
                if st.button("Download Person Analysis Result"):
                    download_image(result_img, "person_analysis_result.jpg")
                
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

def detect_gender(image):
    """Detect persons and their gender, age, emotion, and business activities in the image"""
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
    
    # Use YOLO to detect persons and objects
    results = model(image)
    
    # Extract gender detection results
    gender_results = []
    
    # Detect business context in the overall image
    business_context = detect_business_context(image, results)
    
    # Process YOLO results
    person_boxes = []
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = result.names[int(box.cls)]
            if cls == "person":
                person_boxes.append(box)
    
    # Analyze person interactions
    person_interactions = analyze_person_interactions(person_boxes)
    
    # Process each detected person
    for i, box in enumerate(person_boxes):
        # Get bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf)
        
        # Determine gender based on face detection within person bounding box
        gender = "Unknown"
        emotion = "Neutral"
        age_group = "Unknown"
        max_face_area = 0
        
        for (fx, fy, fw, fh) in faces:
            # Check if face is within this person bounding box
            if (fx > x1 and fy > y1 and fx + fw < x2 and fy + fh < y2):
                face_area = fw * fh
                if face_area > max_face_area:
                    max_face_area = face_area
                    
                    # Extract the face ROI for analysis
                    face_roi = gray[fy:fy+fh, fx:fx+fw]
                    color_face_roi = image[fy:fy+fh, fx:fx+fw]
                    
                    # Simple gender classification based on face width-to-height ratio
                    ratio = fw / fh
                    if ratio > 0.78:
                        gender = "Male"
                    else:
                        gender = "Female"
                    
                    # Improved emotion detection based on pixel intensity patterns
                    if face_roi.size > 0:
                        # Calculate various metrics for emotion detection
                        variance = np.var(face_roi)
                        mean = np.mean(face_roi)
                        std_dev = np.std(face_roi)
                        
                        # Edge detection for facial features
                        edges = cv2.Canny(face_roi, 100, 200)
                        edge_count = np.count_nonzero(edges)
                        
                        # Determine emotion based on multiple factors
                        if edge_count > (fw * fh * 0.15) and variance > 1800:
                            emotion = "Happy/Excited"
                        elif edge_count < (fw * fh * 0.08) and mean < 100:
                            emotion = "Sad/Tired"
                        elif std_dev > 50 and variance > 1500:
                            emotion = "Surprised"
                        elif variance > 1200 and mean > 120:
                            emotion = "Confident"
                        elif variance < 800:
                            emotion = "Neutral"
                        else:
                            emotion = "Thoughtful"
                    
                    # Age estimation based on face texture and proportions
                    if face_roi.size > 0:
                        # Apply Gabor filter for wrinkle detection
                        ksize = 31
                        sigma = 3.0
                        theta = 0
                        lambd = 10.0
                        gamma = 0.5
                        
                        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
                        filtered = cv2.filter2D(face_roi, cv2.CV_8UC3, kernel)
                        
                        # Calculate texture metrics
                        texture_variance = np.var(filtered)
                        
                        # Estimate age based on texture and face proportions
                        if texture_variance < 500 and ratio > 0.85:
                            age_group = "Child (0-12)"
                        elif texture_variance < 1000 and ratio > 0.8:
                            age_group = "Teen (13-19)"
                        elif texture_variance < 1500:
                            age_group = "Young Adult (20-35)"
                        elif texture_variance < 2500:
                            age_group = "Adult (36-50)"
                        else:
                            age_group = "Senior (51+)"
        
        # Determine business activity based on context and person interactions
        activity = determine_business_activity(i, person_interactions, business_context, emotion)
        
        # Draw bounding box with comprehensive information
        if gender == "Male":
            color = (0, 255, 0)  # Green for male
        elif gender == "Female":
            color = (255, 0, 0)  # Red for female
        else:
            color = (0, 165, 255)  # Orange for unknown
            
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        
        # Add text with person information
        label = f"{gender}, {age_group}"
        cv2.putText(img_copy, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add emotion on second line
        cv2.putText(img_copy, f"Emotion: {emotion}", (x1, y1+y2-y1+20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add activity on third line
        cv2.putText(img_copy, f"Activity: {activity[:15]}...", (x1, y1+y2-y1+40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add confidence score
        cv2.putText(img_copy, f"Conf: {conf:.2f}", (x1, y1+y2-y1+60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        gender_results.append((gender, emotion, age_group, activity, conf, (x1, y1, x2, y2)))
    
    return img_copy, gender_results

def detect_business_context(image, yolo_results):
    """Detect business context in the image based on objects and scene"""
    business_indicators = {
        "laptop": 0,
        "cell phone": 0,
        "tie": 0,
        "book": 0,
        "chair": 0,
        "dining table": 0,
        "cup": 0,
        "bottle": 0,
        "suitcase": 0,
        "handbag": 0,
    }
    
    # Count business-related objects
    for result in yolo_results:
        boxes = result.boxes
        for box in boxes:
            cls = result.names[int(box.cls)]
            if cls in business_indicators:
                business_indicators[cls] += 1
    
    # Determine context based on objects
    context = []
    
    # Check for meeting context
    if business_indicators["laptop"] >= 1 and business_indicators["chair"] >= 2:
        context.append("Business Meeting")
    
    # Check for presentation context
    if business_indicators["laptop"] >= 1 and business_indicators["chair"] >= 3:
        context.append("Presentation")
    
    # Check for formal dining
    if business_indicators["dining table"] >= 1 and (business_indicators["cup"] >= 2 or business_indicators["bottle"] >= 2):
        context.append("Business Lunch/Dinner")
    
    # Check for formal attire
    if business_indicators["tie"] >= 1:
        context.append("Formal Business Setting")
    
    # Check for travel/commute
    if business_indicators["suitcase"] >= 1 or business_indicators["handbag"] >= 1:
        context.append("Business Travel")
    
    # Default context if nothing specific is detected
    if not context:
        if sum(business_indicators.values()) >= 3:
            context.append("General Business Environment")
        else:
            context.append("Casual Setting")
    
    return context

def analyze_person_interactions(person_boxes):
    """Analyze interactions between people based on their positions"""
    interactions = []
    
    # Check for pairs of people who might be interacting
    for i, box1 in enumerate(person_boxes):
        x1_1, y1_1, x2_1, y2_1 = map(int, box1.xyxy[0])
        center1 = ((x1_1 + x2_1) // 2, (y1_1 + y2_1) // 2)
        
        for j, box2 in enumerate(person_boxes):
            if i != j:  # Don't compare a person with themselves
                x1_2, y1_2, x2_2, y2_2 = map(int, box2.xyxy[0])
                center2 = ((x1_2 + x2_2) // 2, (y1_2 + y2_2) // 2)
                
                # Calculate distance between centers
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                # Calculate average height of the two people
                avg_height = ((y2_1 - y1_1) + (y2_2 - y1_2)) / 2
                
                # Determine interaction type based on distance
                if distance < avg_height * 0.8:
                    interactions.append((i, j, "Close Interaction"))
                elif distance < avg_height * 1.5:
                    interactions.append((i, j, "Conversation"))
                elif distance < avg_height * 3:
                    interactions.append((i, j, "Distant Interaction"))
    
    return interactions

def determine_business_activity(person_idx, interactions, business_context, emotion):
    """Determine business activity for a person based on context and interactions"""
    activity = []
    
    # Add business context
    if business_context:
        activity.extend(business_context)
    
    # Check person's interactions
    person_interactions = [inter for inter in interactions if inter[0] == person_idx or inter[1] == person_idx]
    
    if person_interactions:
        # Count interaction types
        close_count = sum(1 for inter in person_interactions if inter[2] == "Close Interaction")
        conv_count = sum(1 for inter in person_interactions if inter[2] == "Conversation")
        
        # Determine activity based on interactions and emotion
        if close_count >= 1:
            if "Happy/Excited" in emotion or "Confident" in emotion:
                activity.append("Handshake/Deal Closing")
            elif "Neutral" in emotion or "Thoughtful" in emotion:
                activity.append("Negotiation")
        
        if conv_count >= 1:
            if "Confident" in emotion:
                activity.append("Leading Discussion")
            elif "Thoughtful" in emotion:
                activity.append("Engaged in Discussion")
    else:
        # No interactions detected
        if "Confident" in emotion:
            activity.append("Independent Work")
        elif "Thoughtful" in emotion:
            activity.append("Analyzing/Planning")
    
    # Join all activities
    if not activity:
        return "No specific business activity detected"
    
    return ", ".join(activity)

if __name__ == "__main__":
    main()