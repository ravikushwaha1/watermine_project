import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Underwater Object Detection",
    page_icon="🌊",
    layout="wide"
)

# Title and description
st.title("🌊 Underwater Object Detection")
st.write("Upload images or videos to detect underwater objects using YOLOv8")

# Load the model
@st.cache_resource
def load_model():
    try:
        model = YOLO('best.pt')
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# Sidebar for configuration
st.sidebar.header("Configuration")

# Confidence threshold
confidence = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.5, 
    help="Adjust the detection confidence threshold"
)

# File upload section
uploaded_file = st.file_uploader(
    "Choose an image or video file",
    type=['jpg', 'jpeg', 'png', 'mp4', 'avi', 'mov'],
    help="Supported formats: JPG, PNG, MP4, AVI, MOV"
)

def process_image(image, model, conf_threshold):
    """Process image and return detection results"""
    # Convert PIL image to numpy array
    image_np = np.array(image)
    
    # Run YOLO inference
    results = model(image_np, conf=conf_threshold)
    
    # Plot results on image
    annotated_image = results[0].plot()
    
    # Convert BGR to RGB for display
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return annotated_image_rgb, results[0]

def process_video(video_path, model, conf_threshold):
    """Process video and return detection results"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create temporary output file
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run YOLO inference
        results = model(frame, conf=conf_threshold)
        
        # Plot results on frame
        annotated_frame = results[0].plot()
        
        # Write frame to output video
        out.write(annotated_frame)
        
        processed_frames += 1
        progress = processed_frames / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing video: {processed_frames}/{total_frames} frames")
    
    cap.release()
    out.release()
    
    progress_bar.empty()
    status_text.empty()
    
    return output_path

# Main processing logic
if uploaded_file is not None:
    # Display file info
    file_details = {
        "Filename": uploaded_file.name,
        "File size": f"{uploaded_file.size / 1024:.2f} KB",
        "File type": uploaded_file.type
    }
    st.write("File details:", file_details)
    
    # Process based on file type
    if uploaded_file.type.startswith('image'):
        # Display original image
        image = Image.open(uploaded_file)
        st.subheader("📷 Original Image")
        st.image(image, use_column_width=True)
        
        # Process and display detected image
        st.subheader("🔍 Detected Objects")
        with st.spinner("Detecting objects..."):
            detected_image, results = process_image(image, model, confidence)
            st.image(detected_image, use_column_width=True)
            
            # Display detection statistics
            if len(results.boxes) > 0:
                st.success(f"✅ Detected {len(results.boxes)} objects")
                
                # Show class distribution
                classes = results.boxes.cls.cpu().numpy()
                unique_classes, counts = np.unique(classes, return_counts=True)
                
                st.subheader("📊 Detection Statistics")
                for cls, count in zip(unique_classes, counts):
                    class_name = model.names[int(cls)]
                    st.write(f"- {class_name}: {count}")
            else:
                st.warning("⚠️ No objects detected")
                
    elif uploaded_file.type.startswith('video'):
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Display original video
        st.subheader("🎬 Original Video")
        st.video(uploaded_file.getvalue())
        
        # Process video
        st.subheader("🎯 Processed Video with Detections")
        with st.spinner("Processing video... This may take a while depending on video length."):
            output_video_path = process_video(video_path, model, confidence)
            
            # Display processed video
            with open(output_video_path, 'rb') as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)
            
            # Clean up temporary files
            os.unlink(video_path)
            os.unlink(output_video_path)
    
    else:
        st.error("Unsupported file type")

else:
    st.info("👆 Please upload an image or video file to get started")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit & YOLOv8 | Underwater Object Detection")