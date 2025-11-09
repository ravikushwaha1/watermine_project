import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Underwater Vision AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-top: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .info-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
    .upload-box {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")
    
    # Detection settings
    st.subheader("Detection Parameters")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    edge_low = st.slider("Edge Detection Low", 50, 150, 100, 10)
    edge_high = st.slider("Edge Detection High", 100, 300, 200, 10)
    
    st.markdown("---")
    st.subheader("📊 App Info")
    st.info("""
    **Underwater Vision AI**
    
    Features:
    • Image Upload & Display
    • Edge Detection
    • Color-based Object Detection
    • Real-time Processing
    """)

# Main content
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="main-header">🌊 Underwater Vision AI</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">Advanced Underwater Image Analysis & Object Detection</div>', unsafe_allow_html=True)

# File upload section with better styling
st.markdown("### 📤 Upload Your Underwater Image")
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    " ",
    type=['jpg', 'jpeg', 'png'],
    help="Upload underwater images for analysis",
    label_visibility="collapsed"
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Progress bar
    with st.spinner("🔄 Processing your image..."):
        # Display original image
        image = Image.open(uploaded_file)
        
        # Create columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📷 Original Image")
            st.image(image, use_container_width=True, caption="Uploaded Underwater Image")
            
            # Image info
            st.markdown("#### 📋 Image Information")
            st.write(f"**Dimensions:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Format:** {image.format}")
            st.write(f"**Mode:** {image.mode}")
        
        with col2:
            # Convert to OpenCV format
            image_cv = np.array(image)
            image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)
            
            # Edge Detection
            st.markdown("### 🔍 Edge Detection")
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, edge_low, edge_high)
            st.image(edges, use_container_width=True, caption="Edge Detection Result", clamp=True)
            
            # Color-based Object Detection
            st.markdown("### 🎯 Object Detection")
            hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
            
            # Detect blue objects (common underwater)
            lower_blue = np.array([100, 150, 0])
            upper_blue = np.array([140, 255, 255])
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Find contours
            contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw detected objects
            detected_image = image_cv.copy()
            object_count = 0
            
            for contour in contours:
                if cv2.contourArea(contour) > 100:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(detected_image, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    cv2.putText(detected_image, f'Object {object_count+1}', (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    object_count += 1
            
            detected_image_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)
            st.image(detected_image_rgb, use_container_width=True, caption=f"Detected {object_count} Objects")
    
    # Results section
    st.markdown("---")
    st.markdown("## 📊 Analysis Results")
    
    results_col1, results_col2, results_col3 = st.columns(3)
    
    with results_col1:
        st.metric("Objects Detected", object_count)
    
    with results_col2:
        st.metric("Image Width", f"{image.size[0]} px")
    
    with results_col3:
        st.metric("Image Height", f"{image.size[1]} px")
    
    # Success message
    st.markdown("---")
    if object_count > 0:
        st.markdown(f"""
        <div class="success-box">
            <h3>✅ Analysis Complete!</h3>
            <p>Successfully detected <strong>{object_count}</strong> objects in your underwater image.</p>
        </div>
        """, unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown("""
        <div class="info-box">
            <h3>ℹ️ Analysis Complete</h3>
            <p>No significant objects detected. Try adjusting the detection parameters or upload a different image.</p>
        </div>
        """, unsafe_allow_html=True)

else:
    # Welcome message when no file uploaded
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h3>🚀 Get Started</h3>
            <p>Upload an underwater image to begin analysis.</p>
            <p><em>Supported formats: JPG, JPEG, PNG</em></p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🌊 Underwater Vision AI • Powered by Streamlit & OpenCV"
    "</div>",
    unsafe_allow_html=True
)