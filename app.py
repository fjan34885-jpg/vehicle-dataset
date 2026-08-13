
import streamlit as st
import os
import tempfile
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO

# --- Configuration --- #
MODEL_PATH = 'best.pt' # The trained model file must be in the same directory
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.7

# --- Load Model (Cache to avoid reloading on every rerun) ---
@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.stop()

model = load_yolo_model()

# --- Streamlit App UI ---
st.set_page_config(page_title="YOLOv8 Vehicle Detection", layout="wide")
st.title("🚗 YOLOv8 Vehicle Detection App")
st.markdown("Upload an image or video to detect vehicles using a pre-trained YOLOv8 model.")

# --- File Uploader ---
uploaded_file = st.file_uploader("Choose an image or video file", type=['jpg', 'jpeg', 'png', 'mp4', 'mov', 'avi'])

if uploaded_file is not None:
    file_type = uploaded_file.type

    # Create a temporary file to save the uploaded content
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    if "image" in file_type:
        st.subheader("Processing Image...")
        # Display original image
       st.image(uploaded_file, caption="Original Image", use_container_width=True)

        # Run inference
        results = model.predict(source=temp_file_path, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)

        # Process and display results
        if results and len(results) > 0:
            annotated_img_array = results[0].plot() # Get annotated image as numpy array
            annotated_img_pil = Image.fromarray(cv2.cvtColor(annotated_img_array, cv2.COLOR_BGR2RGB))
           st.image(annotated_img_pil, caption="Detected Vehicles", use_container_width=True)
            st.success(f"Detected {len(results[0].boxes)} vehicles.")
        else:
            st.warning("No vehicles detected in the image.")

    elif "video" in file_type:
        st.subheader("Processing Video...")
        st.warning("Video processing can take a while depending on its length and your internet speed.")

        # Define output video path
        output_video_path = os.path.join(os.path.dirname(temp_file_path), "output_video.mp4")

        # Run inference on the video with saving option
        # We need to explicitly save the output from predict in a custom location if we want to display it.
        # Ultralytics predict saves to runs/detect/..., but for Streamlit, we want a direct path.
        video_results_generator = model.predict(
            source=temp_file_path,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            save=True, # Save annotated video
            project='streamlit_inference', # Custom project for Streamlit runs
            name='video_output', # Custom name for Streamlit video output
            exist_ok=True # Overwrite existing directory if it exists
        )

        total_frames = 0
        total_detections = 0
        output_video_saved_path = None

        for res in video_results_generator:
            if res.boxes is not None:
                total_detections += len(res.boxes)
            total_frames += 1
            # The first result object will contain the save_dir for the video
            if output_video_saved_path is None and res.save_dir is not None:
                # The video is saved within a 'video_output' folder inside the save_dir
                output_video_saved_path = os.path.join(res.save_dir, uploaded_file.name) # Ultralytics uses original filename


        if output_video_saved_path and os.path.exists(output_video_saved_path):
            st.success(f"Video processing complete! Total frames: {total_frames}, Total detections: {total_detections}")
            st.video(output_video_saved_path, format="video/mp4", start_time=0)
            st.download_button(
                label="Download Annotated Video",
                data=open(output_video_saved_path, 'rb').read(),
                file_name=f"detected_{uploaded_file.name}",
                mime="video/mp4"
            )
        else:
            st.error("Failed to process video or find output.")

    # Clean up the temporary file
    os.unlink(temp_file_path)
    # If video, also clean up the ultralytics generated runs folder if needed, but not strictly necessary for this example.

