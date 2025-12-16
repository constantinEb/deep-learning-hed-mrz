"""
Streamlit frontend for MRZ field segmentation inference.
"""
import streamlit as st
import requests
import base64
import io
from PIL import Image
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="MRZ Field Segmentation",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #616161;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)


def check_backend_health():
    """Check if backend is healthy"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except Exception as e:
        return False, str(e)


def encode_image_to_base64(uploaded_file):
    """Convert uploaded file to base64"""
    bytes_data = uploaded_file.getvalue()
    b64_string = base64.b64encode(bytes_data).decode("utf-8")

    if uploaded_file.type == "image/jpeg":
        mime_type = "image/jpeg"
    elif uploaded_file.type == "image/png":
        mime_type = "image/png"
    else:
        mime_type = "image/png"

    return f"data:{mime_type};base64,{b64_string}"


def decode_base64_image(base64_str):
    """Decode base64 string to PIL Image"""
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]

    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes))

    return img


def run_inference(image_b64, model_type):
    """Call backend API for inference"""
    endpoint = f"{BACKEND_URL}/predict/{model_type}"

    payload = {
        "image": image_b64
    }

    try:
        response = requests.post(endpoint, json=payload, timeout=30)

        if response.status_code == 200:
            return response.json()
        else:
            return {
                "success": False,
                "error": f"API error: {response.status_code} - {response.text}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": f"Request failed: {str(e)}"
        }


def main():
    st.markdown('<div class="main-header">🔍 MRZ Field Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a document image to detect MRZ regions</div>', unsafe_allow_html=True)

    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("Backend Status")
        health_status, health_data = check_backend_health()

        if health_status:
            st.success("✅ Backend is healthy")
            if health_data:
                st.json({
                    "status": health_data.get("status"),
                    "device": health_data.get("device"),
                    "models_loaded": health_data.get("models_loaded")
                })
        else:
            st.error("❌ Backend is unavailable")
            st.warning(f"Error: {health_data}")
            st.stop()

        st.divider()

        st.subheader("Model Selection")
        model_type = st.selectbox(
            "Choose model:",
            options=["hough_encoder", "hed_mrz"],
            index=0,
            help="Select which trained model to use for inference"
        )

        st.divider()

        st.subheader("ℹ️ About")
        st.markdown("""
        This application detects MRZ (Machine Readable Zone) regions in document images.

        **Supported Models:**
        - **hough_encoder**: Uses Hough transform encoder
        - **hed_mrz**: Uses HED-based MRZ detector

        **Output Visualizations:**
        1. **Original**: Input image
        2. **Heatmap**: Probability map
        3. **Overlay**: Binary mask overlay
        4. **OBB**: Oriented bounding boxes
        """)

    st.header("📤 Upload Image")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload a document image (JPEG or PNG)"
    )

    if uploaded_file is not None:
        st.subheader("Uploaded Image")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(uploaded_file, caption="Original Image", use_column_width=True)

        if st.button("🚀 Run Inference", type="primary"):
            with st.spinner("Running inference..."):
                image_b64 = encode_image_to_base64(uploaded_file)

                result = run_inference(image_b64, model_type)

                if result.get("success"):
                    st.success("✅ Inference completed successfully!")

                    st.header("📊 Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("🖼️ Original")
                        original_img = decode_base64_image(result["images"]["original"])
                        st.image(original_img, use_column_width=True)

                        st.subheader("🎨 Overlay")
                        overlay_img = decode_base64_image(result["images"]["overlay"])
                        st.image(overlay_img, use_column_width=True)

                    with col2:
                        st.subheader("🔥 Heatmap")
                        heatmap_img = decode_base64_image(result["images"]["heatmap"])
                        st.image(heatmap_img, use_column_width=True)

                        st.subheader("📦 OBB Detection")
                        obb_img = decode_base64_image(result["images"]["obb"])
                        st.image(obb_img, use_column_width=True)

                    st.header("💾 Download Results")
                    download_cols = st.columns(4)

                    image_names = ["original", "heatmap", "overlay", "obb"]
                    for idx, name in enumerate(image_names):
                        with download_cols[idx]:
                            img = decode_base64_image(result["images"][name])
                            buffer = io.BytesIO()
                            img.save(buffer, format="PNG")
                            buffer.seek(0)

                            st.download_button(
                                label=f"Download {name.capitalize()}",
                                data=buffer,
                                file_name=f"{name}_{model_type}.png",
                                mime="image/png"
                            )

                else:
                    st.error(f"❌ Inference failed: {result.get('error')}")

    else:
        st.info("👆 Please upload an image to begin")


if __name__ == "__main__":
    main()
