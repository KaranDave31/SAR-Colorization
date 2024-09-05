import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error
import io  # Required for in-memory file handling
import base64

# Load the model
model = tf.keras.models.load_model('sar_colorization_unet_agri.h5', compile=False)

# Custom CSS to apply dark mode-like styling




# Function to preprocess the image
def preprocess_image(image):
    # Resize to the input size expected by the model
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0  # Normalize the image to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array


# Function to postprocess the image


def postprocess_image(image_array):
    image_array = np.squeeze(image_array, axis=0)  # Remove batch dimension
    image_array = (image_array * 255).astype(np.uint8)  # Convert to uint8
    return Image.fromarray(image_array)


# Function to predict the colorization


def predict_colorization(image):
    processed_image = preprocess_image(image)
    colorized_image = model.predict(processed_image)
    colorized_image = postprocess_image(colorized_image)
    return colorized_image


# Function to display metrics


def display_metrics(original, colorized):
    colorized_resized = colorized.resize(original.size)
    original = np.array(original)
    colorized_resized = np.array(colorized_resized)

    psnr_value = psnr(original, colorized_resized)
    ssim_value = ssim(original, colorized_resized, multichannel=True)
    mse_value = mean_squared_error(original, colorized_resized)

    st.write(f"**PSNR**: {psnr_value:.2f}")
    st.write(f"**SSIM**: {ssim_value:.2f}")
    st.write(f"**MSE**: {mse_value:.2f}")


# Convert image to base64


def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


# Function to display the home page


def show_main_page():
    col1, col2 = st.columns([3, 2])

    

    st.markdown(
        """
        <div style='text-align: left;'>
            <h1>SAR Image Colorization</h1>
            <h7>A tool for enhancing SAR imagery interpretation using Deep Learning.</h7>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """<hr style="height:2px;border:none;color:#333;background-color:#333;" /> """,
        unsafe_allow_html=True)
    show_summary_section()


# Function to display the summary section


def show_summary_section():
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            """
            <div style='text-align: left;'>
                <h2>Summary</h2>
                <p>This project aims to enhance SAR imagery interpretation using Deep Learning for colorization. 
                It is developed using technologies like Python, TensorFlow, and PIX2PIX.</p>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <h2>Major Use Cases</h2>
                <ul>
                    <li><b>Farms</b>: Enhance crop monitoring, yield prediction, and soil analysis.</li>
                    <li><b>Urban Planning</b>: Assist in urban mapping and infrastructure development.</li>
                    <li><b>Grasslands</b>: Monitor vegetation health and manage land use for wildlife habitats.</li>
                    <li><b>Barrenlands</b>: Detect barren and degraded lands for potential restoration efforts.</li>
                    <li><b>Geological Studies</b>: Improve visual interpretation of geological formations.</li>
                    <li><b>Defense</b>: Enhance surveillance, reconnaissance, and terrain analysis for strategic planning.</li>
                    <li><b>Disaster Management</b>: Support rapid assessment and response during natural disasters like floods and earthquakes.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="display: flex; align-items: center; height: 70%;">
                <img src='data:image/png;base64,{}' class='img-fluid' style='padding-top: 5%; padding-bottom: 10%; width: 100%; height:80%'>
            </div>
            """.format(image_to_base64(
                "img1.jpg")),  # Convert image to base64 for embedding
            unsafe_allow_html=True)


# Function to display domain-specific sections


def show_domain_section(domain):
    domain_details = {
        'Farm':
        "SAR image colorization helps in distinguishing crop types and assessing crop health.",
        'Urban':
        "In urban areas, colorization aids in infrastructure planning and disaster management.",
        'Grassland':
        "Colorized images can help in monitoring vegetation and wildlife habitats.",
        'Barrenland':
        "Enhances the detection of barren and degraded lands for restoration efforts."
    }

    st.write(f"### {domain} Domain")
    st.write(domain_details[domain])

    # File uploader for multiple SAR images
    uploaded_files = st.file_uploader(f"Upload SAR images for {domain}...",
                                      type=["png", "jpg", "jpeg"],
                                      accept_multiple_files=True)

    if uploaded_files:
        images = [
            Image.open(uploaded_file).convert('L')
            for uploaded_file in uploaded_files
        ]

        if st.button('Start Colorization'):
            with st.spinner('Colorizing the images...'):
                colorized_images = [
                    predict_colorization(image) for image in images
                ]

                # Display input and colorized images side by side
                st.write("### Image Comparisons")
                for idx, (original_image, colorized_image) in enumerate(
                        zip(images, colorized_images)):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(original_image,
                                 caption=f'Original Image {idx + 1}',
                                 use_column_width=True)
                    with col2:
                        st.image(colorized_image,
                                 caption=f'Colorized Image {idx + 1}',
                                 use_column_width=True)

                    # Optional: Display metrics if needed
                    # display_metrics(original_image, colorized_image)

                st.success('Colorization Complete!')

                # Add download buttons for each colorized image
                for idx, colorized_image in enumerate(colorized_images):
                    buffered = io.BytesIO()  # Create in-memory buffer
                    # Save colorized image to buffer
                    colorized_image.save(buffered, format="PNG")
                    img_bytes = buffered.getvalue(
                    )  # Get the byte data of the image

                    st.download_button(
                        label=f"Download Colorized Image {idx + 1}",
                        data=img_bytes,
                        file_name=f"colorized_image_{idx + 1}.png",
                        mime="image/png")


st.set_page_config(layout="wide")
# Main function to control the page display


def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'Main Page'

    st.sidebar.title("Navigation")
    domain_options = ['Main Page', 'Farm', 'Urban', 'Grassland', 'Barrenland']

    selected_page = st.sidebar.selectbox("Choose a section", domain_options)

    st.session_state.page = selected_page

    if st.session_state.page == 'Main Page':
        show_main_page()
    else:
        show_domain_section(st.session_state.page)


if __name__ == "__main__":
    main()
