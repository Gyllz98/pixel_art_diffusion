import streamlit as st
import requests
import io
from PIL import Image


def generate_image(num_samples: int = 4):
    """Send a request to the backend to generate an image."""
    backend_url = "https://pixel-art-api-146224029275.europe-west1.run.app"
    generate_url = f"{backend_url}/generate"

    try:
        response = requests.post(
            generate_url,
            params={"num_samples": num_samples},  # Add num_samples as query parameter
            timeout=300,  # Longer timeout since generation might take time
        )
        if response.status_code == 200:
            # Convert the response bytes to an image
            image_bytes = response.content
            image = Image.open(io.BytesIO(image_bytes))
            return image
        st.error(f"Error: {response.status_code}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend: {e}")
        return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    st.title("Pixel Art Generator")

    # Add some information about the app
    st.write("Click the button below to generate a new pixel art image!")

    # Add number of samples input
    num_samples = st.number_input(
        "Number of Samples", min_value=1, max_value=64, value=8, help="Number of images to generate in the grid"
    )

    # Generate button
    if st.button("Generate Image", type="primary"):
        with st.spinner("Generating image..."):
            generated_image = generate_image(num_samples)

        if generated_image is not None:
            # Display the generated image
            st.image(generated_image, caption="Generated Image", use_column_width=True)

            # Add a download button
            buf = io.BytesIO()
            generated_image.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(label="Download Image", data=byte_im, file_name="generated_image.png", mime="image/png")


if __name__ == "__main__":
    main()
