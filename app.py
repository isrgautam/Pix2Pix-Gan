import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import config
from generator_model import Generator

# Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def load_model():
    """Load the trained GAN model."""
    model = Generator(in_channels=3, features=64).to(config.DEVICE)
    model.load_state_dict(torch.load(config.CHECKPOINT_GEN, map_location=config.DEVICE)["state_dict"])
    model.eval()
    return model

def main():
    st.title("GAN Image Generator")
    st.write("Upload an image to generate a new one using the trained GAN model.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        # Load and preprocess the image
        image = Image.open(uploaded_file).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(config.DEVICE)

        # Load the model and generate an image
        model = load_model()
        with torch.no_grad():
            generated_image_tensor = model(image_tensor)
            generated_image_tensor = (generated_image_tensor + 1) / 2  # Denormalize
            generated_image_tensor = generated_image_tensor.squeeze().cpu()
            generated_image = transforms.ToPILImage()(generated_image_tensor)

        # Display the images
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.image(generated_image, caption="Generated Image", use_column_width=True)

if __name__ == "__main__":
    main()

#%%
