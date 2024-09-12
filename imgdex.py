import streamlit as st
import torch
from torchvision import models
from torchvision.models import ResNet101_Weights
import torchvision.transforms as transforms
from PIL import Image

weights = ResNet101_Weights.IMAGENET1K_V2
model = models.resnet101(weights=weights)
model.eval()

transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def main():
    st.title("Imgdex")
    uploaded_file = st.file_uploader("Upload image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image...", use_column_width=True)
        final_image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(final_image)
            pred = torch.argmax(output, dim=1).item()
        
        class_names = weights.meta['categories']
        predicted_class = class_names[pred]
        #st.image(image.squeeze(0).permute(1, 2, 0))
        st.success(f"Detected : {predicted_class}")

if __name__ == "__main__":
    main()