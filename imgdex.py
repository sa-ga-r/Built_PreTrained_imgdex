import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
model.eval()

transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def main():
    st.title("Imgdex")
    uploaded_file = st.file_uploader("Upload image...")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            _, pred = torch.max(output, 1)
        
        class_names = model.class_to_idx.keys()
        predicted_class = class_names[pred.item()]
        st.image(image.squeeze(0).permute(1, 2, 0))
        st.write("Detected :", predicted_class)

if __name__ == "__main__":
    main()