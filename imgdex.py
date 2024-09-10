import torch
import streamlit as st
from PIL import Image
from torchvision import models, transforms

uploaded_img = st.file_uploader("Upload an image", type='jpg')
img = Image.open(uploaded_img)
st.image(img, caption="Uploaded image", use_column_width=True)

model = models.resnet101(prettained = True)

transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

