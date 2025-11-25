import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from pillow_heif import register_heif_opener
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import Inception_V3_Weights
from torchvision import transforms
from bright_adjust import *


def predict_image(image_array):
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image_array.astype('uint8'))
    
    # Apply transforms
    input_tensor = transform(pil_image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    
    # Make prediction
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)  # Use output[0] for inception v3
        predicted_class = torch.argmax(probabilities).item()
    
    return predicted_class, probabilities.numpy()

transform = transforms.Compose([
    transforms.Resize(299),             # resize shortest side to 299 pixels
    transforms.CenterCrop(299),         # crop to 299x299 at center
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 15)
checkpoint = torch.load('sports_balls_inception_v3.pth', map_location='cpu')

model.eval()

# Handle checkpoint loading
if isinstance(checkpoint, dict):
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
else:
    model.load_state_dict(checkpoint)

register_heif_opener()
img = Image.open("data/contrast/con1.HEIC")
img = np.array(img.convert('RGB'))
img = cv2.resize(img, (299, 299))

# out = anhe(img)
out2 = contrast_adjust(img)

fig, axes = plt.subplots(2, 2)
axes[0,0].imshow(img)
# axes[0,1].imshow(out)
axes[1,0].imshow(out2)
print(predict_image(img), predict_image(out2))

plt.show()


