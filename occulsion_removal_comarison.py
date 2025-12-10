import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from pillow_heif import register_heif_opener
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import Inception_V3_Weights
from torchvision import transforms

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

if __name__ == "__main__":
    class_names = [
        "football", "baseball", "basketball", "billiard ball", "bowling ball", "cricket ball", "soccer ball", "golf ball", "field hockey ball", "hockey puck", "rugby ball", "shuttlecock", "table tennis ball", "tennis ball","volleyball"
    ]

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

    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    labels = []
    with open('data/occlusion/labels.txt', 'r') as file:
        for line in file:
            line = line.strip()  # Remove whitespace and newline
            if line:  # Skip empty lines
                labels.append(int(line))

    raw_correct = 10
    bilateral_correct = 0
    dncnn_correct = 12
    median_correct = 0
    resized_directory = 'data/occlusion/resized_original/'
    bilateral_directory = 'data/occlusion/bilateral_filtered/'
    dncnn_directory = 'data/occlusion/dncnn/'
    median_directory = 'data/occlusion/median/'
    
    for i in range(len(labels)):
        # img = cv2.imread(resized_directory + "occlusion"+str(i)+".png")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (299, 299))
        # if labels[i] == predict_image(img)[0]:
        #     raw_correct += 1
       #else:
            #print(f"Thought img {i} was {predict_image(img)[0]}, but it was actually {labels[i]}")
        
        img = cv2.imread(bilateral_directory + "occlusion"+str(i)+"_bilateral.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (299, 299))
        if labels[i] == predict_image(img)[0]:
            bilateral_correct += 1
        
        img = cv2.imread(median_directory + "occlusion"+str(i)+"_median.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (299, 299))
        if labels[i] == predict_image(img)[0]:
            median_correct += 1

        # img = cv2.imread(dncnn_directory + "occlusion"+str(i)+"_dncnn.png")
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (299, 299))
        # if labels[i] == predict_image(img)[0]:
        #     dncnn_correct += 1
        
        
    print(raw_correct, bilateral_correct, median_correct, dncnn_correct)