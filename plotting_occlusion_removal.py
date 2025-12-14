import matplotlib.pyplot as plt
import cv2
import os
import torch
import torch.nn as nn
from PIL import Image
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

    resized_directory = 'data/occlusion/resized_original/'
    bilateral_directory = 'data/occlusion/bilateral_filtered/'
    dncnn_directory = 'data/occlusion/dncnn/'
    median_directory = 'data/occlusion/median/'

    fig, ax = plt.subplots(2, 4, figsize=(8, 8))
    imgs = [48, 72]
    for i, img_num in enumerate(imgs):
        original_path = os.path.join(resized_directory, f'occlusion{img_num}.png')
        bilateral_path = os.path.join(bilateral_directory, f'occlusion{img_num}_bilateral.png')
        dncnn_path = os.path.join(dncnn_directory, f'occlusion{img_num}_dncnn.png')
        median_path = os.path.join(median_directory, f'occlusion{img_num}_median.png')
        original_img = cv2.imread(original_path)
        bilateral_img = cv2.imread(bilateral_path)
        dncnn_img = cv2.imread(dncnn_path)
        median_img = cv2.imread(median_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        bilateral_img = cv2.cvtColor(bilateral_img, cv2.COLOR_BGR2RGB)
        dncnn_img = cv2.cvtColor(dncnn_img, cv2.COLOR_BGR2RGB)
        median_img = cv2.cvtColor(median_img, cv2.COLOR_BGR2RGB)
        original_img = cv2.resize(original_img, (299, 299))
        bilateral_img = cv2.resize(bilateral_img, (299, 299))
        dncnn_img = cv2.resize(dncnn_img, (299, 299))
        median_img = cv2.resize(median_img, (299, 299))
        real_label = labels[img_num]
        pred_orig, _ = predict_image(original_img)
        pred_bilat, _ = predict_image(bilateral_img)
        pred_dncnn, _ = predict_image(dncnn_img)
        pred_median, _ = predict_image(median_img)

        ax[i, 0].imshow(original_img)
        ax[i, 0].set_title(f"Raw \n {class_names[pred_orig]}")
        ax[i, 0].axis('off')

        ax[i, 1].imshow(bilateral_img)
        ax[i, 1].set_title(f"Bilateral Filtering \n {class_names[pred_bilat]}")
        ax[i, 1].axis('off')

        ax[i, 2].imshow(dncnn_img)
        ax[i, 2].set_title(f"DnCNN \n {class_names[pred_dncnn]}")
        ax[i, 2].axis('off')

        ax[i, 3].imshow(median_img)
        ax[i, 3].set_title(f"Median Filtering \n {class_names[pred_median]}")
        ax[i, 3].axis('off')

    plt.show()