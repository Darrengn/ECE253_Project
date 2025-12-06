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

def global_cdf(intensities):
    """
    creates a global cdf from a one channel intensity image

    Args:
        intensities: 299x299 np array image of intensities get the global histogram of

    Returns:
        new_img: 256 length array where each entry at index i represents the global CDF(i)
    """
    pdf = np.zeros(256, dtype=float)
    cdf = np.zeros(256, dtype=float)
    pixel_count = len(intensities) * len(intensities[0])
    for i in range(len(intensities)):
        for j in range(len(intensities[0])):
            pdf[intensities[i,j]] += 1
    pdf /= pixel_count
    total = 0
    for i in range(len(pdf)):
        total += pdf[i]
        cdf[i] = total
    return cdf

def anhe(img, N_max = 100, K = 3, T = 20):
    """
    Runs adaptive neighborhood histogram equalization on the input image

    Args:
        img: 299x299x3 np array image in RGB format of image to run adjustment on
        N_max: max number of pixels in the adjustable neighborhood
        K: multiplicitive constant for standard deviation of neighborhood
        T: maximum pixel intensity difference from current pixel for neighborhood

    Returns:
        new_img: 299x299x3 np array image in RGB format of contrast adjusted image
    """
    # convert 3 channels into one intensity channel
    # intensities = np.round(0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(int) 
    intensities = np.round(0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]).astype(int)
    out_intens = np.zeros(intensities.shape)
    out_img = np.zeros_like(img)
    g_cdf = global_cdf(intensities)
    global_hist = 255 * g_cdf
    # loop through every pixel and get the new pixel intensity
    dirs = [(1,1), (1,0), (1,-1), (0,1), (0,-1), (-1,1), (-1,0), (-1,-1)]
    for i in range(len(intensities)):
        for j in range(len(intensities[0])):
            cur_int = intensities[i,j]
            queue = [(i,j)]
            seen = {}
            count = 1
            neighborhood = [cur_int]
            hist = np.zeros(256)
            # find the adjustable neighborhood
            while queue and count < N_max:
                cur = queue.pop(0)
                if cur in seen:
                    continue
                for dir in dirs:
                    next = (cur[0] + dir[0], cur[1] + dir[1])
                    if next in seen or next[0] < 0 or next[0] >= len(intensities) or next[1] < 0 or next[1] >= len(intensities):
                        continue
                    next_int = intensities[next[0],next[1]]
                    if abs(next_int - cur_int) <= T:
                        queue.append(next)
                        neighborhood.append(next_int)
                        hist[next_int] += 1
                        count += 1
            # neighborhood is found
            hist /= count
            n_cdf = np.zeros(256)
            total = 0
            for k in range(len(hist)):
                total += hist[k]
                n_cdf[k] = total
            
            mean = round(np.mean(neighborhood))
            std = np.std(neighborhood)
            imin = max(round(global_hist[mean] - K * std), 0)
            imax = min(round(global_hist[mean] + K * std), 255)
            out_intens[i,j] = round(imin + (imax - imin) * n_cdf[cur_int])
            if cur_int == 0:
                ratio = 0
            else:
                ratio = out_intens[i,j] / cur_int
            out_img[i,j] = np.clip(ratio * img[i,j], 0, 255)
            
    return out_img

def contrast_adjust(img, delta = 50, lam = 1.4):
    intensities = np.round(0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2])/255
    mean = np.mean(intensities)
    gamma = np.sum(((intensities-mean))**2) / (299*299 - 1) * delta
    y = 1 + lam*abs(gamma)
    new_int = y * (intensities + gamma)
    ratio = np.where(intensities != 0.0, new_int / intensities, 0)
    
    new_img = np.clip(img * ratio[:, :, np.newaxis], 0, 255).astype(np.uint8)
    return new_img
    
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

    register_heif_opener()

    
    deltas = []
    with open('data/contrast/deltas.txt', 'r') as file:
        for line in file:
            line = line.strip()  # Remove whitespace and newline
            if line:  # Skip empty lines
                deltas.append(int(line))

    labels = []
    with open('data/contrast/labels.txt', 'r') as file:
        for line in file:
            line = line.strip()  # Remove whitespace and newline
            if line:  # Skip empty lines
                labels.append(int(line))

    # cur = range(1,96)
    # for i in cur:
    #     print(i)
    #     img = Image.open("data/contrast/con"+str(i)+".HEIC")
    #     img = np.array(img.convert('RGB'))
    #     img = cv2.resize(img, (299, 299))
    #     out = anhe(img)
    #     out = cv2.medianBlur(out,5)
    #     out2 = contrast_adjust(img, delta=deltas[i-1])
    #     cv2.imwrite("output/contrast/anhe"+str(i)+".png",cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    #     cv2.imwrite("output/contrast/cs"+str(i)+".png",cv2.cvtColor(out2, cv2.COLOR_RGB2BGR))
    #     print(class_names[predict_image(out)[0]])
    #     print(class_names[predict_image(out2)[0]])

    # cur = range(96,113)
    # for i in cur:
    #     print(i)
    #     img = Image.open("data/contrast/con"+str(i)+".jpg")
    #     img = np.array(img.convert('RGB'))
    #     img = cv2.resize(img, (299, 299))
    #     out = anhe(img)
    #     out = cv2.medianBlur(out,5)
    #     out2 = contrast_adjust(img, delta=deltas[i-1])
    #     cv2.imwrite("output/contrast/anhe"+str(i)+".png",cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    #     cv2.imwrite("output/contrast/cs"+str(i)+".png",cv2.cvtColor(out2, cv2.COLOR_RGB2BGR))
    #     print(class_names[predict_image(out)[0]])
    #     print(class_names[predict_image(out2)[0]])

    raw_correct = 0
    anhe_correct = 0
    cs_correct = 0
    for i in range(len(labels)):
        img = cv2.imread("output/contrast/cs"+str(i+1)+".png")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (299, 299))
        if labels[i] == predict_image(img)[0]:
            cs_correct += 1

        if i > 94:
            raw = Image.open("data/contrast/con"+str(i+1)+".jpg")
        else:
            raw = Image.open("data/contrast/con"+str(i+1)+".HEIC")
        raw = np.array(raw.convert('RGB'))
        raw = cv2.resize(raw, (299, 299))
        if labels[i] == predict_image(raw)[0]:
            raw_correct += 1
            

        img = cv2.imread("output/contrast/anhe"+str(i+1)+".png")
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (299, 299))
        if labels[i] == predict_image(img)[0]:
            anhe_correct += 1

        
        
    print(raw_correct, anhe_correct, cs_correct)
