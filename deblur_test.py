from richardsonlucy import richardson_lucy
from wiener import wiener_deconv
import numpy as np
import cv2
from PIL import Image
from pillow_heif import register_heif_opener
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import Inception_V3_Weights
from torchvision import transforms
import os
import matplotlib.pyplot as plt 

def apply_richardson_lucy(img, psf, steps=30):
    """
    Applies Richardson-Lucy deconvolution.
    """
    img_f = img.astype(np.float32) / 255.0
    img_t = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0) 

    # Prepare PSF for Torch
    psf_t = torch.from_numpy(psf).unsqueeze(0).unsqueeze(0) 

    x0 = img_t.clone()

    out_t = richardson_lucy(
        observation=img_t,
        x_0=x0,
        k=psf_t,
        steps=steps,
        clip=True,
        filter_epsilon=1e-12,
        tv=False
    )

    out = out_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out

def apply_wiener(img):
    """
    Applies Wiener deconvolution using the imported module.
    """
    img_f = img.astype(np.float32) / 255.0
    out = wiener_deconv(img_f)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out

def predict_image(image_array, model, transform):
    pil_image = Image.fromarray(image_array.astype('uint8'))
    input_tensor = transform(pil_image)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
    
    return predicted_class

def get_motion_kernel(size, angle):
    """ Generates a linear motion blur kernel (Line only). """
    size = int(size)
    if size % 2 == 0: size += 1 # Ensure odd size
    
    k = np.zeros((size, size), dtype=np.float32)
    center = (size - 1) / 2
    
    # Draw horizontal line
    k[int(center), :] = 1.0 
    
    # Rotate
    M = cv2.getRotationMatrix2D((center, center), angle, 1)
    k = cv2.warpAffine(k, M, (size, size))
    
    # Normalize
    k /= np.sum(k)
    return k


def show_psf_visualization(psf):
    """Visualizes the kernel being used for deconvolution."""
    plt.figure(figsize=(4, 4))
    plt.imshow(psf, cmap='gray', interpolation='nearest')
    plt.title("Point Spread Function (PSF)")
    plt.colorbar()
    plt.show()

def plot_final_accuracies(raw, rl, wiener, total):
    """Plots a bar chart of the accuracies."""
    methods = ['Raw', 'Richardson-Lucy', 'Wiener']
    accuracies = [raw/total, rl/total, wiener/total]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, accuracies, color=['red', 'green', 'blue'])
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy by Method')
    plt.ylim(0, 1.0)
    
    # Add text labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2%}', ha='center', va='bottom')
    plt.show()

def plot_sample_comparisons(samples, class_names):
    """
    Plots a grid of images: Raw vs RL vs Wiener for collected samples, 
    using class names for titles and uniform text color.
    """
    if not samples:
        print("\nNo samples collected for visualization.")
        return

    num_samples = len(samples)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = [axes]

    # Set uniform text color
    TEXT_COLOR = 'black'

    for i, sample in enumerate(samples):
        gt_name = class_names[sample['gt']] if 0 <= sample['gt'] < len(class_names) else f"Unknown ({sample['gt']})"
        pred_raw_name = class_names[sample['pred_raw']]
        pred_rl_name = class_names[sample['pred_rl']]
        pred_wiener_name = class_names[sample['pred_wiener']]
        
        # Helper function for title
        def make_title(pred_name, gt_name, pred_val, gt_val, method_name=""):
            status = 'Correct' if pred_val == gt_val else 'Incorrect'
            if method_name:
                return f"{method_name} | Pred: {pred_name} ({status})"
            else:
                return f"Raw | GT: {gt_name}\nPred: {pred_name} ({status})"


        # Raw
        axes[i][0].imshow(sample['raw'])
        axes[i][0].set_title(make_title(pred_raw_name, gt_name, sample['pred_raw'], sample['gt']),
                             color=TEXT_COLOR)
        axes[i][0].axis('off')

        # RL
        axes[i][1].imshow(sample['rl'])
        axes[i][1].set_title(make_title(pred_rl_name, gt_name, sample['pred_rl'], sample['gt'], "Richardson-Lucy"),
                             color=TEXT_COLOR)
        axes[i][1].axis('off')

        # Wiener
        axes[i][2].imshow(sample['wiener'])
        axes[i][2].set_title(make_title(pred_wiener_name, gt_name, sample['pred_wiener'], sample['gt'], "Wiener"),
                             color=TEXT_COLOR)
        axes[i][2].axis('off')

    plt.suptitle("Image Deconvolution and Classification Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Class names for visualization
    class_names = [
        "football", "baseball", "basketball", "billiard ball", "bowling ball", 
        "cricket ball", "soccer ball", "golf ball", "field hockey ball", 
        "hockey puck", "rugby ball", "shuttlecock", "table tennis ball", 
        "tennis ball","volleyball"
    ]

    print("Loading InceptionV3 Model...")
    model = models.inception_v3(weights=Inception_V3_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 15)
    
    checkpoint_path = 'sports_balls_inception_v3.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.eval()
        # Loading logic
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: {checkpoint_path} not found. Using random weights (predictions will be random).")

    register_heif_opener()

    labels = []
    if os.path.exists('data/blur/labels.txt'):
        with open('data/blur/labels.txt', 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    labels.append(int(line))
    else:
        print("Error: data/blur/labels.txt not found.")
        labels = [0] * 111 


    psf_size = 9
    angle = 0 
    psf = get_motion_kernel(psf_size, angle)

    # Creating the 2D Kernel (Based on user's current code, but flagged as incorrect for motion blur)
    psf = psf @ psf.T 
    psf = psf.astype(np.float32)

    raw_correct = 0
    rl_correct = 0
    wiener_correct = 0
    
    total_images = 111 
    
    # Store samples for visualization
    sample_rl_better = None # Stores the first RL success where Raw failed
    sample_raw_correct = None # Stores the first Raw success (general good example)

    print(f"Starting evaluation on {total_images} images...")

    for i in range(1, total_images + 1):
        
        filename = f"data/blur/blur{i:03d}.HEIC"
        
        if not os.path.exists(filename):
            if i == 1: print(f"Warning: {filename} not found.")
            continue
            
        try:
            raw_pil = Image.open(filename)
            raw_np = np.array(raw_pil.convert('RGB'))
            raw_resized = cv2.resize(raw_np, (299, 299)) 
        except Exception as e:
            print(f"Error opening {filename}: {e}")
            continue

        if (i-1) < len(labels):
            ground_truth = labels[i-1]
        else:
            ground_truth = -1
            continue

        pred_raw = predict_image(raw_resized, model, transform)
        if pred_raw == ground_truth:
            raw_correct += 1

        rl_img = apply_richardson_lucy(raw_resized, psf, steps=10)
        pred_rl = predict_image(rl_img, model, transform)
        if pred_rl == ground_truth:
            rl_correct += 1

        wiener_img = apply_wiener(raw_resized)
        pred_wiener = predict_image(wiener_img, model, transform)
        if pred_wiener == ground_truth:
            wiener_correct += 1

        current_sample = {
            'id': i,
            'raw': raw_resized.copy(),
            'rl': rl_img.copy(),
            'wiener': wiener_img.copy(),
            'pred_raw': pred_raw,
            'pred_rl': pred_rl,
            'pred_wiener': pred_wiener,
            'gt': ground_truth
        }

        if sample_rl_better is None:
            is_rl_better = (pred_rl == ground_truth) and (pred_raw != ground_truth)
            if is_rl_better:
                sample_rl_better = current_sample
        
        if sample_raw_correct is None and pred_raw == ground_truth:
            if sample_rl_better is None or current_sample['id'] != sample_rl_better['id']:
                sample_raw_correct = current_sample




        if i % 10 == 0:
            print(f"Processed {i}/{total_images}...")

    # Consolidate samples for plotting
    visualization_samples = []
    if sample_rl_better:
        visualization_samples.append(sample_rl_better)
    if sample_raw_correct:
        visualization_samples.append(sample_raw_correct)
    
    # If no RL improvement was found, ensure we at least show one sample
    if not visualization_samples and i > 0:
        visualization_samples.append(current_sample)
        
    print("-" * 30)
    print("Final Results:")
    # Use the last i as total_images if the loop broke early
    final_total = total_images if i == total_images else i 
    print(f"Total Images: {final_total}")
    print(f"Raw Accuracy:             {raw_correct}/{final_total} ({raw_correct/final_total if final_total > 0 else 0:.2%})")
    print(f"Richardson-Lucy Accuracy: {rl_correct}/{final_total} ({rl_correct/final_total if final_total > 0 else 0:.2%})")
    print(f"Wiener Accuracy:          {wiener_correct}/{final_total} ({wiener_correct/final_total if final_total > 0 else 0:.2%})")
    print("-" * 30)

    if visualization_samples:
        print("Displaying sample comparisons...")
        plot_sample_comparisons(visualization_samples, class_names)

    if final_total > 0:
        print("Displaying accuracy chart...")
        plot_final_accuracies(raw_correct, rl_correct, wiener_correct, final_total)