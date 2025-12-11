# ECE 253 Project
## Image Denoising for Sports Ball Classification
### Requriements
numpy, matplotlib, cv2, pytorch, torchvision, skimages

### Running combined noise removal
Change line 63 with the path to the image you want to run deblur, brightness adjust, and occlusion removal on.
run `python main.py`

### Running brightness adjustment
labels are in data/contrast/labels.txt
delta values for contrast stretching are in data/contrast/deltas.txt
run `python bright_adjust.py`
this will create an output folder with all the images after both techniques

### Running deblur
labels are in data/blur/labels.txt
run `python deblur_test.py`
this will output visualizations and plots in a new window, and print out accuracy of each method in terminal