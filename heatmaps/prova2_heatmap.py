import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms.functional as TF
import numpy as np
import torch
import torchvision.models as models
from torchvision.models import resnet50, ResNet50_Weights
import cv2
from utils import grad_cam, process_image, resize_heatmap, normalize_heatmap

# Load the pretrained mode
'''
model = resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()  # Set the model to evaluation mode
'''
# Load a pre-trained Inception v3 model
model = models.inception_v3(pretrained=True, aux_logits=True)  # Disable aux_logits for simplicity
model.eval()

# Select the target layer
target_layer =  model.Mixed_7c #model.layer4[2] #

image_path = 'dogghi/doggo5.jpeg'
image = Image.open(image_path)
image_tensor = process_image(image_path)

# Generate heatmap
heatmap, predicted_label = grad_cam(model, image_tensor, target_layer)
print('Predicted Label:', predicted_label.item())

# Normalize the heatmap to ensure proper visualization
heatmap_normalized = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
heatmap_normalized = np.uint8(255 * heatmap_normalized)  # Convert to uint8

# Resize heatmap to the original image size for accurate masking
heatmap_resized = np.array(Image.fromarray(heatmap_normalized).resize(image.size, Image.BILINEAR))



# Apply a color map to the resized normalized heatmap
heatmap_colored = plt.cm.jet(heatmap_resized)  # Apply color map
heatmap_colored = np.uint8(heatmap_colored[:, :, :3] * 255)  # Convert to RGB

# Convert heatmap to PIL image for blending
heatmap_im = Image.fromarray(heatmap_colored)

# Convert the original image and heatmap to RGBA (adding an alpha channel for blending)
image_rgba = image.convert("RGBA")
heatmap_rgba = heatmap_im.convert("RGBA")

# Create a composite image with the heatmap
composite = Image.blend(image_rgba, heatmap_rgba, alpha=0.5)

threshold = 0.50 * 255
binary_mask = np.where(heatmap_resized >= threshold, 255, 0).astype(np.uint8)

# Find contours from the binary mask
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create bounding image only showing the areas within bounding boxes
bounding_image = np.zeros_like(np.array(image))  # Start with an all-black image

# Draw bounding boxes and mask inside them
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    bounding_image[y:y+h, x:x+w] = np.array(image)[y:y+h, x:x+w]  # Copy parts of the original image

# Create mask based on the heatmap threshold
mask = heatmap_resized >= threshold  # Create a binary mask

# Apply mask to the original image
masked_image = np.array(image) * np.stack([mask]*3, axis=-1)  # Ensure mask is applied across all color channels


# Displaying the composite and masked images
plt.figure(figsize=(15, 7))
plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(composite)
plt.title('Overlay Image')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(masked_image)
plt.title('Masked Image (50% Threshold)')
plt.axis('off')


plt.subplot(1, 4, 4)
plt.imshow(bounding_image)
plt.title('Bounding box (50% Threshold)')
plt.axis('off')

plt.show()