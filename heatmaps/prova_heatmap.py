import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

from torchvision.models import resnet50, ResNet50_Weights

from utils import grad_cam, process_image

model = resnet50(weights=ResNet50_Weights.DEFAULT)

model.eval()  # Set the model to evaluation mode

# Define a function to apply the Grad-CAM method

# Path to your image
image_path = 'dogghi/doggo3.jpeg'
image_tensor = process_image(image_path)

# Select the target layer
target_layer = model.layer4[2]

# Generate heatmap
heatmap, predicted_label = grad_cam(model, image_tensor, target_layer)

# Print predicted label
print('Predicted Label:', predicted_label.item())

# Display heatmap
plt.imshow(heatmap, cmap='hot', interpolation='nearest')
plt.title('Heatmap of Image')
plt.show()
