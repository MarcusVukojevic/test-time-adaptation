import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# Define a function to apply the Grad-CAM method
def grad_cam(model, image_tensor, target_layer):
    gradients = None
    activations = None
    
    # Define hooks for gradients and activations
    def backward_hook(module, grad_in, grad_out):
        nonlocal gradients
        gradients = grad_out[0]
        
    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    # Register hooks
    hook_backward = target_layer.register_full_backward_hook(backward_hook)
    hook_forward = target_layer.register_forward_hook(forward_hook)

    # Forward pass
    output = model(image_tensor)
    _, predicted = output.max(dim=1)

    # Backward pass
    model.zero_grad()
    class_loss = output[0, predicted]
    class_loss.backward()

    # Generate heatmap
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap = heatmap.detach().cpu().numpy()
    
    # Clean up hooks
    hook_backward.remove()
    hook_forward.remove()

    return heatmap, predicted

# Image preprocessing
def process_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def resize_heatmap(heatmap, target_size, interpolation=Image.BILINEAR):
    """Resize heatmap to a target size."""
    heatmap_image = Image.fromarray(heatmap)
    heatmap_resized = heatmap_image.resize(target_size, interpolation)
    return np.array(heatmap_resized)

def normalize_heatmap(heatmap):
    """Normalize heatmap values to the range 0-1."""
    min_val = np.min(heatmap)
    max_val = np.max(heatmap)
    return (heatmap - min_val) / (max_val - min_val)