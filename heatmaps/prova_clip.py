from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import numpy as np


# Load the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


# Prepare your image and text
image_path = "dogghi/doggo4.jpeg"
image = Image.open(image_path)
text = ["a photo of a orange background"]  # Change this to describe your image

# Use the processor to prepare inputs
inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

# Make the model return attention maps and forward the inputs
outputs = model(**inputs, output_attentions=True, return_dict=True)

# Extract and process attention maps from all layers
attentions = outputs.vision_model_output.attentions  # This should be a tuple of 12 elements for 12 layers

# Setting up the plot for all layers
fig, axs = plt.subplots(3, 4, figsize=(20, 15))  # Adjust subplot grid as needed
axs = axs.flatten()  # Flatten if using a grid

for i, attention in enumerate(attentions):
    # Average across heads
    attention = attention
    head_avg_attention = attention.mean(dim=1)  # dim=1 for head dimension

    # Optionally, average across all tokens, focusing on how each token attends to every other token
    token_avg_attention = head_avg_attention.mean(dim=0).detach().numpy()  # dim=2 for receiving tokens dimension

    # Normalize for visualization
    normalized_attention = token_avg_attention / token_avg_attention.max()

    # Ensure the attention map is 2D before resizing
    if normalized_attention.ndim == 1:
        normalized_attention = normalized_attention.reshape(1, -1)  # Reshape to 2D if necessary

    # Resize the attention heatmap to match the image size
    try:
        attention_map_resized = zoom(normalized_attention, (image.size[1] / normalized_attention.shape[0], image.size[0] / normalized_attention.shape[1]), order=1)
    except Exception as e:
        print(f"Error resizing attention map in layer {i+1}: {str(e)}")
        continue  # Skip this layer on error

    # Plotting
    axs[i].imshow(image)
    axs[i].imshow(attention_map_resized, cmap='hot', alpha=0.9)
    axs[i].set_title(f"Layer {i+1} Attention")
    axs[i].axis("off")

plt.tight_layout()
plt.show()
