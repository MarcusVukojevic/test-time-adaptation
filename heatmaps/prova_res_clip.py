import clip
from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image
from scipy.ndimage import zoom

# Load the model
model, preprocess = clip.load("RN50")

# Prepare your image and text
image_path = "dogghi/doggo2.jpeg"
image = Image.open(image_path)
image = preprocess(image).unsqueeze(0)  # Apply preprocessing and add batch dimension

# Prepare text
text = "a photo of a dog"  # Adjust the description to your image
text = clip.tokenize([text])  # Tokenize text

# Pass inputs to the model and calculate logits and attention (if available)
with torch.no_grad():
    logits_per_image, logits_per_text = model(image, text)
    # Attentions may not be directly available, this depends on the version of CLIP being used

# If you need to visualize attentions, ensure your model variant supports it or manually implement a way to extract them
# For now, let's assume we're just displaying the image
fig, ax = plt.subplots()
ax.imshow(to_pil_image(image.squeeze(0)))  # Display the image
ax.set_title("Processed Image")
ax.axis('off')
plt.show()
