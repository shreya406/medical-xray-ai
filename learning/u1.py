from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 1. Open image
img = Image.open("learning/img1.jpeg")

# 2. Resize image
resized = img.resize((224, 224))

print("Original image size:", img.size)
print("Resized image size:", resized.size)

# 3. Convert to NumPy array
image_array = np.array(resized)

print("Array shape:", image_array.shape)

# 4. Convert to grayscale
gray = np.mean(image_array, axis=2)

print("Grayscale shape:", gray.shape)

# 5. Normalize pixels from 0-255 to 0-1
normalized = gray / 255.0

print("Normalized shape:", normalized.shape)
print("Minimum pixel:", normalized.min())
print("Maximum pixel:", normalized.max())

# 6. Display grayscale image
plt.imshow(normalized, cmap="gray")
plt.title("Grayscale X-ray-style Image")
plt.axis("off")
plt.show()