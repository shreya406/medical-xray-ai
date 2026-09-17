import os
from PIL import Image

image_folder = "dataset/images"

files = os.listdir(image_folder)

total = 0
corrupted = 0

for file in files:

    if file.lower().endswith((".png", ".jpg", ".jpeg")):

        total += 1

        path = os.path.join(
            image_folder,
            file
        )

        try:
            image = Image.open(path)
            image.verify()

        except Exception:
            corrupted += 1
            print("CORRUPTED:", file)

print("\nTotal images checked:", total)
print("Corrupted images:", corrupted)

if corrupted == 0:
    print("ALL IMAGES ARE VALID!")
else:
    print("Some images need to be downloaded again.")