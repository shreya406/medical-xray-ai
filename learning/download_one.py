from urllib.request import urlopen, Request
import os

images = [
    "00003528_015.png",
    "00005089_012.png",
    "00021338_000.png",
    "00024116_001.png",
    "00026926_000.png",
    "00027415_073.png",
    "00028499_010.png"
]

os.makedirs("dataset/images", exist_ok=True)

for image_name in images:

    url = (
        "https://nih-chest-x-rays.s3.us-east-2.amazonaws.com/"
        "images_224x224/"
        + image_name
    )

    save_path = "dataset/images/" + image_name

    try:
        request = Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        with urlopen(request, timeout=30) as response:
            with open(save_path, "wb") as file:
                file.write(response.read())

        print("Downloaded:", image_name)

    except Exception as e:
        print("FAILED:", image_name)
        print("Error:", e)

print("\nFinished downloading the 7 images.")