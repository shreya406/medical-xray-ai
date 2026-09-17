import pandas as pd
import os
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# STEP 1: READ BALANCED DATASET
# ============================================================

data = pd.read_csv("dataset/balanced_data.csv")

print("========================================")
print("DATASET INFORMATION")
print("========================================")

print("Total images:", len(data))
print("Total patients:", data["Patient ID"].nunique())


# ============================================================
# STEP 2: PATIENT-LEVEL TRAIN/VALIDATION/TEST SPLIT
# ============================================================

# First split:
# 70% training
# 30% temporary data

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=42
)

train_index, temp_index = next(
    splitter.split(
        data,
        groups=data["Patient ID"]
    )
)

train_data = data.iloc[train_index]
temp_data = data.iloc[temp_index]


# Second split:
# Split the remaining 30% equally
# 15% validation
# 15% testing

splitter2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=42
)

validation_index, test_index = next(
    splitter2.split(
        temp_data,
        groups=temp_data["Patient ID"]
    )
)

validation_data = temp_data.iloc[validation_index]
test_data = temp_data.iloc[test_index]


# ============================================================
# STEP 3: SAVE CSV FILES
# ============================================================

train_data.to_csv(
    "dataset/train.csv",
    index=False
)

validation_data.to_csv(
    "dataset/validation.csv",
    index=False
)

test_data.to_csv(
    "dataset/test.csv",
    index=False
)


# ============================================================
# STEP 4: SHOW SPLIT INFORMATION
# ============================================================

print("\n========================================")
print("TRAIN / VALIDATION / TEST")
print("========================================")

print("\nImages:")
print("Training:", len(train_data))
print("Validation:", len(validation_data))
print("Testing:", len(test_data))

print("\nPatients:")
print(
    "Training:",
    train_data["Patient ID"].nunique()
)

print(
    "Validation:",
    validation_data["Patient ID"].nunique()
)

print(
    "Testing:",
    test_data["Patient ID"].nunique()
)


# ============================================================
# STEP 5: CHECK DOWNLOADED IMAGES
# ============================================================

image_folder = "dataset/images"

print("\n========================================")
print("CHECKING DOWNLOADED X-RAYS")
print("========================================")


if not os.path.exists(image_folder):

    print("ERROR: dataset/images folder does not exist.")

else:

    downloaded_images = set(
        os.listdir(image_folder)
    )

    print(
        "Downloaded images:",
        len(downloaded_images)
    )


    # Get image names from all three CSV files

    train_images = set(
        train_data["Image Index"]
    )

    validation_images = set(
        validation_data["Image Index"]
    )

    test_images = set(
        test_data["Image Index"]
    )


    # Combine all required image names

    all_required_images = (
        train_images
        | validation_images
        | test_images
    )


    # Find missing images

    missing_images = (
        all_required_images
        - downloaded_images
    )


    print(
        "Images required:",
        len(all_required_images)
    )

    print(
        "Images missing:",
        len(missing_images)
    )


    # Show missing image names if there are any

    if len(missing_images) > 0:

        print("\nSome missing images:")

        for image in list(missing_images)[:20]:

            print(image)

        print(
            "\nOnly the first 20 missing images are shown."
        )

    else:

        print("\nALL IMAGES ARE PRESENT! ✅")


# ============================================================
# DONE
# ============================================================

print("\n========================================")
print("DATASET CHECK COMPLETED!")
print("========================================")