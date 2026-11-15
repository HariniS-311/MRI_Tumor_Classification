
import os
import random
import cv2
import matplotlib.pyplot as plt

from preprocessing import (
    DATASET_PATH,
    CLASSES,
    IMAGE_EXTENSIONS,
    preprocess_image
)

random.seed(42)

fig, axes = plt.subplots(8, 5, figsize=(15, 20))

for class_index, class_name in enumerate(CLASSES):

    for dataset_index, dataset_type in enumerate(["Training", "Testing"]):

        folder = os.path.join(
            DATASET_PATH,
            dataset_type,
            class_name
        )

        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]

        selected_files = random.sample(
            files,
            min(5, len(files))
        )

        row = class_index * 2 + dataset_index

        for column, filename in enumerate(selected_files):

            image_path = os.path.join(folder, filename)
            image = preprocess_image(image_path)

            axes[row, column].imshow(image, cmap="gray")
            axes[row, column].axis("off")

            if column == 0:
                axes[row, column].set_ylabel(
                    f"{class_name}\n{dataset_type}",
                    fontsize=10
                )

plt.suptitle(
    "MRI Dataset Comparison - Training vs Testing",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "dataset_comparison.png",
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nImage comparison saved as dataset_comparison.png")
