
import os
import cv2
import numpy as np

from preprocessing import (
    DATASET_PATH,
    CLASSES,
    IMAGE_EXTENSIONS,
    preprocess_image
)

print("\nNeuroScan AI - Processed Image Analysis")

for dataset_type in ["Training", "Testing"]:

    print(f"\n========== {dataset_type} ==========")

    for class_name in CLASSES:

        folder = os.path.join(
            DATASET_PATH,
            dataset_type,
            class_name
        )

        means = []
        stds = []

        for filename in os.listdir(folder):

            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue

            path = os.path.join(folder, filename)

            try:
                image = preprocess_image(path)

                means.append(np.mean(image))
                stds.append(np.std(image))

            except Exception as error:
                print("Error:", filename, error)

        print(f"\n{class_name}")
        print("Processed images:", len(means))
        print("Mean brightness:", round(np.mean(means), 2))
        print("Mean contrast:", round(np.mean(stds), 2))

print("\nAnalysis completed!")
