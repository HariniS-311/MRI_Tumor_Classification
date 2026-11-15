
import os
import cv2
import numpy as np

# Dataset path
DATASET_PATH = r"E:\MRI Brain Tumor Classification\Dataset"

# Image configuration
IMAGE_SIZE = 128
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]

# Supported image formats
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def preprocess_image(image_path):
    """
    Read and preprocess an MRI image.
    """

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # Resize image
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    # Remove noise
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # Normalize pixel values
    image = cv2.normalize(
        image, None, 0, 255, cv2.NORM_MINMAX
    )

    return image


def extract_glcm_features(image):
    """
    Extract simple GLCM texture features.
    """

    quantized = (image // 16).astype(np.uint8)
    features = []

    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dy, dx in directions:

        shifted = np.roll(quantized, shift=(-dy, -dx), axis=(0, 1))

        valid = np.ones(quantized.shape, dtype=bool)

        if dy > 0:
            valid[-dy:, :] = False
        if dx > 0:
            valid[:, -dx:] = False
        if dx < 0:
            valid[:, :-dx] = False

        a = quantized[valid]
        b = shifted[valid]

        glcm = np.zeros((16, 16), dtype=np.float64)
        np.add.at(glcm, (a, b), 1)

        total = glcm.sum()

        if total == 0:
            features.extend([0, 0, 0, 0])
            continue

        glcm /= total

        i, j = np.indices((16, 16))

        contrast = np.sum(glcm * (i - j) ** 2)
        homogeneity = np.sum(glcm / (1 + (i - j) ** 2))
        energy = np.sum(glcm ** 2)

        mean_i = np.sum(i * glcm)
        mean_j = np.sum(j * glcm)

        std_i = np.sqrt(np.sum((i - mean_i) ** 2 * glcm))
        std_j = np.sqrt(np.sum((j - mean_j) ** 2 * glcm))

        correlation = np.sum(
            (i - mean_i) * (j - mean_j) * glcm
        ) / (std_i * std_j + 1e-10)

        features.extend([
            contrast,
            homogeneity,
            energy,
            correlation
        ])

    return np.array(features, dtype=np.float32)


def extract_features(image):
    """
    Extract HOG and GLCM features.
    """

    hog = cv2.HOGDescriptor(
        (IMAGE_SIZE, IMAGE_SIZE),
        (16, 16),
        (8, 8),
        (8, 8),
        9
    )

    hog_features = hog.compute(image).flatten()

    glcm_features = extract_glcm_features(image)

    combined_features = np.concatenate([
        hog_features,
        glcm_features
    ])

    return combined_features


def load_dataset(dataset_type):
    """
    Load images and extract features from
    Training or Testing folders.
    """

    if dataset_type not in ["Training", "Testing"]:
        raise ValueError("Use Training or Testing")

    folder_path = os.path.join(
        DATASET_PATH, dataset_type
    )

    features = []
    labels = []

    print(f"\nLoading {dataset_type} dataset...")

    for label, class_name in enumerate(CLASSES):

        class_path = os.path.join(
            folder_path, class_name
        )

        if not os.path.isdir(class_path):
            raise FileNotFoundError(
                f"Missing category folder: {class_path}"
            )

        count = 0

        for filename in sorted(os.listdir(class_path)):

            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue

            image_path = os.path.join(
                class_path, filename
            )

            try:
                image = preprocess_image(image_path)
                feature_vector = extract_features(image)

                features.append(feature_vector)
                labels.append(label)

                count += 1

            except Exception as error:
                print(f"Skipped {filename}: {error}")

        print(f"{class_name}: {count} images processed")

    return np.array(features), np.array(labels)


if __name__ == "__main__":

    print("NeuroScan AI - Preprocessing Module")

    X_train, y_train = load_dataset("Training")

    print("\nTraining dataset loaded successfully!")
    print("Feature shape:", X_train.shape)
    print("Label shape:", y_train.shape)

    print("\nClass mapping:")

    for index, name in enumerate(CLASSES):
        print(index, "=", name)
