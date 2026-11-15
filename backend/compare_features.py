import numpy as np
from preprocessing import load_dataset, CLASSES

HOG_FEATURES = 8100
GLCM_FEATURES = 16

print("\nNeuroScan AI - Training vs Testing Feature Comparison")

X_train, y_train = load_dataset("Training")
X_test, y_test = load_dataset("Testing")

def analyse_features(X, y, dataset_name):
    print(f"\n========== {dataset_name} FEATURE STATISTICS ==========")

    for index, class_name in enumerate(CLASSES):
        class_features = X[y == index]

        hog = class_features[:, :HOG_FEATURES]
        glcm = class_features[:, HOG_FEATURES:]

        print(f"\n{class_name}")
        print("Images:", len(class_features))
        print("HOG mean:", round(float(np.mean(hog)), 5))
        print("HOG standard deviation:", round(float(np.std(hog)), 5))
        print("GLCM mean:", round(float(np.mean(glcm)), 5))
        print("GLCM standard deviation:", round(float(np.std(glcm)), 5))


analyse_features(X_train, y_train, "TRAINING")
analyse_features(X_test, y_test, "TESTING")

print("\nFeature comparison completed!")