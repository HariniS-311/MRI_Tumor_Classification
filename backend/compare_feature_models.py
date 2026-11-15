
import os
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import load_dataset, CLASSES

print("\nNeuroScan AI - Feature Model Comparison")

print("\nLoading training dataset...")
X_train, y_train = load_dataset("Training")

print("\nLoading testing dataset...")
X_test, y_test = load_dataset("Testing")

# HOG features = first 8100 columns
X_train_hog = X_train[:, :8100]
X_test_hog = X_test[:, :8100]

# Combined HOG + GLCM features
X_train_combined = X_train
X_test_combined = X_test

models = {
    "HOG Only": (X_train_hog, X_test_hog),
    "HOG + GLCM": (X_train_combined, X_test_combined)
}

for model_name, (train_features, test_features) in models.items():

    print("\n================================")
    print("Model:", model_name)
    print("================================")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            class_weight="balanced"
        ))
    ])

    print("Training model...")
    model.fit(train_features, y_train)

    print("Testing model...")
    predictions = model.predict(test_features)

    accuracy = accuracy_score(y_test, predictions)

    print("\nTesting Accuracy:", round(accuracy * 100, 2), "%")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0
    ))

    os.makedirs("models", exist_ok=True)

    filename = (
        "models/svm_hog_only.pkl"
        if model_name == "HOG Only"
        else "models/svm_hog_glcm.pkl"
    )

    joblib.dump(model, filename)
    print("Model saved:", filename)

print("\nFeature comparison completed!")
