
import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from preprocessing import load_dataset, CLASSES

print("\nNeuroScan AI - SVM Model Training")

# Step 1: Load training data
print("\nLoading training data...")
X_train, y_train = load_dataset("Training")

# Step 2: Load testing data
print("\nLoading testing data...")
X_test, y_test = load_dataset("Testing")

# Step 3: Create SVM pipeline
print("\nCreating SVM classifier...")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced"
    ))
])

# Step 4: Train the model
print("\nTraining SVM model...")
model.fit(X_train, y_train)

print("Training completed!")

# Step 5: Make predictions
print("\nPredicting testing data...")
y_pred = model.predict(X_test)

# Step 6: Evaluate model
accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL RESULTS ==========")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 7: Save trained model
os.makedirs("models", exist_ok=True)

model_path = os.path.join("models", "svm_model.pkl")
joblib.dump(model, model_path)

print(f"\nModel saved successfully at: {model_path}")
print("\nNeuroScan AI - Training Finished!")
