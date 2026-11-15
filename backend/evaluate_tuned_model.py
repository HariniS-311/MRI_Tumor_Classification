
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from preprocessing import load_dataset, CLASSES

print("\nNeuroScan AI - Tuned SVM Evaluation")

# Step 1: Load tuned model
print("\nLoading tuned SVM model...")
model = joblib.load("models/svm_tuned_model.pkl")

# Step 2: Load separate testing dataset
print("\nLoading testing dataset...")
X_test, y_test = load_dataset("Testing")

# Step 3: Predict
print("\nPredicting test images...")
y_pred = model.predict(X_test)

# Step 4: Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n========== TUNED MODEL TEST RESULTS ==========")
print(f"Testing Accuracy: {accuracy * 100:.2f}%")

# Step 5: Classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=list(range(len(CLASSES))),
    target_names=CLASSES,
    zero_division=0
))

# Step 6: Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nEvaluation completed!")
