
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import load_dataset, CLASSES

print("\nNeuroScan AI - SVM Hyperparameter Tuning")

# Step 1: Load training dataset only
print("\nLoading training dataset...")
X, y = load_dataset("Training")

# Step 2: Split training data into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))

# Step 3: Create SVM pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(class_weight="balanced"))
])

# Step 4: Define parameters to test
param_grid = {
    "svm__kernel": ["rbf"],
    "svm__C": [1, 10, 100],
    "svm__gamma": ["scale", 0.0001, 0.001]
}

# Step 5: Perform grid search
print("\nStarting hyperparameter tuning...")

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="f1_macro",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

# Step 6: Display best parameters
print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Macro F1:")
print(grid_search.best_score_)

# Step 7: Validate the best model
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_val)

print("\nValidation Accuracy:")
print(f"{accuracy_score(y_val, y_pred) * 100:.2f}%")

print("\nValidation Classification Report:")
print(classification_report(
    y_val,
    y_pred,
    labels=list(range(len(CLASSES))),
    target_names=CLASSES,
    zero_division=0
))

# Step 8: Save tuned model
joblib.dump(best_model, "models/svm_tuned_model.pkl")

print("\nTuned model saved successfully!")
