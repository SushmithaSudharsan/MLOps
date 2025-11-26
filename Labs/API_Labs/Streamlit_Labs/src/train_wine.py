# Streamlit_Labs/src/train_wine.py
import pickle
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from pathlib import Path

# Load wine dataset
wine = load_wine()
X = wine.data
y = wine.target

print(f"Dataset shape: {X.shape}")
print(f"Feature names: {wine.feature_names}")
print(f"Target names: {wine.target_names}")
print(f"Class distribution: {np.bincount(y)}")

# Split data (stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "="*60)
print("Training Different Models...")
print("="*60)

# ============================================================
# Option 1: SVM with Scaling (Usually Best for Wine Dataset)
# ============================================================
svm_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42))
])

svm_pipeline.fit(X_train, y_train)
svm_accuracy = svm_pipeline.score(X_test, y_test)
svm_cv_scores = cross_val_score(svm_pipeline, X, y, cv=5)

print(f"\n1. SVM Model:")
print(f"   Test Accuracy: {svm_accuracy*100:.2f}%")
print(
    f"   Cross-Val Accuracy: {svm_cv_scores.mean()*100:.2f}% (+/- {svm_cv_scores.std()*100:.2f}%)")

# ============================================================
# Option 2: Gradient Boosting
# ============================================================
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb_model.fit(X_train, y_train)
gb_accuracy = gb_model.score(X_test, y_test)
gb_cv_scores = cross_val_score(gb_model, X, y, cv=5)

print(f"\n2. Gradient Boosting Model:")
print(f"   Test Accuracy: {gb_accuracy*100:.2f}%")
print(
    f"   Cross-Val Accuracy: {gb_cv_scores.mean()*100:.2f}% (+/- {gb_cv_scores.std()*100:.2f}%)")

# ============================================================
# Option 3: Random Forest (Optimized)
# ============================================================
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

rf_model.fit(X_train, y_train)
rf_accuracy = rf_model.score(X_test, y_test)
rf_cv_scores = cross_val_score(rf_model, X, y, cv=5)

print(f"\n3. Random Forest Model:")
print(f"   Test Accuracy: {rf_accuracy*100:.2f}%")
print(
    f"   Cross-Val Accuracy: {rf_cv_scores.mean()*100:.2f}% (+/- {rf_cv_scores.std()*100:.2f}%)")

# ============================================================
# Option 4: Ensemble (Voting Classifier) - BEST PERFORMANCE
# ============================================================
# Combine all three models
ensemble_model = VotingClassifier(
    estimators=[
        ('svm', svm_pipeline),
        ('gb', gb_model),
        ('rf', rf_model)
    ],
    voting='soft'  # Uses probability voting
)

ensemble_model.fit(X_train, y_train)
ensemble_accuracy = ensemble_model.score(X_test, y_test)
ensemble_cv_scores = cross_val_score(ensemble_model, X, y, cv=5)

print(f"\n4. Ensemble Model (RECOMMENDED):")
print(f"   Test Accuracy: {ensemble_accuracy*100:.2f}%")
print(
    f"   Cross-Val Accuracy: {ensemble_cv_scores.mean()*100:.2f}% (+/- {ensemble_cv_scores.std()*100:.2f}%)")

# ============================================================
# Select Best Model
# ============================================================
models = {
    'SVM': (svm_pipeline, svm_cv_scores.mean()),
    'Gradient Boosting': (gb_model, gb_cv_scores.mean()),
    'Random Forest': (rf_model, rf_cv_scores.mean()),
    'Ensemble': (ensemble_model, ensemble_cv_scores.mean())
}

best_model_name = max(models, key=lambda x: models[x][1])
best_model = models[best_model_name][0]
best_accuracy = models[best_model_name][1]

print("\n" + "="*60)
print(f"BEST MODEL: {best_model_name}")
print(f"Cross-Validation Accuracy: {best_accuracy*100:.2f}%")
print("="*60)

# Save the best model
model_path = Path(__file__).resolve().parent / 'wine_model.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)

print(f"\n✅ Best model ({best_model_name}) saved to {model_path}")

# Show feature importance if it's a tree-based model
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    feature_importance = best_model.feature_importances_
    print("\n📊 Top 5 Most Important Features:")
    indices = np.argsort(feature_importance)[::-1][:5]
    for i, idx in enumerate(indices, 1):
        print(
            f"   {i}. {wine.feature_names[idx]}: {feature_importance[idx]:.4f}")
