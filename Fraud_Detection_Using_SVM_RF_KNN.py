import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Generate synthetic dataset
def generate_synthetic_data(n_samples=5000):
    np.random.seed(42)
    data = {
        'claim_amount': np.random.normal(1000, 500, n_samples),
        'num_procedures': np.random.randint(1, 10, n_samples),
        'num_physicians': np.random.randint(1, 5, n_samples),
        'patient_age': np.random.randint(18, 90, n_samples),
        'is_fraud': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    }
    return pd.DataFrame(data)

# Load synthetic data
df = generate_synthetic_data()

# Data visualization
sns.pairplot(df, hue='is_fraud')
plt.show()

# Prepare dataset for model training
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)

# Train SVM model
svm_clf = SVC(kernel='linear', probability=True, random_state=42)
svm_clf.fit(X_train, y_train)
y_pred_svm = svm_clf.predict(X_test)

# Train K-Nearest Neighbors (KNN) model
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)
y_pred_knn = knn_clf.predict(X_test)

# Model evaluation
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Model Evaluation:")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

evaluate_model("Random Forest", y_test, y_pred_rf)
evaluate_model("SVM", y_test, y_pred_svm)
evaluate_model("KNN", y_test, y_pred_knn)
