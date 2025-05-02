import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Record the start time
start_time = time.time()

# Load the dataset
data_path = 'modified data/feature_engineering_train.csv'
df = pd.read_csv(data_path)

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target
X = df.drop(['TARGET', 'SK_ID_CURR'], axis=1)
y = df['TARGET']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Drop rows with missing values
X_train = X_train.dropna()
y_train = y_train[X_train.index]  # Align labels with remaining rows

X_test = X_test.dropna()
y_test = y_test[X_test.index]

# Train Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=None,        # No maximum depth
    min_samples_split=2,   # Minimum samples to split a node
    min_samples_leaf=1,    # Minimum samples per leaf
    random_state=42,
    n_jobs=-1              # Use all CPU cores
)

rf_model.fit(X_train, y_train)

# Make predictions
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class
y_pred = (y_pred_proba > 0.5).astype(int)           # Convert probabilities to binary predictions

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Display top 15 features by importance
feature_importances = rf_model.feature_importances_

# Normalize feature importances
total_importance = feature_importances.sum()
normalized_importances = feature_importances / total_importance

# Create a DataFrame with normalized importance values
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Normalized Importance': normalized_importances
}).sort_values(by='Normalized Importance', ascending=False).head(15)

print("\nTop 15 most influential features:")
print(importance_df)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the result
print(f"Execution time: {elapsed_time:.2f} seconds")
