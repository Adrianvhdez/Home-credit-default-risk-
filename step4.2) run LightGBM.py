import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import accuracy_score, roc_auc_score

# Record the start time
start_time = time.time()

# Load the dataset
data_path = 'modified data/feature_engineering_train.csv'
#data_path = 'modified data/missingremoved.csv'
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

# Train LightGBM model
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.1,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42
}

# Train without early stopping
lgb_model = lgb.train(
    params,
    lgb_train,
    num_boost_round=100,
    valid_sets=[lgb_train, lgb_test],
    valid_names=['train', 'test'],
)

# Make predictions
y_pred_proba = lgb_model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

# Display top 15 features by importance
feature_importances = lgb_model.feature_importance()

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