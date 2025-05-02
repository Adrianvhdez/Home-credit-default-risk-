import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
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

# Train LightGBM model
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

# Define parameter grid for GridSearch
param_grid = {
    'num_leaves': [31, 50, 100],
    'max_depth': [-1, 10, 20],
    'learning_rate': [0.01, 0.05, 0.1],
}

# GridSearchCV
grid_search = GridSearchCV(
    estimator=lgb.LGBMClassifier(objective='binary', metric='auc', boosting_type='gbdt', seed=42),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,  # 5-fold cross-validation
    n_jobs=-1,
    verbose=1,
    refit=True
)

# Perform GridSearch
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f"Best Parameters: {best_params}")

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Best Accuracy: {accuracy:.4f}")
print(f"Best ROC AUC: {roc_auc:.4f}")

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the result
print(f"Execution time: {elapsed_time:.2f} seconds")