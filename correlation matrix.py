import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Record the start time
start_time = time.time()

# Load the dataset
data_path = 'modified data/application_40deleted.csv'
# data_path = 'modified data/missingremoved.csv'
df = pd.read_csv(data_path)

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']):
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Plot the correlation matrix
print("\nGenerating the correlation matrix...")
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()  # Compute the correlation matrix
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, fmt=".2f")
plt.title("Correlation Matrix of Features")
plt.show()

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the result
print(f"Execution time: {elapsed_time:.2f} seconds")
