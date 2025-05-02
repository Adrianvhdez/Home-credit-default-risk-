import pandas as pd

# Define a function to process and aggregate data by SK_ID_CURR
def process_and_merge(main_df, additional_df, key_column="SK_ID_CURR", suffix="_additional"):
    # Ensure key_column is in the dataframe
    if key_column not in additional_df.columns:
        if additional_df.index.name == key_column:
            additional_df = additional_df.reset_index()
        else:
            raise KeyError(f"Key column '{key_column}' not found in the dataset.")

    # Split into numeric and non-numeric columns
    numeric_cols = additional_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = additional_df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Remove key_column from numeric and categorical columns (if present)
    numeric_cols = [col for col in numeric_cols if col != key_column]
    categorical_cols = [col for col in categorical_cols if col != key_column]

    # Aggregate numeric columns (mean)
    numeric_agg = additional_df.groupby(key_column)[numeric_cols].mean().reset_index()

    # Aggregate categorical columns (mode)
    if categorical_cols:
        mode_agg = (
            additional_df[categorical_cols + [key_column]]
            .groupby(key_column)
            .agg(lambda x: x.mode()[0] if not x.mode().empty else None)
            .reset_index()
        )
    else:
        mode_agg = pd.DataFrame()

    # Merge back to the main dataset with unique suffixes
    merged_df = main_df.merge(
        numeric_agg, on=key_column, how="left", suffixes=("", suffix)
    )
    if not mode_agg.empty:
        merged_df = merged_df.merge(
            mode_agg, on=key_column, how="left", suffixes=("", suffix)
        )

    return merged_df

# Load main datasets
application_train = pd.read_csv("original data/application_train.csv")
application_test = pd.read_csv("original data/application_test.csv")

# Load additional datasets
datasets = {
    "previous_application": "original data/previous_application.csv",
    "bureau": "original data/bureau.csv",
    "bureau_balance": "original data/bureau_balance.csv",
    "credit_card_balance": "original data/credit_card_balance.csv",
    "installments_payments": "original data/installments_payments.csv",
    "POS_CASH_balance": "original data/POS_CASH_balance.csv",
}

# Process and merge each dataset
for name, path in datasets.items():
    print(f"Processing {name}...")
    data = pd.read_csv(path)

    # Debugging step: ensure SK_ID_CURR exists in the additional dataset
    if "SK_ID_CURR" not in data.columns:
        print(f"Error: SK_ID_CURR is missing in {name}. Please check the dataset.")
        continue

    application_train = process_and_merge(application_train, data, suffix=f"_{name}")
    application_test = process_and_merge(application_test, data, suffix=f"_{name}")

# Save the merged datasets
application_train.to_csv("application_train_merged.csv", index=False)
application_test.to_csv("application_test_merged.csv", index=False)

print("Datasets merged successfully!")
