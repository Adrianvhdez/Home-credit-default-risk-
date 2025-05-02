import pandas as pd
import numpy as np

# Load the data
data_path = 'modified data/application_40deleted.csv'
df = pd.read_csv(data_path)
# installments
data_path_1 = 'original data/installments_payments.csv'
inst = pd.read_csv(data_path_1)
# credit card balance
data_path_2 = 'original data/credit_card_balance.csv'
cred_card_bal = pd.read_csv(data_path_2)
# bureau
data_path_3 = 'original data/bureau.csv'
bureau = pd.read_csv(data_path_3)
# appliaction
data_path_4 = 'original data/previous_application.csv'
prev_app = pd.read_csv(data_path_4)

#CREDIT_ANNUITY_RATIO
# Ensure there are no division by zero errors by handling missing or zero values in AMT_ANNUITY
df['CREDIT_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
# Optionally handle missing or infinite values
df['CREDIT_ANNUITY_RATIO'].replace([float('inf'), float('nan')], None, inplace=True)

# LATE_PAYMENT_FEATURE
inst_temp = inst.loc[inst.DAYS_ENTRY_PAYMENT >= -365].copy()  # Filter payments and create a copy
inst_temp['LATE_PAYMENT'] = inst_temp['DAYS_INSTALMENT'] - inst_temp['DAYS_ENTRY_PAYMENT']  # Calculate late payment days
late_payment_feature = inst_temp.groupby('SK_ID_CURR')[['LATE_PAYMENT']].min().reset_index()  # Aggregate by client and find the minimum late payment
# Merge late_payment_feature into the df dataset
df = df.merge(late_payment_feature, on='SK_ID_CURR', how='left')

# CREDIT_UTILIZATION
month = -2
cred_temp = cred_card_bal.loc[cred_card_bal.MONTHS_BALANCE >= month].copy()  # Filter and create a copy
cred_temp['CRED_UTIL'] = cred_temp['AMT_BALANCE'] / cred_temp['AMT_CREDIT_LIMIT_ACTUAL']  # Calculate credit utilization
# Replace infinite and NaN values in CRED_UTIL
cred_temp['CRED_UTIL'] = cred_temp['CRED_UTIL'].replace([float('inf'), -float('inf')], np.nan)
cred_temp['CRED_UTIL'] = cred_temp['CRED_UTIL'].fillna(cred_temp['CRED_UTIL'].median())
cred_util_feature = (
    cred_temp.groupby('SK_ID_CURR')['CRED_UTIL']
    .max()
    .reset_index()
    .rename(columns={'CRED_UTIL': 'CRED_UTIL_' + str(month * -1)})
)
# Merge credit utilization into the df dataset
df = df.merge(cred_util_feature, on='SK_ID_CURR', how='left')
# Replace missing values in the new credit utilization feature
df[f'CRED_UTIL_{month * -1}'] = df[f'CRED_UTIL_{month * -1}'].fillna(df[f'CRED_UTIL_{month * -1}'].median())

# DEBT_RATIO
bureau['DEBT_RATIO'] = bureau['AMT_CREDIT_SUM_DEBT'] / bureau['AMT_CREDIT_SUM']
# Replace infinite and NaN values in DEBT_RATIO
bureau['DEBT_RATIO'] = bureau['DEBT_RATIO'].replace([float('inf'), -float('inf')], np.nan)
bureau['DEBT_RATIO'] = bureau['DEBT_RATIO'].fillna(bureau['DEBT_RATIO'].median())
# Calculate the maximum DEBT_RATIO for each SK_ID_CURR
debt_ratio_feature = bureau.groupby('SK_ID_CURR')['DEBT_RATIO'].max().reset_index()
# Merge debt ratio feature into the df dataset
df = df.merge(debt_ratio_feature, on='SK_ID_CURR', how='left')
# Replace missing values in the merged DEBT_RATIO column
df['DEBT_RATIO'] = df['DEBT_RATIO'].fillna(df['DEBT_RATIO'].median())



# INTEREST RATE OF PREVIOUS APPLICATION
# Calculate interest-related features in the previous application dataset
#prev_app['INTEREST'] = prev_app['CNT_PAYMENT'] * prev_app['AMT_ANNUITY'] - prev_app['AMT_CREDIT']
#prev_app['INTEREST_RATE'] = 2 * 12 * prev_app['INTEREST'] / (prev_app['AMT_CREDIT'] * (prev_app['CNT_PAYMENT'] + 1))
#prev_app['INTEREST_SHARE'] = prev_app['INTEREST'] / prev_app['AMT_CREDIT']
# Aggregate the INTEREST_RATE by taking the average for each SK_ID_CURR
#interest_rate_feature = prev_app.groupby('SK_ID_CURR')['INTEREST_RATE'].mean().reset_index()
# Merge the aggregated INTEREST_RATE feature into the main dataset (df)
#df = df.merge(interest_rate_feature, on='SK_ID_CURR', how='left')
# Optionally, handle missing values (fill NaN with the median or a default value)
#df['INTEREST_RATE'] = df['INTEREST_RATE'].fillna(df['INTEREST_RATE'].median())

# amount of credit / goods price
# Create the new feature CREDIT_GOODS_RATIO
df['CREDIT_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
# Optionally, handle cases where 'GOODS_PRICE' might be zero or missing (avoid division by zero or infinite values)
df['CREDIT_GOODS_RATIO'].replace([float('inf'), float('nan')], None, inplace=True)
# Optionally, handle missing values by replacing NaNs with a default value (e.g., the median of the column)
df['CREDIT_GOODS_RATIO'].fillna(df['CREDIT_GOODS_RATIO'].median(), inplace=True)

# Save the filtered dataset to a new file
output_path = "modified data/feature_engineering_train.csv"
df.to_csv(output_path, index=False)


