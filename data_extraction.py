import pandas as pd
import numpy as np

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Processing Dataset....")

## Get all the columns that are Binary Text Answers -> YES or NO
binary_q_col = []
for column in df.columns:
  if (type(df.head(1)[column].item()) == str) and (df.head(1)[column].item().lower() == "yes" or df.head(1)[column].item().lower() == "no"):
    binary_q_col.append(column)

# ## GO through each column and convert a Binary Text Answer into Number -> YES = 1, NO = 0

choices = [1,0]

for column in binary_q_col:
  conditions = [
    df[column] == "Yes",
    df[column] == "No"
  ]
  df[column] = np.select(conditions, choices, default=None)

df =  df.drop('customerID', axis = 1)

## Convert the price from str -> float64
df['TotalCharges'] = df['TotalCharges'].replace(' ', 0)
df['TotalCharges'] = df['TotalCharges'].astype(float)

## Convert Gender column into binary
df['gender'] = np.select([df['gender'] == "Female", df['gender'] == "Male"], [0,1], default=None)

distinct_columns = {}

for column in df.columns:
  if type(df.head(1)[column].item()) == str and column not in binary_q_col:
    distinct_columns[column] = df[column].unique().tolist()

df = pd.get_dummies(df, columns=list(distinct_columns.keys()), drop_first=True, dtype=int)

## Replace all the NaN values with 0
df.fillna(0, inplace=True)

df.to_csv("./Telo_Customer_Dataset_LR.csv", index=False)
print("Extracted Data Set is Saved to Telo_Customer_Dataset_LR.csv")