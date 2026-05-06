#Homework 3 Part 1
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder, StandardScaler

dataset = fetch_ucirepo(id=419) 

x = dataset.data.features
y = dataset.data.targets

df = pd.concat([x, y], axis=1)
df = df.replace('?', pd.NA).dropna()

irrelevant_cols = ['age_desc', 'used_app_before', 'relation']
df = df.drop(columns=[c for c in irrelevant_cols if c in df.columns])

target_col = [c for c in df.columns if 'class' in c.lower() or 'asd' in c.lower()][-1]

le = LabelEncoder()
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

scaler = StandardScaler()
numerical_cols = ['age', 'result']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

x_final = df.drop(columns=[target_col])
y_final = df[target_col]

print(f"Success! Target column identified as: {target_col}")
print(f"Preprocessed dataset size: {x_final.shape}")
print(x_final.head())