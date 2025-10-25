import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loading
file_path = "global_freelancers_raw.csv"
fd = pd.read_csv(file_path)
print(fd.info())
print(fd.describe())
print(fd.isnull().mean()*100)
#CLEANING
if 'gender' in fd.columns:
    fd['gender'] = fd['gender'].astype(str).str.lower().str.strip()
    fd['gender'] = fd['gender'].replace({'f': 'Female', 'female': 'Female', 'm': 'Male', 'male': 'Male'})

if 'hourly_rate (USD)' in fd.columns:
    fd['hourly_rate (USD)'] = fd['hourly_rate (USD)'].astype(str).str.replace('USD', '', regex=False)
    fd['hourly_rate (USD)'] = fd['hourly_rate (USD)'].str.replace('$', '', regex=False).str.strip()
    fd['hourly_rate (USD)'] = pd.to_numeric(fd['hourly_rate (USD)'], errors='coerce')

if 'is_active' in fd.columns:
    fd['is_active'] = fd['is_active'].astype(str).str.lower().str.strip()
    yes = ['1', 'true', 'yes', 'y']
    no = ['0', 'false', 'no', 'n']
    fd['is_active'] = fd['is_active'].apply(lambda x: 'YES' if x in yes else ('NO' if x in no else np.nan))

if 'client_satisfaction' in fd.columns:
    fd['client_satisfaction'] = fd['client_satisfaction'].astype(str).str.replace('%', '', regex=False)
    fd['client_satisfaction'] = pd.to_numeric(fd['client_satisfaction'], errors='coerce')

#Imputing
for col in ['age', 'years_of_experience', 'hourly_rate (USD)', 'rating', 'client_satisfaction']:
    if col in fd.columns:
        fd[col].fillna(fd[col].median(), inplace=True)

if 'is_active' in fd.columns:
    fd['is_active'].fillna(fd['is_active'].mode()[0], inplace=True)

#checking outlier
num_cols = ['age', 'years_of_experience', 'hourly_rate (USD)', 'rating', 'client_satisfaction']
for col in num_cols:
    if col in fd.columns:
        Q1 = fd[col].quantile(0.25)
        Q3 = fd[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        fd[col] = np.where(fd[col] < lower, lower, fd[col])
        fd[col] = np.where(fd[col] > upper, upper, fd[col])

#correlation & summary
summary = fd.describe(include='all')
corr_matrix = fd[num_cols].corr()

#visualisation
sns.set(style="whitegrid", palette="coolwarm")

for col in num_cols:
    if col in fd.columns:
        plt.figure(figsize=(7,4))
        sns.histplot(fd[col], kde=True, bins=30)
        plt.title(f"Distribution of {col}")
        plt.show()

if 'gender' in fd.columns:
    plt.figure(figsize=(5,4))
    fd['gender'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=["#2f98fa", "#e2a666"])
    plt.title("Gender Distribution")
    plt.ylabel("")
    plt.show()

if 'is_active' in fd.columns:
    plt.figure(figsize=(5,4))
    sns.countplot(x='is_active', data=fd)
    plt.title("Freelancer Activity Status")
    plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(fd[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

if 'years_of_experience' in fd.columns and 'hourly_rate (USD)' in fd.columns:
    plt.figure(figsize=(7,5))
    sns.scatterplot(data=fd, x='years_of_experience', y='hourly_rate (USD)', hue='gender', alpha=0.7)
    plt.title("Experience vs Hourly Rate by Gender")
    plt.show()

if 'rating' in fd.columns and 'gender' in fd.columns:
    plt.figure(figsize=(7,5))
    sns.boxplot(x='gender', y='rating', data=fd)
    plt.title("Rating Distribution by Gender")
    plt.show()

print(fd.isnull().mean()*100)
print(fd.info())
fd.to_csv("global_freelancers_cleaned.csv", index=False)

