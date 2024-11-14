
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split


matplotlib.use('TkAgg')


def data_quality(df):
    
    print("Missing values:\n", df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nData types:\n", df.dtypes)

    print("\νSummary statistics:\n", df.describe())

    z_scores = stats.zscore(df.select_dtypes(include=[np.number]))
    outliers = (np.abs(z_scores) > 3).sum(axis=0)
    print("\nNumber of outliers (Z-score > 3):\n", outliers)

    if 'Outcome' in df.columns:
        class_balance = df['Outcome'].value_counts(normalize=True)
        print("\nClass balance:\n", class_balance)

        plt.figure(figsize=(8,6))
        sns.countplot(x='Outcome', data=df)
        plt.title('Class Distribution')
        plt.show()

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    'Pregancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age', 'Outcome'
]

data = pd.read_csv(url, names=columns)

data_quality(data)

cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[cols_with_zero] = data[cols_with_zero].replace(0, np.nan)
data[cols_with_zero] = data[cols_with_zero].fillna(data[cols_with_zero].median())

# Verify missing values after imputation
print(data.isnull().sum())

plt.figure(figsize=(12, 6))
data.boxplot(column=cols_with_zero)
plt.title('Box Plots for Numerical Features')
plt.xticks(rotation=45)
plt.show()

numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
numeric_cols.remove('Outcome')

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    sns.histplot(data[col], kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {col}')

plt.tight_layout()
plt.show()

#HeatMap Correlation
plt.figure(figsize=(12,10))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm',linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


#Pair plot
sns.pairplot(data, hue='Outcome', diag_kind='kde')
plt.suptitle('Pair Plot of Features by Outcome', y=1.02)
plt.show()

#Splitting the data
x = data.drop('Outcome', axis=1)
y = data['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
