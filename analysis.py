#https://medium.com/@pythonshield/machine-learning-in-python-building-your-first-predictive-model-757a67cd5cd8

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

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
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)


smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts(normalize=True))

scaler = StandardScaler()
x_train_resampled = scaler.fit_transform(x_train_resampled)
x_test = scaler.transform(x_test)

x_train_df = pd.DataFrame(x_train_resampled, columns=x.columns)
x_test_df = pd.DataFrame(x_test, columns=x.columns)

x_train_df['Glucose_BMI'] = x_train_df['Glucose'] * x_train_df['BMI']
x_test_df['Glucose_BMI'] = x_test_df['Glucose'] * x_test_df['BMI']


model = LogisticRegression(max_iter=500, solver='lbfgs')
rfe = RFE(estimator=model, n_features_to_select=5)
rfe.fit(x_train_df, y_train_resampled)
selected_features = x_train_df.columns[rfe.support_].tolist()
print("Selected features:", selected_features)

x_train_selected = rfe.transform(X_train_df)
x_test_selected = rfe.transform(X_test_df)

#Logistic Regression
lr_model = LogisticRegression(max_iter=500, solver='lbfgs')
lr_scores = cross_val_score(lr_model, x_train_selected, y_train_resampled, cv=5, scoring='roc_auc')
print(f'Logistic Regression CV AUV:{lr_scores.mean():.2f} + {lr_scores.std():.2f}' )

#Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_scores = cross_val_score(rf_model, x_train_selected, y_train_resampled, cv=5, scoring='roc_auc' )
print(f'Random Forest CV AUC: {rf_scores.mean():.2f} ± {rf_scores.std():.2f}')

# Support Vector Machine
svm_model = SVC(probability=True)
svm_scores = cross_val_score(svm_model, x_train_selected, y_train_resampled, cv=5, scoring='roc_auc')
print(f'SVM CV AUC: {svm_scores.mean():.2f} ± {svm_scores.std():.2f}')


rf_model.fit(x_train_selected, y_train_resampled)
