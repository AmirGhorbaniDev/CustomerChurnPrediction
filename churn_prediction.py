import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Step 1: Generate synthetic dataset
def generate_data():
    np.random.seed(42)
    num_customers = 2000
    data = {
        'customer_id': np.arange(1, num_customers + 1),
        'age': np.random.randint(18, 80, num_customers),
        'monthly_spend': np.random.uniform(20, 300, num_customers),
        'contract_type': np.random.choice(['Month-to-Month', 'One-Year', 'Two-Year'], num_customers),
        'support_calls': np.random.poisson(2, num_customers),
        'tenure': np.random.randint(1, 72, num_customers),
        'churn': np.random.choice([0, 1], num_customers, p=[0.8, 0.2])
    }
    df = pd.DataFrame(data)
    return df

# Step 2: Exploratory Data Analysis (EDA)
def eda(df):
    print("Dataset Summary:")
    print(df.describe())
    print("\nNull Values:")
    print(df.isnull().sum())
    churn_rate = df['churn'].value_counts(normalize=True)
    print("\nChurn Rate:")
    print(churn_rate)


    # Plot Churn Distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='churn', palette='viridis')
    plt.title("Churn Distribution")
    plt.savefig("charts/churn_distribution.png")
    plt.show()

# Step 3: Preprocess data
def preprocess_data(df):
    df = pd.get_dummies(df, columns=['contract_type'], drop_first=True)
    X = df.drop(columns=['customer_id', 'churn'])
    y = df['churn']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
