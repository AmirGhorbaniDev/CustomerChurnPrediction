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
 