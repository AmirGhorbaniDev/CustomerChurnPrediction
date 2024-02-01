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

# Step 4: Train models and evaluate
def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC(probability=True),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
 # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'classification_report': report
        }

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(conf_matrix, name)

        # ROC Curve
        if y_prob is not None:
            plot_roc_curve(y_test, y_prob, name)

    return results

# Step 5: Plot confusion matrix
def plot_confusion_matrix(matrix, model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"charts/confusion_matrix_{model_name}.png")
    plt.show()

# Step 6: Plot ROC Curve
def plot_roc_curve(y_test, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(f"ROC Curve: {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(f"charts/roc_curve_{model_name}.png")
    plt.show()
    