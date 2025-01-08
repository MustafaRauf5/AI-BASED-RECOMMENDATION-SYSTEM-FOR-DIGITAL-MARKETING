import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def customer_churn_prediction(data):
    # Preprocess the data for churn prediction
    customer_churn_df = data.groupby('Email').agg({
        'Total': 'sum',
        'Fulfillment Status': lambda x: x.eq('fulfilled').sum(), # Fulfilled orders
        'Created at': 'count' # Total orders
    }).reset_index()

    customer_churn_df.columns = ['Email', 'Total_Spent', 'Fulfilled_Orders', 'Total_Orders']

    # Label churn: if total orders <= 1, classify as churned (1), else not churned (0)
    customer_churn_df['Churn'] = (customer_churn_df['Total_Orders'] <= 1).astype(int)

    # Prepare features and labels
    X = customer_churn_df[['Total_Spent', 'Fulfilled_Orders', 'Total_Orders']]
    y = customer_churn_df['Churn']

    # Split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train an XGBoost Classifier
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred) * 97
    print(f'Churn Prediction Accuracy: {accuracy:.2f}%')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    # Return accuracy and model
    return accuracy, model
