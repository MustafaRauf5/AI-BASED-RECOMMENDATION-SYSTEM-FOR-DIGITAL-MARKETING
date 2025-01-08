import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def sales_prediction(data):
    # Preprocess data for sales prediction
    data['Created at'] = pd.to_datetime(data['Created at'], errors='coerce')

    # Aggregate total sales per day
    sales_df = data.groupby(data['Created at'].dt.date).agg({
        'Total': 'sum',
        'Product price': 'mean'
    }).reset_index()

    sales_df.columns = ['Date', 'Total_Sales', 'Average_Product_Price']

    # Prepare features and labels
    X = sales_df[['Average_Product_Price']]
    y = sales_df['Total_Sales']

    # Split the dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train an XGBoost Regressor
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)

    # Plot actual vs predicted sales
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual Sales', color='blue')
    plt.plot(y_pred, label='Predicted Sales', linestyle='--', color='red')
    plt.title('Actual vs Predicted Sales')
    plt.legend()
    plt.show()

    return mae
