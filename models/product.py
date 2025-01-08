import pandas as pd
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore

def product_recommendation(data):
    # Preprocess data for product recommendation
    customer_product_df = data.pivot_table(index='Email', columns='Product name', values='Total', aggfunc='sum').fillna(0)

    # Neural Network Model for Product Recommendations
    input_dim = customer_product_df.shape[1]

    # Build a simple feed-forward Neural Network
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dense(32, activation='relu'),
        Dense(input_dim, activation='linear')  # Output layer to recommend products
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train on the entire dataset (no train/test split for recommendations)
    model.fit(customer_product_df, customer_product_df, epochs=10, batch_size=16, verbose=1)

    # Recommend products for a customer (e.g., customer 0)
    customer_index = 0  # Example customer index
    customer_vector = customer_product_df.iloc[customer_index].values.reshape(1, -1)
    predicted_vector = model.predict(customer_vector)

    # Get top 5 recommended products
    recommended_indices = predicted_vector.argsort()[0][::-1][:5]
    top_products = customer_product_df.columns[recommended_indices]
    
    return top_products.tolist(), customer_index
