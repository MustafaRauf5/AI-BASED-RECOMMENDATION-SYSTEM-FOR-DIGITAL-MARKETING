import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from models.churn import customer_churn_prediction
from models.product import product_recommendation
from models.sales import sales_prediction

# Load and clean the data
@st.cache_data
def load_data():
    data = pd.read_csv('data/orders_export_new_!.csv')  # Adjust the path as needed
    data['Created at'] = pd.to_datetime(data['Created at'], errors='coerce')
    data['Total'] = pd.to_numeric(data['Total'], errors='coerce')
    data['Product price'] = pd.to_numeric(data['Product price'], errors='coerce')
    return data

data = load_data()

# Set up page based on query params
query_params = st.query_params
current_page = query_params.get('page', 'dashboard')

# Custom CSS for a more attractive blue theme and gradient styling
st.markdown(
    """
    <style>
    body {
        background-color: #0b4046;
    }
    .main {
        background-color: #FFFFFF;
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .css-184tjsw {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 15px;
        border: 1px solid #ddd;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
    }
    .stMetric {
        padding: 20px;
        background-color: #E3F2FD;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
        font-size: 1.2rem;
        color: #1A237E;
    }
    .gradient-bg {
        background: linear-gradient(90deg, rgba(33,150,243,1) 0%, rgba(0,188,212,1) 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
st.sidebar.markdown("<h2 style='color:#1A237E;'>🧭 Model Selection</h2>", unsafe_allow_html=True)
model_selection = st.sidebar.radio("Select a Model:", ["Dashboard", "Customer Churn", "Product Recommendation", "Sales Prediction"])

# Change page based on the selected model
if model_selection == "Customer Churn":
    st.query_params = {"page": "customer_churn"}
    current_page = "customer_churn"
elif model_selection == "Product Recommendation":
    st.query_params = {"page": "product_recommendation"}
    current_page = "product_recommendation"
elif model_selection == "Sales Prediction":
    st.query_params = {"page": "sales_prediction"}
    current_page = "sales_prediction"
else:
    st.query_params = {"page": "dashboard"}
    current_page = "dashboard"

# Conditional Rendering Based on Current Page
if current_page == "dashboard":
    # Landing Page: Display the full dashboard with all analytics
    st.markdown("<h1 style='text-align: center; color: #1A237E;'>Sales Dashboard Overview</h1>", unsafe_allow_html=True)

    # Key Performance Indicators (KPIs)
    st.markdown("<h2 style='color: #0D47A1;'>Key Performance Indicators</h2>", unsafe_allow_html=True)
    total_orders = len(data)
    total_sales = data['Total'].sum()
    total_products_sold = data['Product price'].sum()
    average_order_value = total_sales / total_orders if total_orders > 0 else 0

    # Display KPIs side-by-side
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Orders", total_orders, delta=" +10%")
    col2.metric("Total Sales", f"${total_sales:,.2f}", delta=" +5%")
    col3.metric("Avg Order Value", f"${average_order_value:,.2f}")

    # Fulfillment Status and Sales Over Time side-by-side
    st.markdown("<h2 style='color: #0D47A1;'>Fulfillment Status & Sales Over Time</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='color: #0D47A1;'>Fulfillment Status Overview</h3>", unsafe_allow_html=True)
        fulfillment_counts = data['Fulfillment Status'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=fulfillment_counts.index, y=fulfillment_counts.values, ax=ax, palette="Blues_d")
        ax.set_xlabel("Status")
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with col2:
        st.markdown("<h3 style='color: #0D47A1;'>Sales Over Time</h3>", unsafe_allow_html=True)
        sales_over_time = data.groupby(data['Created at'].dt.date)['Total'].sum()
        fig, ax = plt.subplots(figsize=(8, 5))
        sales_over_time.plot(ax=ax, color='#42A5F5')
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Sales ($)")
        st.pyplot(fig)

    # Product Popularity and Vendor Performance side-by-side
    st.markdown("<h2 style='color: #0D47A1;'>Product Popularity & Vendor Performance</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='color: #0D47A1;'>Top Products Sold</h3>", unsafe_allow_html=True)
        top_products = data['Product name'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=top_products.values, y=top_products.index, ax=ax, palette="coolwarm")
        ax.set_xlabel("Quantity Sold")
        st.pyplot(fig)

    with col2:
        st.markdown("<h3 style='color: #0D47A1;'>Top Vendors by Sales</h3>", unsafe_allow_html=True)
        vendor_sales = data.groupby('Vendor')['Total'].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=vendor_sales.values, y=vendor_sales.index, ax=ax, palette="Blues")
        ax.set_xlabel("Total Sales ($)")
        st.pyplot(fig)

    # Geographic Distribution and Orders Per Day side-by-side
    st.markdown("<h2 style='color: #0D47A1;'>Geographic Distribution & Orders Per Day</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3 style='color: #0D47A1;'>Geographic Distribution of Orders</h3>", unsafe_allow_html=True)
        m = folium.Map(location=[30.3753, 69.3451], zoom_start=5)  
        city_counts = data['Shipping City'].value_counts().head(10)
        for city in city_counts.index:
            folium.Marker(
                location=[30.3753, 69.3451],  
                popup=f"{city}: {city_counts[city]} orders",
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(m)
        st_folium(m, width=500, height=300)

    with col2:
        st.markdown("<h3 style='color: #0D47A1;'>Orders Per Day</h3>", unsafe_allow_html=True)
        orders_per_day = data.groupby(data['Created at'].dt.date)['Total'].count()
        fig, ax = plt.subplots(figsize=(8, 5))
        orders_per_day.plot.bar(ax=ax, color='#4682B4')
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Orders")
        st.pyplot(fig)

    # Average Order Value
    st.markdown("<h2 style='color: #0D47A1;'>Average Order Value Over Time</h2>", unsafe_allow_html=True)
    average_order_value_per_day = data.groupby(data['Created at'].dt.date)['Total'].mean()

    fig, ax = plt.subplots(figsize=(10, 5))
    average_order_value_per_day.plot(ax=ax, color='#2E8B57')
    ax.set_xlabel("Date")
    ax.set_ylabel("Average Order Value ($)")
    st.pyplot(fig)

elif current_page == "customer_churn":
    # Customer Churn Prediction Page
    st.header("Customer Churn Prediction")
    if st.button('Run Churn Prediction'):
        accuracy, model = customer_churn_prediction(data)
        # st.write(f"Churn Prediction Accuracy: {accuracy:.2f}%")
        st.write("Confusion Matrix:")
        fig = plt.gcf()
        st.pyplot(fig)

elif current_page == "product_recommendation":
    # Product Recommendation Page
    st.header("Product Recommendation")
    if st.button('Run Product Recommendation'):
        top_products, customer_index = product_recommendation(data)
        st.write(f"Top 5 Recommended Products for Customer {customer_index}:")
        for product in top_products:
            st.write(f"- {product}")

elif current_page == "sales_prediction":
    # Sales Prediction Page
    st.header("Sales Prediction")
    if st.button('Run Sales Prediction'):
        mae = sales_prediction(data)
        # st.write(f"Sales Prediction Mean Absolute Error (MAE): {mae:.2f}")
        st.write("Actual vs Predicted Sales Chart:")
        fig = plt.gcf()
        st.pyplot(fig)
