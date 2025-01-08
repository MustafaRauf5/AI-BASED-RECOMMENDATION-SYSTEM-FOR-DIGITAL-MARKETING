# Sales Dashboard Application

This is a Streamlit application that provides an interactive dashboard for analyzing sales data, predicting customer churn, recommending products, and forecasting future sales.

## Features

- **Sales Dashboard Overview**: Visualize key performance indicators, fulfillment status, sales over time, product popularity, vendor performance, geographic distribution of orders, and average order value.
- **Customer Churn Prediction**: Predict customer churn using an XGBoost classifier.
- **Product Recommendation**: Recommend products to customers based on a neural network model.
- **Sales Prediction**: Forecast future sales using an XGBoost regressor.

## Installation

To run this application, you need to have Python 3.6 or later installed. Follow these steps to set up the environment:

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Ensure you have your data file (`data/orders_export_new_!.csv`) in the `data` directory.

5. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```

## Usage

- Navigate to the dashboard to view various analytics.
- Use the sidebar to switch between different models: Customer Churn, Product Recommendation, and Sales Prediction.
- Click buttons to run predictions and recommendations.

## File Structure

- `app.py`: Main Streamlit application script.
- `models/churn.py`: Contains the customer churn prediction model.
- `models/product.py`: Contains the product recommendation model.
- `models/sales.py`: Contains the sales prediction model.
- `data/orders_export_new_!.csv`: Sample data file.

## Dependencies

List of required Python packages are included in `requirements.txt`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Streamlit
- XGBoost
- TensorFlow
- Seaborn
- Matplotlib
- Folium

