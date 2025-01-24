from flask import Flask, render_template
import pandas as pd
import numpy as np
import scipy.stats as stats

app = Flask(__name__)

@app.route('/')
def home():
    # Data generation and statistics
    np.random.seed(42)
    data = {
        'product_id': range(1, 21),
        'product_name': [f'Product{i}' for i in range(1, 21)],
        'category': np.random.choice(['Electronic', 'Clothing', 'Home', 'Sports'], 20),
        'units_sold': np.random.poisson(lam=20, size=20),
        'sales_date': pd.date_range(start='2023-01-01', periods=20, freq='D')
    }
    sales_data = pd.DataFrame(data)

    mean_sales = sales_data['units_sold'].mean()
    median_sales = sales_data['units_sold'].median()
    mode_sales = sales_data['units_sold'].mode()[0]
    variance_sales = sales_data['units_sold'].var()
    std_deviation_sales = sales_data['units_sold'].std()

    category_stats = sales_data.groupby('category')['units_sold'].agg(['sum', 'mean']).reset_index()

    confidence_level = 0.95
    degrees_freedom = len(sales_data['units_sold']) - 1
    sample_standard_error = std_deviation_sales / np.sqrt(len(sales_data['units_sold']))
    t_score = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    margin_of_error = t_score * sample_standard_error
    confidence_interval = (mean_sales - margin_of_error, mean_sales + margin_of_error)

    # Rendering the template with the calculated data
    return render_template('index.html', 
                           mean_sales=mean_sales,
                           median_sales=median_sales,
                           mode_sales=mode_sales,
                           variance_sales=variance_sales,
                           std_deviation_sales=std_deviation_sales,
                           category_stats=category_stats.to_dict(orient='records'),
                           confidence_interval=confidence_interval)

if __name__ == '__main__':
    app.run(debug=True)
