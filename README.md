This repository shows how I implemented different forecasting models on datasets.

!! "One Hot Label Encoding.py" and "Preparation with External Drivers.py" files indicates how I modified the dataset by adding external drivers or handling categorical variables. They are independent of statistical and machine learning files.

!! For "Preparation with External Drivers.py" file, you have to use both 'norway_car_sales' and 'GDP.csv'.

# Statistical_Forecast: 

I would like to show how to implement forecasting models starting from scratch as follows of:

1. Moving Average
2. Simple Exponential Smoothing
3. Double Exponential Smoothing with damped trend
4. Multiplicative Triple Exponential Smoothing
5. Additive Triple Exponential Smoothing

# ML_Forecast:

I ran many machine learning algorithms (Linear Regression, Random Forest, Extremely Randomized Trees, AdaBoost and Extreme Gradient Boosting) on time series dataset.

Required dataset: 'norway_car_sales.csv' in the repository

ML_Forecast_v1.0 - Linear Regression and Regression Trees

ML_Forecast_v2.0 - Extremely Randomized Trees and Feature Optimization

ML_Forecast_v3.0 - Adaptive Boosting (adaBoost) and xgBoost (Extreme Gradient Boosting)

# Walmart_Forecasting: 

I implemented 4 statistical forecasting methods with tuned optimization on Walmart dataset which was published in 2014 and calculated forecasting accuracy for all SKU's.

1. Moving Average
2. Simple Exponential Smoothing
3. Double Exponential Smoothing
4. Multiplicative Triple Exponential Smoothing with damped trend

The link of the dataset: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting or 'train.csv' in the repository
