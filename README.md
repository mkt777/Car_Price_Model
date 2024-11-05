# Car Price Prediction

## Overview
This project aims to predict the selling price of cars based on various features such as brand, year, mileage, fuel type, seller type, transmission type, engine size, and max power. The model is built using linear regression from the Scikit-learn library and is trained on a dataset containing car details.

## Table of Contents
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Getting Started](#getting-started)
- [Model Training](#model-training)
- [Model Prediction](#model-prediction)
- [Saving the Model](#saving-the-model)
- [Usage](#usage)
- [License](#license)

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn

## Dataset
The dataset used in this project is `Cardetails.csv`, which contains various features of cars. The columns include:
- `name`: Brand name of the car
- `year`: Year of manufacture
- `km_driven`: Distance driven in kilometers
- `fuel`: Type of fuel (Diesel, Petrol, LPG, CNG)
- `seller_type`: Type of seller (Individual, Dealer, Trustmark Dealer)
- `transmission`: Type of transmission (Manual, Automatic)
- `owner`: Ownership status (First Owner, Second Owner, etc.)
- `mileage`: Mileage of the car
- `engine`: Engine size
- `max_power`: Maximum power
- `seats`: Number of seats
- `selling_price`: Selling price (target variable)

