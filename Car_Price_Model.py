#!/usr/bin/env python
# coding: utf-8

# In[217]:


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[218]:


cars_data = pd.read_csv('Cardetails.csv')


# In[219]:


cars_data.head()


# In[220]:


cars_data.drop(columns=['torque'], inplace=True)


# In[221]:


cars_data.head()


# In[222]:


cars_data.shape


# In[223]:


#preprocessing


# In[224]:


#NULL ChecK


# In[225]:


cars_data.isnull().sum()


# In[226]:


cars_data.dropna(inplace=True)


# In[227]:


cars_data.shape


# In[228]:


#Duplicate Check


# In[229]:


cars_data.duplicated().sum()


# In[230]:


cars_data.drop_duplicates(inplace=True)


# In[231]:


cars_data.shape


# In[232]:


cars_data


# In[233]:


cars_data.info()


# In[234]:


#Data Analysis


# In[235]:


for col in cars_data.columns:
    print('Unique values of ' + col)
    print(cars_data[col].unique())
    print("======================")


# In[236]:


def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()


# In[237]:


def clean_data(value):
    value = value.split(' ')[0]
    value = value.strip()
    if value == '':
        value = 0
    return float(value)


# In[238]:


get_brand_name('Maruti  Swift Dzire VDI')


# In[239]:


cars_data['name'] = cars_data['name'].apply(get_brand_name)


# In[240]:


cars_data['name'].unique()


# In[241]:


cars_data['mileage'] = cars_data['mileage'].apply(clean_data)


# In[242]:


cars_data['max_power'] = cars_data['max_power'].apply(clean_data)


# In[243]:


cars_data['engine'] = cars_data['engine'].apply(clean_data)


# In[244]:


for col in cars_data.columns:
    print('Unique values of ' + col)
    print(cars_data[col].unique())
    print("======================")


# In[245]:


cars_data['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                          [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
                          ,inplace=True)


# In[246]:


cars_data['transmission'].unique()


# In[247]:


cars_data['transmission'].replace(['Manual', 'Automatic'],[1,2], inplace=True)


# In[248]:


cars_data['seller_type'].unique()


# In[249]:


cars_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'],[1,2,3], inplace=True)


# In[250]:


cars_data.info()


# In[251]:


cars_data['fuel'].unique()


# In[252]:


cars_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'],[1,2,3,4], inplace=True)


# In[253]:


cars_data.info()


# In[254]:


cars_data.reset_index(inplace=True)


# In[255]:


cars_data


# In[256]:


cars_data['owner'].unique()


# In[257]:


cars_data['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner', 'Test Drive Car'],
                           [1,2,3,4,5], inplace=True)


# In[258]:


cars_data.drop(columns=['index'], inplace=True)


# In[259]:


for col in cars_data.columns:
    print('------------')
    print(col)
    print(cars_data[col].unique())


# In[260]:


cars_data.isnull().sum()


# In[261]:


input_data = cars_data.drop(columns=['selling_price'])
output_data =cars_data['selling_price']


# In[268]:


x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2)


# In[269]:


#model Creation


# In[270]:


model = LinearRegression()


# In[271]:


#Train MOdel


# In[272]:


model.fit(x_train, y_train)


# In[273]:


predict = model.predict(x_test)


# In[274]:


predict


# In[276]:


x_train.head(1)


# In[289]:


input_data_model = pd.DataFrame(
    [[5,2022,12000,1,1,1,1,12.99,2494.0,100.6,5.0]],
    columns=['name','year','km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats'])


# In[290]:


input_data_model


# In[291]:


model.predict(input_data_model)


# In[292]:


import pickle as pk


# In[294]:


pk.dump(model,open('model.pkl','wb'))


# In[ ]:





# In[ ]:




