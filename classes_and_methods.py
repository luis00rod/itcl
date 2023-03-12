#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import datetime


# In[2]:


class Time_Series:
    def __init__(self, filename: str):
        self.filename = filename
        self.df = None

    def read_csv_file(self) -> pd.DataFrame:
        """
        Read a CSV file and return a pandas DataFrame.
        """
        with open(self.filename) as file:
            self.df = pd.read_csv(file)
        return self.df

    def drop_duplicates(self) -> pd.DataFrame:
        """
        Drop duplicate rows from the pandas DataFrame and return the resulting DataFrame.
        """
        self.df = self.df.drop_duplicates()
        return self.df

    def cast_timestamp(self) -> pd.DataFrame:
        """
        Cast the 'timestamp' column of the pandas DataFrame to a datetime object and return the resulting DataFrame.
        """
        self.df['timestamp'] = self.df['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))
        return self.df

    def sort_by_timestamp(self) -> pd.DataFrame:
        """
        Sort the pandas DataFrame by the 'timestamp' column in ascending order and return the resulting DataFrame.
        """
        self.df = self.df.sort_values(by=['timestamp'], ascending=True).reset_index().drop(columns=['index'])
        return self.df
    
    def data_preparation(self):
        df_no_duplicates = self.df.drop_duplicates()
        df_sorted = df_no_duplicates.sort_values(by=['timestamp'], ascending=True).reset_index().drop(columns=['index'])
        self.df_sorted = df_sorted
        return self.df_sorted[:int(self.df_sorted.shape[0]*0.85)], self.df_sorted[int(self.df_sorted.shape[0]*0.85):]

    def scale_data(self) -> pd.DataFrame:
        """
        Scale the 'temperatura' column of the pandas DataFrame and return the resulting DataFrame.
        """
        scaler = StandardScaler()
        self.df["temperatura"] = scaler.fit_transform(self.df["temperatura"].values.reshape(-1, 1))
        return self.df  

    def split_train_test_data_train(self):
        """
        Split the pandas DataFrame into train/test sets and return X_train, X_test, y_train, y_test as numpy arrays.
        """
        df_prepared = self.data_preparation()[0]
        X_train, X_test, y_train, y_test = train_test_split(
        df_prepared["temperatura"].values, df_prepared["fallo"].values, test_size=0.2, random_state=42, shuffle=False)
        X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, 1))
        X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, 1))
        return X_train_reshaped, X_test_reshaped, y_train, y_test
 

    def build_model(self) -> Sequential:
        """
        Build an LSTM model and return the model.
        """
        model = Sequential()
        model.add(LSTM(50, input_shape=(1,1)))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def fit_model(self):
        model = self.build_model()
        model.fit(X_train_reshaped, y_train, epochs=20, batch_size=1)
        return model
    
    def evaluate_model(self):
        test_loss, test_acc = model.evaluate(X_test_reshaped, y_test)
        print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')
  
    def predictions(self):
        X = np.reshape(np.array(self.data_preparation()[1]["temperatura"]), (self.data_preparation()[1].shape[0], 1, 1))
        predictions = model.predict(X)
        return predictions


# In[3]:


time_series = Time_Series("historicos series temporales.csv")


# In[4]:


time_series.read_csv_file()


# In[5]:


time_series.drop_duplicates()


# In[6]:


time_series.cast_timestamp()


# In[7]:


time_series.sort_by_timestamp()


# In[8]:


time_series.scale_data()


# In[9]:


time_series.data_preparation()[0]


# In[10]:


df_test = time_series.data_preparation()[1]


# In[11]:


X_train_reshaped, X_test_reshaped, y_train, y_test = time_series.split_train_test_data_train()


# In[12]:


X_train_reshaped.shape[0], X_test_reshaped.shape[0]


# In[13]:


model = time_series.build_model()


# In[14]:


time_series.fit_model()


# In[15]:


predictions= time_series.predictions()


# In[25]:


pd.DataFrame(predictions).describe()


# In[26]:


predictions.shape, df_test.shape


# In[47]:


predictions


# In[46]:


y_pred


# In[44]:


threshold = 0.48
y_pred = np.where(predictions <= threshold, 0, 1)


# In[45]:


f1_score(df_test['fallo'], y_pred)

