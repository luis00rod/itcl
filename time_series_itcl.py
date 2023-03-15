#!/usr/bin/env python
# coding: utf-8

# ### Aceptaría la propuesta de los dueños de la máquina para desarrollar un sistema de alerta temprana que prediga y anticipe posibles problemas en la máquina. Para desarrollar este sistema, seguiría los siguientes pasos:
# 
# 
# 
# ### 1.) Análisis y preprocesamiento de datos: En primer lugar, analizaría y preprocesaría los datos proporcionados para asegurarme de que estén limpios y listos para su uso. Esto puede incluir la eliminación outliers y la conversión de los datos de series temporales en un formato adecuado.
# 
# ### 2.) Selección de características: A continuación, identificaría las características relevantes que pueden contribuir a la detección temprana de fallos. En este caso, los datos ya incluyen valores normalizados de temperatura y una variable índice que indica la ocurrencia de un fallo. Sin embargo, también podría considerar la posibilidad de utilizar características derivadas, como la tasa de cambio de temperatura o promedios móviles, que podrían ser útiles para detectar patrones en los datos.
# 
# ### 3.) División de datos: Dividiría los datos en conjuntos de entrenamiento, validación y prueba para evaluar y ajustar el rendimiento del modelo.
# 
# ### 4.) Selección del algoritmo: Para este problema, se podrían utilizar  algoritmos de aprendizaje supervisado, como regresión logística o redes neuronales recurrentes (RNN) y las redes de memoria a largo plazo (LSTM). También se podrían utilizar algoritmos específicos para series temporales, como ARIMA o SARIMAX.
# 
# ### 5.) Entrenamiento y ajuste del modelo: Entrenaría el modelo seleccionado utilizando el conjunto de entrenamiento y ajustaría sus hiperparámetros utilizando el conjunto de validación para mejorar su rendimiento.
# 
# ### 6.) Evaluación del rendimiento: Una vez que el modelo esté entrenado y ajustado, lo evaluaría en el conjunto de prueba para obtener una estimación realista de su rendimiento en datos no vistos previamente. Utilizaría métricas como el accuracy,  el valor F1_SCORE y la curva ROC-AUC para medir el rendimiento del modelo.
# 
# ### 7.) Implementación del sistema de alerta: Si el rendimiento del modelo es satisfactorio, lo implementaría como un sistema de alerta en tiempo real que monitorea las lecturas de temperatura y genera alertas cuando se predice un fallo inminente.
# 
# ### En resumen, no es necesario desarrollar un algoritmo ad hoc para resolver este problema, ya que hay varios algoritmos y técnicas disponibles que pueden abordar este tipo de problemas de series temporales. La clave es seleccionar el algoritmo adecuado y ajustarlo correctamente para obtener el mejor rendimiento en este caso específico.

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
    
    def data_preparation_train(self, df):
        return df[:int(df.shape[0]*0.85)]

    def data_preparation_test(self, df):
        return df[int(df.shape[0]*0.85):]
    
    def scale_data(self,df) -> pd.DataFrame:
        """
        Scale the 'temperatura' column of the pandas DataFrame and return the resulting DataFrame.
        """
        scaler = StandardScaler()
        df["temperatura"] = scaler.fit_transform(df["temperatura"].values.reshape(-1, 1))
        return df  

    def split_train_test_data_train(self, df):
        """
        Split the pandas DataFrame into train/test sets and return X_train, X_test, y_train, y_test as numpy arrays.
        """
        X_train, X_test, y_train, y_test = train_test_split(
        df["temperatura"].values, df["fallo"].values, test_size=0.2, random_state=42, shuffle=False)
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
  
    def predictions(self, df):
        predictions = model.predict(df)
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


df = time_series.sort_by_timestamp()
df


# In[8]:


df_train = time_series.data_preparation_train(df)
df_train


# In[9]:


df_test = time_series.data_preparation_test(df)
df_test


# In[10]:


df_test = time_series.scale_data(df_test)
df_test


# In[11]:


df_train = time_series.scale_data(df_train)
df_train


# In[12]:


X_train_reshaped, X_test_reshaped, y_train, y_test = time_series.split_train_test_data_train(df_train)


# In[13]:


model = time_series.build_model()


# In[14]:


model.summary()


# In[15]:


time_series.fit_model()


# In[16]:


new_data = np.reshape(np.array(df_test.temperatura),(df_test.shape[0], 1, 1))


# In[17]:


# Get the predicted probabilities
predictions = time_series.predictions(new_data)


# In[18]:


threshold = pd.DataFrame(predictions).mean().iloc[0]
pred = (predictions > threshold).astype(int)
# Compute F1 score
f1 = f1_score(df_test.fallo, pred, average='macro')
print(f'F1 score: {f1}')


# In[20]:


pd.DataFrame(predictions).describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




