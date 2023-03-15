#!/usr/bin/env python
# coding: utf-8

# ## Detección de anomalías en series temporales
# 
# ### Para desarrollar un sistema de alerta temprana que prediga y anticipe posibles problemas en la máquina. Para desarrollar este sistema, seguiría los siguientes pasos:
# 
# #### 1. Análisis y preprocesamiento de datos: En primer lugar, analizaría y preprocesaría los datos proporcionados para asegurarme de que estén limpios y listos para su uso. Esto puede incluir la eliminación outliers y la conversión de los datos de series temporales en un formato adecuado.
# 
# #### 2. División de datos: Divider los datos en conjuntos de entrenamiento, validación y prueba para evaluar y ajustar el rendimiento del modelo.
# 
# #### 3. Selección del algoritmo: Para este problema, se podrían utilizar  algoritmos de aprendizaje supervisado, como regresión logística o redes neuronales recurrentes (RNN) y las redes de memoria a largo plazo (LSTM). También se podrían utilizar algoritmos específicos para series temporales, como ARIMA o SARIMA.
# 
# #### 4. Entrenamiento y ajuste del modelo: Entrenaría el modelo seleccionado utilizando el conjunto de entrenamiento y ajustaría sus hiperparámetros utilizando el conjunto de validación para mejorar su rendimiento.
# 
# #### 5. Evaluación del rendimiento: Una vez que el modelo esté entrenado y ajustado, lo evaluaría en el conjunto de prueba para obtener una estimación realista de su rendimiento en datos no vistos previamente. Utilizaría métricas como el accuracy,  el valor F1_SCORE y la curva ROC-AUC para medir el rendimiento del modelo.
# 
# #### 6. Implementación del sistema de alerta: Si el rendimiento del modelo es satisfactorio, lo implementaría como un sistema de alerta en tiempo real que monitorea las lecturas de temperatura y genera alertas cuando se predice un fallo inminente.
# 
# ### En una primera aproximacion, no es necesario desarrollar un algoritmo ad hoc para resolver este problema, ya que hay varios algoritmos y técnicas disponibles que pueden abordar este tipo de problemas de series temporales. La clave es seleccionar el algoritmo adecuado y ajustarlo correctamente para obtener el mejor rendimiento en este caso específico.

# ### Para llevar a la propuesta he implementado el siguiente codigo: 

# #### Importar librerias necesarias

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


# #### Creacion de clase cuyos metodos seran utilizados para la prediccion de anomalias

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
        self.X_train_reshaped = np.reshape(X_train, (X_train.shape[0], 1, 1))
        self.X_test_reshaped = np.reshape(X_test, (X_test.shape[0], 1, 1))
        self.y_train = y_train
        self.y_test = y_test
        return self.X_train_reshaped, self.X_test_reshaped, self.y_train, self.y_test
 

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
        self.model = self.build_model()
        self.history = self.model.fit(self.X_train_reshaped, self.y_train, epochs=20, batch_size=1)
        return self.history
  
    def predictions(self, df):
        predictions = model.predict(df)
        return predictions    


# ### Paso 1

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


# ### Paso 2

# In[12]:


X_train_reshaped, X_test_reshaped, y_train, y_test = time_series.split_train_test_data_train(df_train)


# ### Paso 3. Para este paso he elegido las redes neuronales LSTM  debido a su habilidad para gestionar datos secuenciales con dependencias a largo plazo.

# In[13]:


model = time_series.build_model()


# In[14]:


model.summary()


# ### Paso 4.

# In[15]:


history = time_series.fit_model()


# In[16]:


model.evaluate(X_test_reshaped, y_test)


# In[17]:


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper right')
    plt.show()


# In[18]:


plot_loss(history)


# ### Paso 5.

# In[19]:


new_data = np.reshape(np.array(df_test.temperatura),(df_test.shape[0], 1, 1))


# In[20]:


# Get the predicted probabilities
predictions = time_series.predictions(new_data)


# In[21]:


threshold = pd.DataFrame(predictions).mean().iloc[0]
pred = (predictions > threshold).astype(int)
data_validation = df_test.fallo
# Compute F1 score
f1 = f1_score(data_validation, pred, average='macro')
print(f'F1 score: {f1}')


# In[22]:


pd.DataFrame(predictions).describe()


# ###  Conclusiones y proximos pasos:
# #### a) Mejorar el preprocesamiento de datos: Revisar y mejorar el proceso de limpieza y preprocesamiento de datos. Asegurarse de que no haya errores y de que se hayan abordado adecuadamente los valores faltantes y los outliers.
# #### b) Ingeniería de características: Crear características adicionales que puedan ser útiles para la predicción, como variables derivadas de las series temporales, estadísticas resumidas y otras características relevantes para el contexto del problema.
# #### c) Revisar el algoritmo seleccionado: Puede ser útil probar otros algoritmos de aprendizaje supervisado. Además, explorar técnicas de ensembling, como la combinación de múltiples modelos, para mejorar el rendimiento.
# #### d) Optimización de hiperparámetros: Realizar una búsqueda más exhaustiva de los hiperparámetros del modelo para mejorar su rendimiento. Se pueden utilizar técnicas como la búsqueda en cuadrícula (Grid Search) o la optimización bayesiana para encontrar los hiperparámetros óptimos.
# #### e) Aumentar el tamaño del conjunto de datos: Si es posible, obtener más datos para aumentar el tamaño del conjunto de entrenamiento, validación y prueba. Más datos pueden mejorar el rendimiento del modelo y proporcionar una evaluación más sólida.
# #### f) Reevaluar el rendimiento: Después de realizar las mejoras, volver a evaluar el rendimiento del modelo utilizando las métricas apropiadas, como el F1_score y la curva ROC-AUC. Si el rendimiento ha mejorado significativamente, considerar la implementación del sistema de alerta temprana.
