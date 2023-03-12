#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv("lidar.csv")
df


# In[3]:


df.describe()


# In[4]:


# Visualizar los datos en un gráfico 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(df["X"], df["Y"], df["Z"], s=0.2)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()


# In[5]:


# Seleccionar las columnas que se usarán para el clustering
X = df[["X", "Y", "Z"]]

# Inicializar el algoritmo K-Means con el número de clusters deseado
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters)

# Ejecutar el algoritmo de clustering en los datos
kmeans.fit(X)

# Obtener las etiquetas de cluster asignadas a cada punto
labels = kmeans.labels_

# Imprimir las etiquetas de cluster asignadas a cada punto
print(labels)


# In[6]:


df["label"] = pd.Series(labels)


# In[7]:


df.label.unique()


# In[8]:


df.label.value_counts()


# In[9]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

for label, color in zip(np.unique(df.label), colors):
    if label == -1:
        # Noisy points
        col = 'k'
        msize = 1
    else:
        col = color
        msize = 3
    
    ax.scatter(df[df.label==label]["X"],
               df[df.label==label]["Y"],
               df[df.label==label]["Z"],
               c=col,
               s=msize)

plt.show()


# In[48]:


df_train = df[:int(len(df)*0.7)]
df_test = df[int(len(df)*0.7):]
df_train.shape, df_test.shape


# In[39]:


y_test


# In[62]:


X_test.shape[0],X_train.shape[0], X_test.shape[0]/X_train.shape[0]


# In[59]:


# Cargar y dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(df_train.drop('label', axis=1),                                    df_train['label'], train_size=2264,  shuffle=True)


# In[60]:


# Crear el modelo de Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
rf.fit(X_train, y_train)

# Predecir sobre el conjunto de prueba
y_pred = rf.predict(X_test)


# In[63]:


f1 = f1_score(df_test.label, y_pred, average="macro")

print("f1_score:", f1)


# In[64]:


pd.DataFrame( y_pred).value_counts()


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





# In[14]:


# Preprocesamiento de los datos
# X_train = X_train.reshape(-1, 1, 3, 1)  # Reorganizar los datos para que sean compatibles con una CNN
# X_test = X_test.reshape(-1, 1, 3, 1)
# y_train = tf.keras.utils.to_categorical(y_train)  # Codificación one-hot de las etiquetas
# y_test = tf.keras.utils.to_categorical(y_test)


# In[15]:


# Construir el modelo de la CNN
# model = Sequential([
#     Conv2D(32, kernel_size=(1, 3), activation="relu", input_shape=(1, 3, 1)),
#     MaxPooling2D(pool_size=(1, 1)),
#     Flatten(),
#     Dense(64, activation="relu"),
#     Dense(3, activation="softmax")
# ])


# In[16]:


# model.summary()


# In[17]:


# model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# In[18]:


# model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


# In[19]:


# y_pred = model.predict(X_test)
# np.round(y_pred)


# In[20]:


# pd.DataFrame(np.round(y_pred))[0].sum(),pd.DataFrame(np.round(y_pred))[1].sum(),pd.DataFrame(np.round(y_pred))[2].sum()


# In[21]:


# pred = pd.DataFrame( y_pred.argmax(axis=1))
# pred


# In[ ]:




