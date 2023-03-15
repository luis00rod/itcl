#!/usr/bin/env python
# coding: utf-8

# ### Para desarrollar un algoritmo que detecte y clasifique las estructuras ferroviarias presentes en la nube de puntos, seguiría los siguientes pasos:
# 
# ### 1. Preprocesamiento de datos:
# ### a.) Limpiar y filtrar los datos para eliminar ruido y puntos irrelevantes (como puntos del suelo).
# ### b.) Aplicar un filtro de tamaño a la nube de puntos para eliminar elementos muy pequeños que puedan ser ruido o elementos no deseados, como ramas de árboles.
# 
# ### 2. Segmentación del terreno y extracción de características:
# ### a.) Realizar una segmentación del terreno utilizando técnicas como el algoritmo RANSAC para identificar y separar los elementos estructurales de la vía férrea.
# ### b.) Extraer características de los segmentos, como tamaño, forma, orientación y distribución espacial.
# 
# ### 3. Clustering y agrupación:
# ### a.) Aplicar técnicas de clustering como k-means, DBSCAN para agrupar puntos similares en función de sus características y ubicaciones espaciales.
# ### b.) Establecer umbrales para determinar si un grupo representa una estructura ferroviaria relevante o no  (como el tamaño mínimo de un poste, por ejemplo).
# 
# ### 4. Clasificación:
# ### a.) Entrenar un modelo de aprendizaje supervisado, como un árbol de decisión, una red neuronal, un clasificador SVM o un clasificador de Random Forest, utilizando datos etiquetados de estructuras ferroviarias y no ferroviarias.
# ### b.) Utilizar el modelo entrenado para clasificar los grupos de puntos como diferentes tipos de estructuras ferroviarias (postes, hilos de contacto, brazos de atirantado, etc.) o elementos no deseados (ramas de árboles, paredes, etc).
# 
# ### 5. Validación y ajuste del modelo:
# ### a.) Validar el rendimiento del modelo utilizando métricas de evaluación como accuracy, F1-score  o matriz de confusión.
# 
# ### 6. Visualización y análisis de resultados:
# ### a.) Visualizar las clasificaciones y las estructuras detectadas en la nube de puntos para verificar si el algoritmo  ha detectado correctamente los elementos de interés. 
# 

# In[1]:


import open3d as o3d
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# In[2]:


# Cargar datos (reemplazar con la ruta de tu archivo CSV)
data = pd.read_csv('lidar.csv')

# Convertir el dataframe a una nube de puntos de Open3D
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(data[['X', 'Y', 'Z']].values)


# In[3]:


# Filtrado de datos: eliminar ruido y puntos irrelevantes
pcd_filtered = pcd.voxel_down_sample(voxel_size=0.05)

# Segmentación del terreno y extracción de características
# Aquí aplicamos un filtro de segmentación del plano usando RANSAC
plane_model, inliers = pcd_filtered.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
pcd_without_ground = pcd_filtered.select_by_index(inliers, invert=True)


# In[4]:


# Extraer características (en este caso solo se utiliza la posición XYZ)
features = np.asarray(pcd_without_ground.points)

# Normalizar las características para mejorar el rendimiento del clustering
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# In[5]:


# Clustering y agrupación
dbscan = DBSCAN(eps=0.1, min_samples=15)
labels = dbscan.fit_predict(features_scaled)


# In[6]:


labels


# In[7]:


# Asignar colores a las etiquetas
max_label = labels.max() + 1
label_to_color = {i: np.random.rand(3) for i in range(max_label)}
label_to_color[-1] = np.array([0, 0, 0])  
# Visualización de resultados de clustering
pcd_without_ground.colors = o3d.utility.Vector3dVector(np.asarray([label_to_color[label] for label in labels]))
o3d.visualization.draw_geometries([pcd_without_ground])

# Aquí deberías obtener tus datos etiquetados para entrenar un modelo de aprendizaje supervisado
# y aplicar la clasificación como se describió en la respuesta anterior.


# In[13]:


points = np.asarray(pcd_without_ground.points)


# In[14]:


df = pd.DataFrame(points, columns=['X', 'Y', 'Z'])
df['label'] = labels

# Mostrar las primeras filas del DataFrame
df.head()


# In[15]:


labels.shape


# In[17]:


points.shape


# In[ ]:




