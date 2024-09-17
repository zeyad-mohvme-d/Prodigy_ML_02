import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.cluster import KMeans

#Data Reading

dataset = pd.read_csv('C:\\Users\\zeyad\\OneDrive\\Documents\\My_Projects\\Prodigy InfoTech_Clustering\\Mall_Customers.csv')
# print(dataset.head)

x = dataset.iloc[:, [3,4]]
# print(x)

#Data Visualization


x.plot()
plt.show()


#Elbow Method Visualization

wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state= 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.show()

plt.scatter(x['Annual Income (k$)'], x['Spending Score (1-100)'])
plt.show()

model = KMeans(n_clusters=5, random_state=42)
model.fit(x)
predictions = model.predict(x)


plt.scatter(x.iloc[predictions == 0, 0], x.iloc[predictions == 0, 1], s = 100, c = 'blue', label = 'Cluster 1') #for first cluster  
plt.scatter(x.iloc[predictions == 1, 0], x.iloc[predictions == 1, 1], s = 100, c = 'green', label = 'Cluster 2') #for second cluster  
plt.scatter(x.iloc[predictions== 2, 0], x.iloc[predictions == 2, 1], s = 100, c = 'red', label = 'Cluster 3') #for third cluster  
plt.scatter(x.iloc[predictions == 3, 0], x.iloc[predictions == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4') #for fourth cluster  
plt.scatter(x.iloc[predictions == 4, 0], x.iloc[predictions == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5') #for fifth cluster  
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroid')   
plt.title('Clusters of customers')  
plt.xlabel('Annual Income (k$)')  
plt.ylabel('Spending Score (1-100)')  
plt.legend()  
plt.show()


