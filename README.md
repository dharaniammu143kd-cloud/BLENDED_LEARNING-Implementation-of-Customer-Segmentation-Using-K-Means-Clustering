# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import required libraries and load the **Mall Customers dataset**.
2. Select features (Age, Annual Income, Spending Score) and scale the data using **StandardScaler**.
3. Use the **Elbow Method** to find the optimal number of clusters and apply **K-Means clustering**.
4. Assign cluster labels to customers and visualize the clusters using a scatter plot.


## Program:
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
data = pd.read_csv('CustomerData.csv')
print(data.head())
print(data.columns)
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(8,4))
plt.plot(range(1, 11), wcss, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()
optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(X_scaled)
data['Cluster'] = kmeans.labels_
sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')
plt.figure(figsize=(10,5))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis', s=100, alpha=0.7)
```

## Output:
<img width="789" height="147" alt="image" src="https://github.com/user-attachments/assets/b71ac0ad-0b4f-4f8e-8801-b4bdfea8029a" />
<img width="751" height="356" alt="image" src="https://github.com/user-attachments/assets/91a21313-2454-4c22-9110-42c2832da97a" />
<img width="813" height="408" alt="image" src="https://github.com/user-attachments/assets/9434d69c-969a-4d23-beb0-e9a6545f4622" />




## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
