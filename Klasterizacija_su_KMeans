import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px

#Duomenų valymas ir paruošimas naudojimui
df = pd.read_csv('ecommerce_customer_data_custom_ratios.csv', nrows=1000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# --Sutvarkome returns skiltį, pašaliname NaN reikšmes--
df['Returns'] = df['Returns'].fillna(0).astype(int)

# --Pašaliname 'customer age' stulpelį, nes jis identiškas 'age' stulpeliui--
df = df.drop('Customer Age', axis=1)

# --Pasirenkame savybes--

features = df[['Customer ID', 'Purchase Date', 'Quantity', 'Product Price']]

df['Suma'] = df['Quantity'] * df['Product Price']

features = df.groupby('Customer ID').agg({'Purchase Date': 'nunique', 'Quantity': 'sum', 'Suma': ['sum', 'mean']}).reset_index()

features.columns = ['Customer ID', 'Purchase Date', 'Total quantity', 'Total spent', 'Average spent per purchase']

# --Standartizuojame duomenis--
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features[['Purchase Date', 'Total quantity',
                                                           'Total spent', 'Average spent per purchase']])


#-Tikriname ir randame geriausia klasteri, apskaičiuojame Sill score-
silhouette_scores = []
cluster_values = [3,6,9]
for k in cluster_values:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, max_iter=300)
    kmeans.fit(features_scaled)
    score = silhouette_score(features_scaled, kmeans.labels_)
    silhouette_scores.append(score)
    print(f'K-Means silhouette score: k= {k}, {score}')

# --Geriausio silhoutette score apskaičiavimas--
best_score = cluster_values[silhouette_scores.index(max(silhouette_scores))]
print(f'Best silhouette score for k = {best_score}')

# --Kmeans best--
kmeans_best = KMeans(n_clusters=best_score, init='k-means++', random_state=42)

kmeans_best = KMeans(n_clusters=best_score, random_state=42)
features['Cluster'] = kmeans_best.fit_predict(features_scaled)

# --Vizualizacija--
plt.figure(figsize=(10, 8))
plt.scatter(features_scaled[:, 2], features_scaled[:, 3], c=features['Cluster'], cmap='viridis')
plt.title('Klientų segmentacija naudojant Kmeans')
plt.xlabel('Standartizuotas išleistas bendras kiekis')
plt.ylabel('Standartizuotas bendras išleistas pinigų kiekis per pirkimą')
plt.colorbar(label='Klasteris')
plt.show()


