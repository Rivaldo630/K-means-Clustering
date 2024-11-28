import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Charger les données
uploaded_file = "Iris.csv"  # Charger le fichier Iris fourni
data = pd.read_csv(uploaded_file)

# Interface Streamlit
st.title("Clustering K-Means ")
st.write("Explorez les clusters K-Means de manière interactive avec les données Iris.")

# Afficher les premières lignes du dataset
st.subheader("Aperçu des données")
st.write(data.head())

# Sélection des colonnes pour le clustering
st.sidebar.subheader("Options de clustering")
columns = st.sidebar.multiselect("Sélectionnez les colonnes pour le clustering :", data.columns, default=data.columns[1:4])

# Vérifier qu'au moins deux colonnes sont sélectionnées
if len(columns) < 2:
    st.error("Veuillez sélectionner au moins deux colonnes pour effectuer le clustering.")
else:
    # Normalisation des données
    scaler = StandardScaler()
    X = scaler.fit_transform(data[columns])

    # Sélectionner le nombre de clusters
    n_clusters = st.sidebar.slider("Nombre de clusters (k)", min_value=1, max_value=10, value=3, step=1)

    # Calcul et affichage du WCSS pour la méthode du coude
    st.subheader("Méthode du coude")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Tracer le graphique du coude
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 11), wcss, marker='o')
    ax.set_title('La méthode du coude')
    ax.set_xlabel('Nombre de clusters')
    ax.set_ylabel('WCSS')
    st.pyplot(fig)

    # Appliquer K-means avec le k optimal
    kmeans_optimal = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans_optimal = kmeans_optimal.fit_predict(X)

    # Ajouter les clusters au dataframe
    data['Cluster'] = y_kmeans_optimal

    # Visualisation des clusters
    st.subheader(f"Visualisation des clusters pour k={n_clusters}")
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y_kmeans_optimal, cmap='viridis', s=50)
    centers = kmeans_optimal.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, label='Centres')
    ax.set_title("Clusters et leurs centres")
    ax.set_xlabel(columns[0])
    ax.set_ylabel(columns[1])
    ax.legend()
    st.pyplot(fig)

    # Afficher les données avec les clusters
    st.subheader("Données avec clusters")
    st.write(data)
