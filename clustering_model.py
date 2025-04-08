import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder

def euclidean_squared(p1, p2):
    """Euclidean squared distance between two points."""
    return np.sum((p1 - p2) ** 2)

def kmeans(data, k, max_iters=100, tol=1e-4):
    """K-means clustering algorithm."""
    np.random.seed(0)
    initial_indices = np.random.choice(len(data), k, replace=False)
    centroids = data[initial_indices]
    
    for iteration in range(max_iters):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # Check for convergence
        shift = np.sum((centroids - new_centroids) ** 2)
        if shift < tol:
            print(f"Converged after {iteration} iterations.")
            break

        centroids = new_centroids
    return centroids, labels

def assign_centroid_genres(data, centroids, song_df, num_closest=20):
    """Assign genres to centroids based on closest songs."""
    centroid_genre_labels = []
    
    for i in range(len(centroids)):
        distances = np.linalg.norm(data - centroids[i], axis=1)
        closest_indices = np.argsort(distances)[1:num_closest + 1]

        closest_genres = song_df['Genres'].iloc[closest_indices].tolist()
        genre_counts = pd.Series(closest_genres).value_counts(normalize=True)

        dominant_genre = genre_counts.idxmax()
        dominant_percentage = genre_counts.max()

        if dominant_percentage >= 0.6:
            centroid_genre_labels.append(dominant_genre)
        else:
            fusion_genres = genre_counts.index.tolist()
            centroid_genre_labels.append('-'.join(fusion_genres) if len(fusion_genres) == 2 else "Mixed")
    
    return centroid_genre_labels

def plot_clusters_3d_interactive(data, centroids, labels, song_names, centroid_labels=None):
    """Plot 3D interactive visualization of clusters."""
    le = LabelEncoder()
    genre_labels = le.fit_transform(song_names)
    
    trace = go.Scatter3d(
        x=data[:, 0],
        y=data[:, 1],
        z=data[:, 2],
        mode='markers',
        marker=dict(size=5, color=genre_labels, colorscale='Viridis'),
        text=song_names,
        hoverinfo='text'
    )

    centroid_texts = [f"Centroid {i+1}: {label}" if centroid_labels else f"Centroid {i+1}" 
                      for i, label in enumerate(centroid_labels or [])]
    centroids_trace = go.Scatter3d(
        x=centroids[:, 0],
        y=centroids[:, 1],
        z=centroids[:, 2],
        mode='markers+text',
        marker=dict(size=10, color='black'),
        text=centroid_texts,
        textposition="top center",
        hoverinfo='text'
    )
    layout = go.Layout(
        title="K-Means Clustering (3D Visualization with Genre Labels)",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        showlegend=False
    )

    fig = go.Figure(data=[trace, centroids_trace], layout=layout)
    fig.show()

def generate_playlist_interactive(data, song_names, centroids, labels, random_index=1250, num_closest=20):
    """Generate playlist recommendations based on a selected song."""
    target_point = data[random_index]
    
    distances = np.linalg.norm(data - target_point, axis=1)
    closest_indices = np.argsort(distances)[1:num_closest + 1]

    traces = []

    traces.append(go.Scatter3d(
        x=data[:, 0], y=data[:, 1], z=data[:, 2],
        mode='markers', marker=dict(size=5, color=labels, colorscale='Viridis'),
        text=song_names, hoverinfo='text', name="All Songs"
    ))

    traces.append(go.Scatter3d(
        x=centroids[:, 0], y=centroids[:, 1], z=centroids[:, 2],
        mode='markers+text', marker=dict(size=10, color='black'),
        text=[f"Centroid {i+1}" for i in range(len(centroids))],
        textposition="top center", hoverinfo='text', name="Centroids"
    ))

    # Selected song
    traces.append(go.Scatter3d(
        x=[target_point[0]], y=[target_point[1]], z=[target_point[2]],
        mode='markers+text', marker=dict(size=12, color='red', symbol='x'),
        text=["Selected Song"], textposition="bottom center", name="Selected Song"
    ))

    # Recommended songs
    recommended_points = data[closest_indices]
    recommended_titles = [song_names[i] for i in closest_indices]
    traces.append(go.Scatter3d(
        x=recommended_points[:, 0], y=recommended_points[:, 1], z=recommended_points[:, 2],
        mode='markers', marker=dict(size=8, color='pink', symbol='circle'),
        text=recommended_titles, hoverinfo='text', name="Recommendations"
    ))

    layout = go.Layout(
        title=f"Playlist Recommendations for '{song_names[random_index]}'",
        scene=dict(xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"),
        showlegend=True
    )
    fig = go.Figure(data=traces, layout=layout)
    fig.show()

    return song_names[random_index], recommended_titles

if __name__ == "__main__":
    # Load PCA data
    pca_file_path = "cleaned_Tscores.csv"
    data = np.loadtxt(pca_file_path, delimiter=",")
    
    # Load song data
    song_file_path = "cleaned_mega.csv"
    song_df = pd.read_csv(song_file_path)
    song_names = song_df['Track Name'].tolist()
    song_genres = song_df['Genres'].tolist()

    assert len(song_names) == len(data), "Mismatch between number of song names and PCA score entries."

    # Run clustering
    k = 6  # Number of clusters
    centroids, labels = kmeans(data, k)

    centroid_genre_labels = assign_centroid_genres(data, centroids, song_df)
    plot_clusters_3d_interactive(data, centroids, labels, song_names, centroid_labels=centroid_genre_labels)

    random_index = np.random.randint(len(data))
    selected_song, recommendations = generate_playlist_interactive(
        data=data, song_names=song_names, centroids=centroids, labels=labels, 
        random_index=random_index, num_closest=20
    )

    print("🎵 Selected Song:", selected_song)
    print("🎧 Recommended Playlist:")
    for song in recommendations:
        print("-", song)
