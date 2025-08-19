import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from datetime import datetime
import warnings
from pathlib import Path
import json
import joblib

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class PlayerProfileClustering:
    """Advanced clustering analysis for player profiles and behavior patterns."""
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.pca = None
        self.tsne = None
        self.clustering_models = {}
        self.cluster_labels = {}
        self.cluster_centers = {}
        self.cluster_profiles = {}
        self.feature_importance = {}
        self.clustering_metrics = {}
        
        # Initialize clustering algorithms
        self._initialize_clustering_models()
    
    def _initialize_clustering_models(self):
        """Initialize different clustering algorithms."""
        
        self.clustering_models = {
            'kmeans': KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            ),
            'gmm': GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.random_state,
                covariance_type='full'
            ),
            'dbscan': DBSCAN(
                eps=0.5,
                min_samples=5,
                metric='euclidean'
            ),
            'hierarchical': AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage='ward'
            )
        }
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and engineer features for clustering."""
        
        logger.info("Preparing features for clustering...")
        
        # Basic statistics
        features = pd.DataFrame()
        
        # Disciplinary features
        features['avg_fouls_per_game'] = df.groupby('player_id')['fouls'].mean()
        features['avg_yellow_cards_per_game'] = df.groupby('player_id')['yellow_cards'].mean()
        features['avg_red_cards_per_game'] = df.groupby('player_id')['red_cards'].mean()
        features['total_fouls'] = df.groupby('player_id')['fouls'].sum()
        features['total_yellow_cards'] = df.groupby('player_id')['yellow_cards'].sum()
        features['total_red_cards'] = df.groupby('player_id')['red_cards'].sum()
        
        # Performance features
        if 'minutes_played' in df.columns:
            features['fouls_per_minute'] = (
                df.groupby('player_id')['fouls'].sum() / 
                df.groupby('player_id')['minutes_played'].sum()
            ).fillna(0)
            
            features['cards_per_minute'] = (
                (df.groupby('player_id')['yellow_cards'].sum() + 
                 df.groupby('player_id')['red_cards'].sum()) / 
                df.groupby('player_id')['minutes_played'].sum()
            ).fillna(0)
        
        # Consistency features
        features['fouls_std'] = df.groupby('player_id')['fouls'].std().fillna(0)
        features['fouls_cv'] = (
            features['fouls_std'] / features['avg_fouls_per_game']
        ).replace([np.inf, -np.inf], 0).fillna(0)
        
        # Temporal features
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['month'] = df['date'].dt.month
            df['day_of_week'] = df['date'].dt.dayofweek
            
            # Seasonal patterns
            features['fouls_early_season'] = (
                df[df['month'].isin([8, 9, 10])]
                .groupby('player_id')['fouls'].mean()
            ).fillna(features['avg_fouls_per_game'])
            
            features['fouls_late_season'] = (
                df[df['month'].isin([4, 5, 6])]
                .groupby('player_id')['fouls'].mean()
            ).fillna(features['avg_fouls_per_game'])
        
        # Position-based features (if available)
        if 'position' in df.columns:
            position_dummies = pd.get_dummies(
                df.groupby('player_id')['position'].first(),
                prefix='pos'
            )
            features = features.join(position_dummies, how='left')
        
        # Team-based features
        if 'team' in df.columns:
            features['team_avg_fouls'] = (
                df.groupby(['player_id', 'team'])['fouls'].mean()
                .groupby('player_id').mean()
            )
        
        # Advanced disciplinary metrics
        features['disciplinary_points'] = (
            features['total_yellow_cards'] + features['total_red_cards'] * 2
        )
        
        features['foul_to_card_ratio'] = (
            features['total_fouls'] / 
            (features['total_yellow_cards'] + features['total_red_cards'] + 1)
        )
        
        # Risk indicators
        features['high_foul_games'] = (
            df[df['fouls'] >= 3].groupby('player_id').size()
        ).fillna(0)
        
        features['card_games'] = (
            df[(df['yellow_cards'] > 0) | (df['red_cards'] > 0)]
            .groupby('player_id').size()
        ).fillna(0)
        
        # Fill missing values
        features = features.fillna(0)
        
        # Remove infinite values
        features = features.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Prepared {len(features.columns)} features for {len(features)} players")
        
        return features
    
    def find_optimal_clusters(
        self, 
        X: pd.DataFrame, 
        max_clusters: int = 15,
        methods: List[str] = None
    ) -> Dict[str, Any]:
        """Find optimal number of clusters using multiple methods."""
        
        if methods is None:
            methods = ['elbow', 'silhouette', 'calinski_harabasz']
        
        logger.info("Finding optimal number of clusters...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        results = {}
        cluster_range = range(2, max_clusters + 1)
        
        # Elbow method
        if 'elbow' in methods:
            inertias = []
            for k in cluster_range:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
            
            results['elbow'] = {
                'k_values': list(cluster_range),
                'inertias': inertias,
                'optimal_k': self._find_elbow_point(list(cluster_range), inertias)
            }
        
        # Silhouette analysis
        if 'silhouette' in methods:
            silhouette_scores = []
            for k in cluster_range:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                score = silhouette_score(X_scaled, labels)
                silhouette_scores.append(score)
            
            optimal_k = cluster_range[np.argmax(silhouette_scores)]
            
            results['silhouette'] = {
                'k_values': list(cluster_range),
                'scores': silhouette_scores,
                'optimal_k': optimal_k
            }
        
        # Calinski-Harabasz index
        if 'calinski_harabasz' in methods:
            ch_scores = []
            for k in cluster_range:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                score = calinski_harabasz_score(X_scaled, labels)
                ch_scores.append(score)
            
            optimal_k = cluster_range[np.argmax(ch_scores)]
            
            results['calinski_harabasz'] = {
                'k_values': list(cluster_range),
                'scores': ch_scores,
                'optimal_k': optimal_k
            }
        
        # Davies-Bouldin index (lower is better)
        if 'davies_bouldin' in methods:
            db_scores = []
            for k in cluster_range:
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = kmeans.fit_predict(X_scaled)
                score = davies_bouldin_score(X_scaled, labels)
                db_scores.append(score)
            
            optimal_k = cluster_range[np.argmin(db_scores)]
            
            results['davies_bouldin'] = {
                'k_values': list(cluster_range),
                'scores': db_scores,
                'optimal_k': optimal_k
            }
        
        # Consensus optimal k
        optimal_ks = [results[method]['optimal_k'] for method in results.keys()]
        consensus_k = max(set(optimal_ks), key=optimal_ks.count)
        
        results['consensus'] = {
            'optimal_k': consensus_k,
            'method_votes': dict(zip(results.keys(), optimal_ks))
        }
        
        logger.info(f"Optimal number of clusters: {consensus_k}")
        
        return results
    
    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """Find elbow point using the knee locator method."""
        
        # Simple elbow detection using second derivative
        if len(inertias) < 3:
            return k_values[0]
        
        # Calculate second derivatives
        second_derivatives = []
        for i in range(1, len(inertias) - 1):
            second_deriv = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            second_derivatives.append(second_deriv)
        
        # Find maximum second derivative (elbow point)
        if second_derivatives:
            elbow_idx = np.argmax(second_derivatives) + 1
            return k_values[elbow_idx]
        
        return k_values[len(k_values)//2]  # Default to middle value
    
    def fit_clustering(
        self, 
        X: pd.DataFrame, 
        algorithm: str = 'kmeans',
        n_clusters: int = None
    ) -> Dict[str, Any]:
        """Fit clustering algorithm to the data."""
        
        if n_clusters is not None:
            self.n_clusters = n_clusters
            if algorithm in ['kmeans', 'gmm', 'hierarchical']:
                if algorithm == 'kmeans':
                    self.clustering_models[algorithm].n_clusters = n_clusters
                elif algorithm == 'gmm':
                    self.clustering_models[algorithm].n_components = n_clusters
                elif algorithm == 'hierarchical':
                    self.clustering_models[algorithm].n_clusters = n_clusters
        
        logger.info(f"Fitting {algorithm} clustering with {self.n_clusters} clusters...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit clustering model
        model = self.clustering_models[algorithm]
        
        if algorithm == 'dbscan':
            labels = model.fit_predict(X_scaled)
            n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
            logger.info(f"DBSCAN found {n_clusters_found} clusters")
        else:
            model.fit(X_scaled)
            labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X_scaled)
        
        # Store results
        self.cluster_labels[algorithm] = labels
        
        # Calculate cluster centers
        if algorithm == 'kmeans':
            self.cluster_centers[algorithm] = model.cluster_centers_
        elif algorithm == 'gmm':
            self.cluster_centers[algorithm] = model.means_
        else:
            # Calculate centers manually for other algorithms
            centers = []
            for cluster_id in np.unique(labels):
                if cluster_id != -1:  # Exclude noise points in DBSCAN
                    cluster_points = X_scaled[labels == cluster_id]
                    center = np.mean(cluster_points, axis=0)
                    centers.append(center)
            self.cluster_centers[algorithm] = np.array(centers)
        
        # Calculate clustering metrics
        if len(np.unique(labels)) > 1:
            metrics = {
                'silhouette_score': silhouette_score(X_scaled, labels),
                'calinski_harabasz_score': calinski_harabasz_score(X_scaled, labels),
                'davies_bouldin_score': davies_bouldin_score(X_scaled, labels),
                'n_clusters': len(np.unique(labels)) - (1 if -1 in labels else 0),
                'n_noise_points': np.sum(labels == -1) if -1 in labels else 0
            }
        else:
            metrics = {'error': 'Only one cluster found'}
        
        self.clustering_metrics[algorithm] = metrics
        
        # Generate cluster profiles
        self.cluster_profiles[algorithm] = self._generate_cluster_profiles(
            X, labels, algorithm
        )
        
        logger.info(f"Clustering completed. Silhouette score: {metrics.get('silhouette_score', 'N/A')}")
        
        return {
            'labels': labels,
            'metrics': metrics,
            'cluster_profiles': self.cluster_profiles[algorithm]
        }
    
    def _generate_cluster_profiles(
        self, 
        X: pd.DataFrame, 
        labels: np.ndarray, 
        algorithm: str
    ) -> Dict[int, Dict[str, Any]]:
        """Generate detailed profiles for each cluster."""
        
        profiles = {}
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Skip noise points
                continue
            
            cluster_mask = labels == cluster_id
            cluster_data = X[cluster_mask]
            
            profile = {
                'size': int(np.sum(cluster_mask)),
                'percentage': float(np.sum(cluster_mask) / len(X) * 100),
                'statistics': {},
                'top_features': {},
                'player_examples': []
            }
            
            # Calculate statistics for each feature
            for feature in X.columns:
                feature_data = cluster_data[feature]
                profile['statistics'][feature] = {
                    'mean': float(feature_data.mean()),
                    'std': float(feature_data.std()),
                    'min': float(feature_data.min()),
                    'max': float(feature_data.max()),
                    'median': float(feature_data.median())
                }
            
            # Identify top distinguishing features
            overall_means = X.mean()
            cluster_means = cluster_data.mean()
            
            # Calculate feature importance as relative difference from overall mean
            feature_importance = abs((cluster_means - overall_means) / overall_means)
            feature_importance = feature_importance.replace([np.inf, -np.inf], 0).fillna(0)
            
            top_features = feature_importance.nlargest(5)
            profile['top_features'] = {
                feature: {
                    'importance': float(importance),
                    'cluster_mean': float(cluster_means[feature]),
                    'overall_mean': float(overall_means[feature]),
                    'difference': float(cluster_means[feature] - overall_means[feature])
                }
                for feature, importance in top_features.items()
            }
            
            # Add player examples (if index represents player IDs)
            if len(cluster_data) > 0:
                sample_size = min(5, len(cluster_data))
                sample_players = cluster_data.sample(n=sample_size, random_state=42)
                profile['player_examples'] = sample_players.index.tolist()
            
            profiles[int(cluster_id)] = profile
        
        return profiles
    
    def predict_cluster(
        self, 
        X: pd.DataFrame, 
        algorithm: str = 'kmeans'
    ) -> np.ndarray:
        """Predict cluster membership for new data."""
        
        if algorithm not in self.clustering_models:
            raise ValueError(f"Algorithm {algorithm} not available")
        
        model = self.clustering_models[algorithm]
        
        if not hasattr(model, 'predict') and algorithm != 'dbscan':
            raise ValueError(f"Model {algorithm} not fitted")
        
        # Scale features using fitted scaler
        X_scaled = self.scaler.transform(X)
        
        if algorithm == 'dbscan':
            # For DBSCAN, we need to use a different approach
            # Find closest cluster center
            if algorithm in self.cluster_centers:
                centers = self.cluster_centers[algorithm]
                distances = np.linalg.norm(
                    X_scaled[:, np.newaxis] - centers, axis=2
                )
                predictions = np.argmin(distances, axis=1)
            else:
                raise ValueError("DBSCAN model not fitted")
        else:
            predictions = model.predict(X_scaled)
        
        return predictions
    
    def reduce_dimensions(
        self, 
        X: pd.DataFrame, 
        method: str = 'pca',
        n_components: int = 2
    ) -> np.ndarray:
        """Reduce dimensionality for visualization."""
        
        X_scaled = self.scaler.fit_transform(X)
        
        if method == 'pca':
            self.pca = PCA(n_components=n_components, random_state=self.random_state)
            X_reduced = self.pca.fit_transform(X_scaled)
            
            logger.info(
                f"PCA explained variance ratio: {self.pca.explained_variance_ratio_}"
            )
            
        elif method == 'tsne':
            self.tsne = TSNE(
                n_components=n_components,
                random_state=self.random_state,
                perplexity=min(30, len(X) - 1)
            )
            X_reduced = self.tsne.fit_transform(X_scaled)
            
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
        
        return X_reduced
    
    def visualize_clusters(
        self, 
        X: pd.DataFrame, 
        algorithm: str = 'kmeans',
        save_path: str = None
    ) -> go.Figure:
        """Create interactive cluster visualization."""
        
        if algorithm not in self.cluster_labels:
            raise ValueError(f"Clustering {algorithm} not fitted")
        
        # Reduce dimensions for visualization
        X_reduced = self.reduce_dimensions(X, method='pca', n_components=2)
        labels = self.cluster_labels[algorithm]
        
        # Create interactive plot
        fig = go.Figure()
        
        # Plot each cluster
        for cluster_id in np.unique(labels):
            if cluster_id == -1:
                cluster_name = 'Noise'
                color = 'black'
            else:
                cluster_name = f'Cluster {cluster_id}'
                color = px.colors.qualitative.Set1[cluster_id % len(px.colors.qualitative.Set1)]
            
            mask = labels == cluster_id
            
            fig.add_trace(go.Scatter(
                x=X_reduced[mask, 0],
                y=X_reduced[mask, 1],
                mode='markers',
                name=cluster_name,
                marker=dict(
                    color=color,
                    size=8,
                    opacity=0.7
                ),
                text=[f'Player: {idx}' for idx in X.index[mask]],
                hovertemplate='%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
            ))
        
        # Add cluster centers if available
        if algorithm in self.cluster_centers and algorithm != 'dbscan':
            centers = self.cluster_centers[algorithm]
            if self.pca is not None:
                centers_reduced = self.pca.transform(centers)
                
                fig.add_trace(go.Scatter(
                    x=centers_reduced[:, 0],
                    y=centers_reduced[:, 1],
                    mode='markers',
                    name='Cluster Centers',
                    marker=dict(
                        color='red',
                        size=15,
                        symbol='x',
                        line=dict(width=2, color='white')
                    ),
                    showlegend=True
                ))
        
        # Update layout
        fig.update_layout(
            title=f'Player Clustering - {algorithm.upper()}',
            xaxis_title='First Principal Component',
            yaxis_title='Second Principal Component',
            hovermode='closest',
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Cluster visualization saved to {save_path}")
        
        return fig
    
    def get_cluster_summary(self, algorithm: str = 'kmeans') -> Dict[str, Any]:
        """Get comprehensive cluster analysis summary."""
        
        if algorithm not in self.cluster_profiles:
            raise ValueError(f"Clustering {algorithm} not fitted")
        
        summary = {
            'algorithm': algorithm,
            'n_clusters': len(self.cluster_profiles[algorithm]),
            'metrics': self.clustering_metrics.get(algorithm, {}),
            'cluster_profiles': self.cluster_profiles[algorithm]
        }
        
        # Add cluster interpretations
        interpretations = self._interpret_clusters(algorithm)
        summary['interpretations'] = interpretations
        
        return summary
    
    def _interpret_clusters(self, algorithm: str) -> Dict[int, str]:
        """Generate human-readable interpretations of clusters."""
        
        profiles = self.cluster_profiles[algorithm]
        interpretations = {}
        
        for cluster_id, profile in profiles.items():
            top_features = profile['top_features']
            
            # Generate interpretation based on top features
            interpretation_parts = []
            
            for feature, info in list(top_features.items())[:3]:
                if 'foul' in feature.lower():
                    if info['cluster_mean'] > info['overall_mean']:
                        interpretation_parts.append("high fouling tendency")
                    else:
                        interpretation_parts.append("low fouling tendency")
                elif 'card' in feature.lower():
                    if info['cluster_mean'] > info['overall_mean']:
                        interpretation_parts.append("frequent cards")
                    else:
                        interpretation_parts.append("disciplined play")
                elif 'minute' in feature.lower():
                    if info['cluster_mean'] > info['overall_mean']:
                        interpretation_parts.append("high playing time")
                    else:
                        interpretation_parts.append("limited playing time")
            
            if interpretation_parts:
                interpretation = f"Players with {', '.join(interpretation_parts)}"
            else:
                interpretation = f"Cluster {cluster_id} - {profile['size']} players"
            
            interpretations[cluster_id] = interpretation
        
        return interpretations
    
    def save_clustering_results(self, directory: str):
        """Save clustering results and models."""
        
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save clustering models
        models_to_save = {}
        for algorithm, model in self.clustering_models.items():
            if hasattr(model, 'predict') or algorithm == 'dbscan':
                models_to_save[algorithm] = model
        
        if models_to_save:
            joblib.dump(models_to_save, directory / "clustering_models.joblib")
        
        # Save scaler
        joblib.dump(self.scaler, directory / "clustering_scaler.joblib")
        
        # Save results
        results = {
            'cluster_labels': {k: v.tolist() for k, v in self.cluster_labels.items()},
            'cluster_centers': {k: v.tolist() for k, v in self.cluster_centers.items()},
            'cluster_profiles': self.cluster_profiles,
            'clustering_metrics': self.clustering_metrics,
            'n_clusters': self.n_clusters
        }
        
        with open(directory / "clustering_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Clustering results saved to {directory}")
    
    def load_clustering_results(self, directory: str):
        """Load clustering results and models."""
        
        directory = Path(directory)
        
        # Load models
        models_path = directory / "clustering_models.joblib"
        if models_path.exists():
            self.clustering_models = joblib.load(models_path)
        
        # Load scaler
        scaler_path = directory / "clustering_scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
        
        # Load results
        results_path = directory / "clustering_results.json"
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            self.cluster_labels = {
                k: np.array(v) for k, v in results.get('cluster_labels', {}).items()
            }
            self.cluster_centers = {
                k: np.array(v) for k, v in results.get('cluster_centers', {}).items()
            }
            self.cluster_profiles = results.get('cluster_profiles', {})
            self.clustering_metrics = results.get('clustering_metrics', {})
            self.n_clusters = results.get('n_clusters', 5)
        
        logger.info(f"Clustering results loaded from {directory}")

class AnomalyDetection:
    """Anomaly detection for identifying unusual player behavior."""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.anomaly_models = {}
        self.anomaly_scores = {}
        self.anomaly_labels = {}
        
        # Initialize anomaly detection models
        self._initialize_anomaly_models()
    
    def _initialize_anomaly_models(self):
        """Initialize different anomaly detection algorithms."""
        
        self.anomaly_models = {
            'isolation_forest': IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'one_class_svm': OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='scale'
            ),
            'local_outlier_factor': LocalOutlierFactor(
                contamination=self.contamination,
                n_neighbors=20,
                n_jobs=-1
            )
        }
    
    def detect_anomalies(
        self, 
        X: pd.DataFrame, 
        algorithm: str = 'isolation_forest'
    ) -> Dict[str, Any]:
        """Detect anomalies in player behavior data."""
        
        logger.info(f"Detecting anomalies using {algorithm}...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model and predict
        model = self.anomaly_models[algorithm]
        
        if algorithm == 'local_outlier_factor':
            # LOF returns labels directly
            labels = model.fit_predict(X_scaled)
            scores = model.negative_outlier_factor_
        else:
            # Other models can be fitted separately
            model.fit(X_scaled)
            labels = model.predict(X_scaled)
            
            if hasattr(model, 'decision_function'):
                scores = model.decision_function(X_scaled)
            elif hasattr(model, 'score_samples'):
                scores = model.score_samples(X_scaled)
            else:
                scores = np.zeros(len(X_scaled))
        
        # Store results
        self.anomaly_labels[algorithm] = labels
        self.anomaly_scores[algorithm] = scores
        
        # Identify anomalies (label = -1)
        anomaly_mask = labels == -1
        n_anomalies = np.sum(anomaly_mask)
        
        # Get anomaly details
        anomaly_details = self._analyze_anomalies(
            X, anomaly_mask, scores, algorithm
        )
        
        results = {
            'algorithm': algorithm,
            'n_anomalies': int(n_anomalies),
            'anomaly_rate': float(n_anomalies / len(X)),
            'anomaly_indices': X.index[anomaly_mask].tolist(),
            'anomaly_scores': scores[anomaly_mask].tolist(),
            'anomaly_details': anomaly_details
        }
        
        logger.info(
            f"Detected {n_anomalies} anomalies ({n_anomalies/len(X)*100:.1f}%)"
        )
        
        return results
    
    def _analyze_anomalies(
        self, 
        X: pd.DataFrame, 
        anomaly_mask: np.ndarray, 
        scores: np.ndarray,
        algorithm: str
    ) -> List[Dict[str, Any]]:
        """Analyze detected anomalies in detail."""
        
        anomaly_data = X[anomaly_mask]
        anomaly_scores_subset = scores[anomaly_mask]
        
        details = []
        
        for idx, (player_id, player_data) in enumerate(anomaly_data.iterrows()):
            # Calculate how much each feature deviates from normal
            feature_deviations = {}
            
            for feature in X.columns:
                feature_mean = X[feature].mean()
                feature_std = X[feature].std()
                
                if feature_std > 0:
                    z_score = (player_data[feature] - feature_mean) / feature_std
                    feature_deviations[feature] = {
                        'value': float(player_data[feature]),
                        'mean': float(feature_mean),
                        'z_score': float(z_score),
                        'percentile': float(
                            (X[feature] < player_data[feature]).sum() / len(X) * 100
                        )
                    }
            
            # Identify most extreme features
            extreme_features = sorted(
                feature_deviations.items(),
                key=lambda x: abs(x[1]['z_score']),
                reverse=True
            )[:5]
            
            detail = {
                'player_id': player_id,
                'anomaly_score': float(anomaly_scores_subset[idx]),
                'extreme_features': dict(extreme_features),
                'summary': self._generate_anomaly_summary(extreme_features)
            }
            
            details.append(detail)
        
        return details
    
    def _generate_anomaly_summary(self, extreme_features: List[Tuple]) -> str:
        """Generate human-readable summary of anomaly."""
        
        if not extreme_features:
            return "Unusual overall pattern"
        
        summaries = []
        
        for feature, info in extreme_features[:3]:
            z_score = info['z_score']
            
            if abs(z_score) > 2:
                direction = "extremely high" if z_score > 0 else "extremely low"
                feature_name = feature.replace('_', ' ').replace('avg ', '').replace('per game', '')
                summaries.append(f"{direction} {feature_name}")
        
        if summaries:
            return f"Player with {', '.join(summaries)}"
        else:
            return "Unusual behavioral pattern"
    
    def visualize_anomalies(
        self, 
        X: pd.DataFrame, 
        algorithm: str = 'isolation_forest',
        save_path: str = None
    ) -> go.Figure:
        """Visualize detected anomalies."""
        
        if algorithm not in self.anomaly_labels:
            raise ValueError(f"Anomaly detection {algorithm} not fitted")
        
        # Reduce dimensions for visualization
        pca = PCA(n_components=2, random_state=self.random_state)
        X_scaled = self.scaler.transform(X)
        X_reduced = pca.fit_transform(X_scaled)
        
        labels = self.anomaly_labels[algorithm]
        scores = self.anomaly_scores[algorithm]
        
        # Create plot
        fig = go.Figure()
        
        # Normal points
        normal_mask = labels == 1
        fig.add_trace(go.Scatter(
            x=X_reduced[normal_mask, 0],
            y=X_reduced[normal_mask, 1],
            mode='markers',
            name='Normal',
            marker=dict(
                color='blue',
                size=6,
                opacity=0.6
            ),
            text=[f'Player: {idx}' for idx in X.index[normal_mask]],
            hovertemplate='%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
        ))
        
        # Anomaly points
        anomaly_mask = labels == -1
        if np.any(anomaly_mask):
            fig.add_trace(go.Scatter(
                x=X_reduced[anomaly_mask, 0],
                y=X_reduced[anomaly_mask, 1],
                mode='markers',
                name='Anomaly',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='x',
                    line=dict(width=2)
                ),
                text=[
                    f'Player: {idx}<br>Score: {score:.3f}' 
                    for idx, score in zip(X.index[anomaly_mask], scores[anomaly_mask])
                ],
                hovertemplate='%{text}<br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Anomaly Detection - {algorithm.replace("_", " ").title()}',
            xaxis_title='First Principal Component',
            yaxis_title='Second Principal Component',
            hovermode='closest',
            width=800,
            height=600
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Anomaly visualization saved to {save_path}")
        
        return fig
    
    def get_anomaly_summary(self, algorithm: str = 'isolation_forest') -> Dict[str, Any]:
        """Get comprehensive anomaly detection summary."""
        
        if algorithm not in self.anomaly_labels:
            raise ValueError(f"Anomaly detection {algorithm} not fitted")
        
        labels = self.anomaly_labels[algorithm]
        scores = self.anomaly_scores[algorithm]
        
        n_anomalies = np.sum(labels == -1)
        
        summary = {
            'algorithm': algorithm,
            'total_samples': len(labels),
            'n_anomalies': int(n_anomalies),
            'anomaly_rate': float(n_anomalies / len(labels)),
            'contamination_setting': self.contamination,
            'score_statistics': {
                'min': float(scores.min()),
                'max': float(scores.max()),
                'mean': float(scores.mean()),
                'std': float(scores.std())
            }
        }
        
        return summary