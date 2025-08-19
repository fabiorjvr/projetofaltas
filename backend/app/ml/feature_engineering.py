import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FootballFeatureEngineer:
    """Feature engineering for football fouls prediction."""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.feature_names = []
        self.target_columns = ['fouls', 'yellow_cards', 'red_cards']
        
    def create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create base statistical features from player data."""
        logger.info("Creating base features...")
        
        df = df.copy()
        
        # Basic rate features
        df['fouls_per_game'] = df['fouls'] / np.maximum(df['appearances'], 1)
        df['fouls_drawn_per_game'] = df['fouls_drawn'] / np.maximum(df['appearances'], 1)
        df['yellow_cards_per_game'] = df['yellow_cards'] / np.maximum(df['appearances'], 1)
        df['red_cards_per_game'] = df['red_cards'] / np.maximum(df['appearances'], 1)
        df['minutes_per_game'] = df['minutes'] / np.maximum(df['appearances'], 1)
        
        # Efficiency features
        df['fouls_per_minute'] = df['fouls'] / np.maximum(df['minutes'], 1) * 90
        df['cards_per_minute'] = (df['yellow_cards'] + df['red_cards']) / np.maximum(df['minutes'], 1) * 90
        df['fouls_drawn_per_minute'] = df['fouls_drawn'] / np.maximum(df['minutes'], 1) * 90
        
        # Disciplinary features
        df['total_cards'] = df['yellow_cards'] + df['red_cards']
        df['card_to_foul_ratio'] = df['total_cards'] / np.maximum(df['fouls'], 1)
        df['foul_balance'] = df['fouls_drawn'] - df['fouls']
        df['disciplinary_index'] = (df['yellow_cards'] + df['red_cards'] * 2) / np.maximum(df['appearances'], 1)
        
        # Playing time features
        df['minutes_percentage'] = df['minutes'] / np.maximum(df['appearances'] * 90, 1)
        df['substitute_indicator'] = (df['minutes_percentage'] < 0.7).astype(int)
        df['starter_indicator'] = (df['minutes_percentage'] > 0.8).astype(int)
        
        # Aggression features
        df['aggression_score'] = (
            df['fouls_per_game'] * 0.4 + 
            df['yellow_cards_per_game'] * 0.4 + 
            df['red_cards_per_game'] * 0.2
        )
        
        # Fair play features
        df['fair_play_score'] = (
            df['fouls_drawn_per_game'] * 0.6 - 
            df['fouls_per_game'] * 0.3 - 
            df['total_cards'] / np.maximum(df['appearances'], 1) * 0.1
        )
        
        return df
    
    def create_positional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create position-based features."""
        logger.info("Creating positional features...")
        
        df = df.copy()
        
        # Position encoding
        position_mapping = {
            'GK': 0, 'DF': 1, 'MF': 2, 'FW': 3,
            'DF,MF': 1.5, 'MF,FW': 2.5, 'DF,FW': 2
        }
        
        df['position_numeric'] = df['position'].map(position_mapping).fillna(2)
        
        # Position-specific features
        df['is_goalkeeper'] = (df['position'] == 'GK').astype(int)
        df['is_defender'] = df['position'].str.contains('DF', na=False).astype(int)
        df['is_midfielder'] = df['position'].str.contains('MF', na=False).astype(int)
        df['is_forward'] = df['position'].str.contains('FW', na=False).astype(int)
        df['is_versatile'] = df['position'].str.contains(',', na=False).astype(int)
        
        # Position-based expected behavior
        position_foul_expectations = {
            'GK': 0.2, 'DF': 1.5, 'MF': 1.2, 'FW': 0.8,
            'DF,MF': 1.35, 'MF,FW': 1.0, 'DF,FW': 1.15
        }
        
        df['expected_fouls_per_game'] = df['position'].map(position_foul_expectations).fillna(1.0)
        df['fouls_vs_expected'] = df['fouls_per_game'] - df['expected_fouls_per_game']
        
        return df
    
    def create_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create team-based features."""
        logger.info("Creating team features...")
        
        df = df.copy()
        
        # Team aggregations
        team_stats = df.groupby(['team', 'season']).agg({
            'fouls': ['mean', 'std', 'sum'],
            'yellow_cards': ['mean', 'std', 'sum'],
            'red_cards': ['mean', 'std', 'sum'],
            'fouls_drawn': ['mean', 'std', 'sum'],
            'appearances': 'mean'
        }).round(3)
        
        team_stats.columns = ['_'.join(col).strip() for col in team_stats.columns]
        team_stats = team_stats.add_prefix('team_')
        
        # Merge team stats
        df = df.merge(
            team_stats.reset_index(),
            on=['team', 'season'],
            how='left'
        )
        
        # Player vs team comparisons
        df['fouls_vs_team_avg'] = df['fouls'] - df['team_fouls_mean']
        df['cards_vs_team_avg'] = df['total_cards'] - (df['team_yellow_cards_mean'] + df['team_red_cards_mean'])
        df['player_team_foul_ratio'] = df['fouls'] / np.maximum(df['team_fouls_sum'], 1)
        
        # Team discipline ranking
        df['team_discipline_rank'] = df.groupby('season')['team_fouls_mean'].rank(ascending=True)
        df['team_aggression_rank'] = df.groupby('season')['team_fouls_mean'].rank(ascending=False)
        
        return df
    
    def create_league_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create league-based features."""
        logger.info("Creating league features...")
        
        df = df.copy()
        
        # League aggregations
        league_stats = df.groupby(['league', 'season']).agg({
            'fouls': ['mean', 'std'],
            'yellow_cards': ['mean', 'std'],
            'red_cards': ['mean', 'std'],
            'fouls_drawn': ['mean', 'std']
        }).round(3)
        
        league_stats.columns = ['_'.join(col).strip() for col in league_stats.columns]
        league_stats = league_stats.add_prefix('league_')
        
        # Merge league stats
        df = df.merge(
            league_stats.reset_index(),
            on=['league', 'season'],
            how='left'
        )
        
        # League competitiveness features
        df['league_competitiveness'] = df['league_fouls_std'] / np.maximum(df['league_fouls_mean'], 0.1)
        df['player_vs_league_fouls'] = (df['fouls_per_game'] - df['league_fouls_mean']) / np.maximum(df['league_fouls_std'], 0.1)
        
        # League encoding
        league_aggression_mapping = {
            'Premier League': 1.2,
            'La Liga': 1.1,
            'Serie A': 1.3,
            'Bundesliga': 1.0,
            'Ligue 1': 1.15,
            'Brasileirão': 1.4
        }
        
        df['league_aggression_factor'] = df['league'].map(league_aggression_mapping).fillna(1.0)
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        logger.info("Creating temporal features...")
        
        df = df.copy()
        
        # Season features
        df['season_start_year'] = df['season'].str[:4].astype(int)
        df['season_numeric'] = df['season_start_year'] - 2000  # Normalize to smaller numbers
        
        # Career stage features (approximate)
        df['estimated_age'] = 25  # Default age, would need actual age data
        df['career_stage'] = pd.cut(
            df['estimated_age'],
            bins=[0, 23, 28, 32, 50],
            labels=['young', 'prime', 'experienced', 'veteran']
        )
        
        # Experience proxy (appearances)
        df['experience_level'] = pd.cut(
            df['appearances'],
            bins=[0, 10, 25, 35, 100],
            labels=['rookie', 'developing', 'regular', 'veteran']
        )
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different variables."""
        logger.info("Creating interaction features...")
        
        df = df.copy()
        
        # Position-league interactions
        df['defender_in_aggressive_league'] = df['is_defender'] * df['league_aggression_factor']
        df['midfielder_minutes_interaction'] = df['is_midfielder'] * df['minutes_per_game']
        
        # Playing time interactions
        df['starter_foul_interaction'] = df['starter_indicator'] * df['fouls_per_game']
        df['substitute_aggression'] = df['substitute_indicator'] * df['aggression_score']
        
        # Team-player interactions
        df['star_player_indicator'] = (df['minutes_per_game'] > df['team_appearances_mean'] * 0.8).astype(int)
        df['team_outlier_fouls'] = df['star_player_indicator'] * df['fouls_vs_team_avg']
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, window_sizes: List[int] = [3, 5]) -> pd.DataFrame:
        """Create rolling window features for time series analysis."""
        logger.info("Creating rolling features...")
        
        df = df.copy()
        df = df.sort_values(['player_name', 'season_start_year'])
        
        for window in window_sizes:
            for col in ['fouls_per_game', 'yellow_cards_per_game', 'aggression_score']:
                if col in df.columns:
                    df[f'{col}_rolling_{window}'] = (
                        df.groupby('player_name')[col]
                        .rolling(window=window, min_periods=1)
                        .mean()
                        .reset_index(0, drop=True)
                    )
                    
                    df[f'{col}_trend_{window}'] = (
                        df.groupby('player_name')[col]
                        .rolling(window=window, min_periods=2)
                        .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
                        .reset_index(0, drop=True)
                    )
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features."""
        logger.info("Encoding categorical features...")
        
        df = df.copy()
        
        categorical_columns = ['team', 'league', 'career_stage', 'experience_level']
        
        for col in categorical_columns:
            if col in df.columns:
                if fit:
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.encoders:
                        # Handle unseen categories
                        unique_values = set(df[col].astype(str).unique())
                        known_values = set(self.encoders[col].classes_)
                        
                        if not unique_values.issubset(known_values):
                            # Add unknown category
                            unknown_mask = ~df[col].astype(str).isin(known_values)
                            df.loc[unknown_mask, col] = 'unknown'
                            
                            # Update encoder if needed
                            if 'unknown' not in known_values:
                                self.encoders[col].classes_ = np.append(self.encoders[col].classes_, 'unknown')
                        
                        df[f'{col}_encoded'] = self.encoders[col].transform(df[col].astype(str))
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        logger.info("Scaling features...")
        
        df = df.copy()
        
        # Features to scale
        scale_features = [
            'fouls_per_game', 'fouls_drawn_per_game', 'yellow_cards_per_game',
            'minutes_per_game', 'aggression_score', 'fair_play_score',
            'fouls_vs_expected', 'fouls_vs_team_avg', 'player_vs_league_fouls'
        ]
        
        # Add rolling features
        scale_features.extend([col for col in df.columns if 'rolling' in col or 'trend' in col])
        
        # Filter existing columns
        scale_features = [col for col in scale_features if col in df.columns]
        
        if scale_features:
            if fit:
                self.scalers['standard'] = StandardScaler()
                df[scale_features] = self.scalers['standard'].fit_transform(df[scale_features])
            else:
                if 'standard' in self.scalers:
                    df[scale_features] = self.scalers['standard'].transform(df[scale_features])
        
        return df
    
    def select_features(self, df: pd.DataFrame, target_col: str, k: int = 50, fit: bool = True) -> pd.DataFrame:
        """Select best features using statistical tests."""
        logger.info(f"Selecting top {k} features for {target_col}...")
        
        # Get feature columns (exclude target and identifier columns)
        exclude_cols = [
            'player_name', 'team', 'league', 'season', 'position',
            'fouls', 'yellow_cards', 'red_cards', 'fouls_drawn',
            'appearances', 'minutes', 'source', 'scraped_at',
            'created_at', 'updated_at', 'id'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols and col != target_col]
        
        if not feature_cols:
            logger.warning("No feature columns found for selection")
            return df
        
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        if fit:
            selector_key = f'{target_col}_selector'
            self.feature_selectors[selector_key] = SelectKBest(
                score_func=f_regression,
                k=min(k, len(feature_cols))
            )
            
            X_selected = self.feature_selectors[selector_key].fit_transform(X, y)
            selected_features = X.columns[self.feature_selectors[selector_key].get_support()].tolist()
            
        else:
            selector_key = f'{target_col}_selector'
            if selector_key in self.feature_selectors:
                X_selected = self.feature_selectors[selector_key].transform(X)
                selected_features = X.columns[self.feature_selectors[selector_key].get_support()].tolist()
            else:
                selected_features = feature_cols[:k]  # Fallback
        
        # Store selected feature names
        self.feature_names = selected_features
        
        # Return dataframe with selected features plus target and identifiers
        keep_cols = selected_features + [target_col] + [
            'player_name', 'team', 'league', 'season', 'position'
        ]
        keep_cols = [col for col in keep_cols if col in df.columns]
        
        return df[keep_cols]
    
    def create_all_features(self, df: pd.DataFrame, target_col: str = 'fouls', fit: bool = True) -> pd.DataFrame:
        """Create all features in the correct order."""
        logger.info("Starting comprehensive feature engineering...")
        
        # Create features step by step
        df = self.create_base_features(df)
        df = self.create_positional_features(df)
        df = self.create_team_features(df)
        df = self.create_league_features(df)
        df = self.create_temporal_features(df)
        df = self.create_interaction_features(df)
        df = self.create_rolling_features(df)
        
        # Encode categorical features
        df = self.encode_categorical_features(df, fit=fit)
        
        # Scale numerical features
        df = self.scale_features(df, fit=fit)
        
        # Select best features
        df = self.select_features(df, target_col, fit=fit)
        
        logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        logger.info(f"Selected features: {len(self.feature_names)}")
        
        return df
    
    def get_feature_importance_analysis(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Analyze feature importance and correlations."""
        feature_cols = [col for col in self.feature_names if col in df.columns]
        
        if not feature_cols:
            return {}
        
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        # Correlation analysis
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        
        # Mutual information
        mi_scores = mutual_info_regression(X, y)
        mi_df = pd.DataFrame({
            'feature': feature_cols,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)
        
        return {
            'correlations': correlations.to_dict(),
            'mutual_information': mi_df.to_dict('records'),
            'feature_count': len(feature_cols),
            'target_stats': {
                'mean': float(y.mean()),
                'std': float(y.std()),
                'min': float(y.min()),
                'max': float(y.max())
            }
        }
    
    def save_feature_engineering_artifacts(self, filepath: str):
        """Save feature engineering components for later use."""
        import joblib
        
        artifacts = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_selectors': self.feature_selectors,
            'feature_names': self.feature_names
        }
        
        joblib.dump(artifacts, filepath)
        logger.info(f"Feature engineering artifacts saved to {filepath}")
    
    def load_feature_engineering_artifacts(self, filepath: str):
        """Load feature engineering components."""
        import joblib
        
        artifacts = joblib.load(filepath)
        
        self.scalers = artifacts.get('scalers', {})
        self.encoders = artifacts.get('encoders', {})
        self.feature_selectors = artifacts.get('feature_selectors', {})
        self.feature_names = artifacts.get('feature_names', [])
        
        logger.info(f"Feature engineering artifacts loaded from {filepath}")

def prepare_ml_dataset(
    df: pd.DataFrame, 
    target_column: str = 'fouls',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare dataset for machine learning."""
    from sklearn.model_selection import train_test_split
    
    # Initialize feature engineer
    fe = FootballFeatureEngineer()
    
    # Create features
    df_features = fe.create_all_features(df, target_col=target_column, fit=True)
    
    # Prepare X and y
    feature_cols = [col for col in fe.feature_names if col in df_features.columns]
    X = df_features[feature_cols].fillna(0)
    y = df_features[target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    logger.info(f"Dataset prepared: Train {X_train.shape}, Test {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, fe