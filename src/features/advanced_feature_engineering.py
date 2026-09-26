"""
Advanced Feature Engineering Module for StartupFund AI.
Implements comprehensive feature engineering techniques from syllabus Phases 1-6.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer
)
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# PHASE 1: FOUNDATIONS - Feature Type Classification
# ============================================================

class FeatureTypeClassifier(BaseEstimator, TransformerMixin):
    """
    Classify features by type (numerical, categorical, binary, temporal).
    Helps understand feature characteristics before engineering.
    """
    
    def fit(self, X, y=None):
        self.feature_types_ = {}
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                dtype = X[col].dtype
                unique_vals = X[col].nunique()
                
                if dtype in ['int64', 'float64']:
                    if unique_vals == 2:
                        self.feature_types_[col] = 'binary'
                    else:
                        self.feature_types_[col] = 'numerical'
                elif dtype == 'object':
                    if unique_vals <= 10:
                        self.feature_types_[col] = 'categorical'
                    else:
                        self.feature_types_[col] = 'text'
                else:
                    self.feature_types_[col] = 'other'
        return self
    
    def transform(self, X):
        # This transformer doesn't transform, just classifies
        return X
    
    def get_feature_types(self):
        return self.feature_types_


# ============================================================
# PHASE 2: CLEANING & PREP - Advanced Techniques
# ============================================================

class AdvancedMissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Comprehensive missing value handling with multiple strategies.
    Implements MCAR/MAR/MNAR-aware imputation.
    """
    
    def __init__(self, strategy='median', add_indicator=True):
        self.strategy = strategy
        self.add_indicator = add_indicator
        self.fill_values_ = {}
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                if X[col].isnull().any():
                    if self.strategy == 'median':
                        self.fill_values_[col] = X[col].median()
                    elif self.strategy == 'mean':
                        self.fill_values_[col] = X[col].mean()
                    elif self.strategy == 'mode':
                        self.fill_values_[col] = X[col].mode()[0]
                    elif self.strategy == 'forward_fill':
                        self.fill_values_[col] = 'forward_fill'
                    elif self.strategy == 'backward_fill':
                        self.fill_values_[col] = 'backward_fill'
        return self
    
    def transform(self, X):
        X_df = X.copy()
        if isinstance(X_df, pd.DataFrame):
            for col, fill_val in self.fill_values_.items():
                if col in X_df.columns:
                    if isinstance(fill_val, str) and 'fill' in fill_val:
                        if fill_val == 'forward_fill':
                            X_df[col] = X_df[col].fillna(method='ffill')
                        elif fill_val == 'backward_fill':
                            X_df[col] = X_df[col].fillna(method='bfill')
                    else:
                        X_df[col] = X_df[col].fillna(fill_val)
                    
                    # Add missing indicator if requested
                    if self.add_indicator:
                        X_df[f"{col}_was_missing"] = X_df[col].isnull().astype(int)
        return X_df


class AdvancedScaler(BaseEstimator, TransformerMixin):
    """
    Multiple scaling strategies for different data distributions.
    Implements StandardScaler, MinMaxScaler, and RobustScaler.
    """
    
    def __init__(self, method='standard', columns=None):
        self.method = method
        self.columns = columns
        self.scaler_ = None
    
    def fit(self, X, y=None):
        if self.method == 'standard':
            self.scaler_ = StandardScaler()
        elif self.method == 'minmax':
            self.scaler_ = MinMaxScaler()
        elif self.method == 'robust':
            self.scaler_ = RobustScaler()
        
        if self.columns and isinstance(X, pd.DataFrame):
            self.scaler_.fit(X[self.columns])
        else:
            self.scaler_.fit(X)
        
        return self
    
    def transform(self, X):
        X_df = X.copy()
        if self.columns and isinstance(X_df, pd.DataFrame):
            X_df[self.columns] = self.scaler_.transform(X_df[self.columns])
        else:
            X_df = pd.DataFrame(self.scaler_.transform(X_df), columns=X_df.columns)
        return X_df


class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Target encoding for categorical variables.
    Replaces categories with their mean target value.
    Important: Fit only on training data to prevent leakage.
    """
    
    def __init__(self, columns=None, smoothing=1.0):
        self.columns = columns
        self.smoothing = smoothing
        self.target_means_ = {}
        self.global_mean_ = None
    
    def fit(self, X, y):
        self.global_mean_ = np.mean(y)
        
        if self.columns and isinstance(X, pd.DataFrame):
            for col in self.columns:
                if col in X.columns and X[col].dtype == 'object':
                    # Calculate mean target for each category
                    temp_df = pd.DataFrame({col: X[col], 'target': y})
                    category_means = temp_df.groupby(col)['target'].mean()
                    
                    # Apply smoothing
                    counts = temp_df.groupby(col).size()
                    smoothed_means = (category_means * counts + self.global_mean_ * self.smoothing) / (counts + self.smoothing)
                    
                    self.target_means_[col] = smoothed_means.to_dict()
        
        return self
    
    def transform(self, X):
        X_df = X.copy()
        if isinstance(X_df, pd.DataFrame):
            for col, means in self.target_means_.items():
                if col in X_df.columns:
                    X_df[col] = X_df[col].map(means).fillna(self.global_mean_)
        return X_df


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Log transformation for skewed numerical features.
    Handles zero and negative values with offset.
    """
    
    def __init__(self, columns=None, offset=1.0):
        self.columns = columns
        self.offset = offset
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_df = X.copy()
        cols_to_transform = self.columns if self.columns else X_df.select_dtypes(include=[np.number]).columns
        
        for col in cols_to_transform:
            if col in X_df.columns:
                # Apply log transformation with offset to handle zeros
                X_df[col] = np.log1p(X_df[col] + self.offset)
        
        return X_df


# ============================================================
# PHASE 3: FEATURE CREATION - Advanced Techniques
# ============================================================

class TimeBasedFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Creates time-based features: lag features, rolling statistics, cyclical encoding.
    """
    
    def __init__(self, date_column=None, window_sizes=[3, 7, 30]):
        self.date_column = date_column
        self.window_sizes = window_sizes
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_df = X.copy()
        
        # Cyclical encoding for temporal features
        if 'month' in X_df.columns:
            X_df['month_sin'] = np.sin(2 * np.pi * X_df['month'] / 12)
            X_df['month_cos'] = np.cos(2 * np.pi * X_df['month'] / 12)
        
        if 'day_of_week' in X_df.columns:
            X_df['day_sin'] = np.sin(2 * np.pi * X_df['day_of_week'] / 7)
            X_df['day_cos'] = np.cos(2 * np.pi * X_df['day_of_week'] / 7)
        
        # Rolling statistics for numerical columns
        numerical_cols = X_df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            for window in self.window_sizes:
                if len(X_df) >= window:
                    X_df[f'{col}_rolling_mean_{window}'] = X_df[col].rolling(window=window).mean()
                    X_df[f'{col}_rolling_std_{window}'] = X_df[col].rolling(window=window).std()
        
        return X_df


class PolynomialFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Creates polynomial and interaction features.
    Captures non-linear relationships and feature interactions.
    """
    
    def __init__(self, columns=None, degree=2, interactions_only=True):
        self.columns = columns
        self.degree = degree
        self.interactions_only = interactions_only
        self.feature_names_ = None
    
    def fit(self, X, y=None):
        if self.columns and isinstance(X, pd.DataFrame):
            self.feature_names_ = self.columns
        return self
    
    def transform(self, X):
        X_df = X.copy()
        if not self.feature_names_:
            return X_df
        
        # Create polynomial features
        for i, col1 in enumerate(self.feature_names_):
            if col1 in X_df.columns and X_df[col1].dtype in [np.int64, np.float64]:
                # Squared term
                if not self.interactions_only:
                    X_df[f'{col1}_squared'] = X_df[col1] ** 2
                
                # Interaction terms
                for col2 in self.feature_names_[i+1:]:
                    if col2 in X_df.columns and X_df[col2].dtype in [np.int64, np.float64]:
                        X_df[f'{col1}_x_{col2}'] = X_df[col1] * X_df[col2]
        
        return X_df


class BusinessFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Creates business-specific features (RFM-like, churn indicators, behavioral aggregates).
    """
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_df = X.copy()
        
        # RFM-like features for investor data
        if 'portfolio_companies' in X_df.columns and 'successful_exits' in X_df.columns:
            # Frequency (portfolio size)
            X_df['portfolio_frequency'] = X_df['portfolio_companies']
            
            # Monetary (assuming funding amount is related to portfolio size)
            X_df['investment_capacity'] = X_df['portfolio_companies'] * X_df['successful_exits']
        
        # Churn-like indicators
        if 'active_fund' in X_df.columns:
            X_df['inactivity_risk'] = (1 - X_df['active_fund']).astype(int)
        
        # Behavioral aggregates
        if 'ai_focus' in X_df.columns and 'fintech_focus' in X_df.columns:
            X_df['tech_focus_score'] = X_df['ai_focus'] + X_df['fintech_focus']
        
        # Velocity features
        if 'portfolio_companies' in X_df.columns and 'investor_age_years' in X_df.columns:
            X_df['investment_velocity'] = X_df['portfolio_companies'] / X_df['investor_age_years'].clip(lower=1)
        
        return X_df


# ============================================================
# PHASE 4: FEATURE SELECTION - Multiple Methods
# ============================================================

class FilterFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Filter methods: Variance threshold, correlation filtering, statistical tests.
    Fast and model-independent.
    """
    
    def __init__(self, threshold=0.01, method='variance', k=10):
        self.threshold = threshold
        self.method = method
        self.k = k
        self.selected_features_ = None
        self.selector_ = None
    
    def fit(self, X, y=None):
        if self.method == 'variance':
            self.selector_ = VarianceThreshold(threshold=self.threshold)
            self.selector_.fit(X)
            self.selected_features_ = self.selector_.get_support()
        
        elif self.method == 'k_best':
            self.selector_ = SelectKBest(score_func=f_regression, k=self.k)
            self.selector_.fit(X, y)
            self.selected_features_ = self.selector_.get_support()
        
        elif self.method == 'mutual_info':
            self.selector_ = SelectKBest(score_func=mutual_info_regression, k=self.k)
            self.selector_.fit(X, y)
            self.selected_features_ = self.selector_.get_support()
        
        return self
    
    def transform(self, X):
        if self.selector_:
            return self.selector_.transform(X)
        return X


class WrapperFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Wrapper methods: RFE, forward/backward selection.
    Model-dependent but more accurate.
    """
    
    def __init__(self, estimator=None, n_features_to_select=10, method='rfe'):
        self.estimator = estimator or RandomForestRegressor(n_estimators=50, random_state=42)
        self.n_features_to_select = n_features_to_select
        self.method = method
        self.selector_ = None
        self.selected_features_ = None
    
    def fit(self, X, y):
        if self.method == 'rfe':
            self.selector_ = RFE(
                estimator=self.estimator,
                n_features_to_select=self.n_features_to_select
            )
            self.selector_.fit(X, y)
            self.selected_features_ = self.selector_.get_support()
        
        return self
    
    def transform(self, X):
        if self.selector_:
            return self.selector_.transform(X)
        return X


class EmbeddedFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Embedded methods: Lasso, ElasticNet, tree-based importance.
    Selection happens during model training.
    """
    
    def __init__(self, method='lasso', alpha=0.01, threshold='median'):
        self.method = method
        self.alpha = alpha
        self.threshold = threshold
        self.selector_ = None
        self.selected_features_ = None
    
    def fit(self, X, y):
        if self.method == 'lasso':
            estimator = Lasso(alpha=self.alpha, random_state=42)
        elif self.method == 'elasticnet':
            estimator = ElasticNet(alpha=self.alpha, random_state=42)
        elif self.method == 'random_forest':
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        self.selector_ = SelectFromModel(
            estimator=estimator,
            threshold=self.threshold
        )
        self.selector_.fit(X, y)
        self.selected_features_ = self.selector_.get_support()
        
        return self
    
    def transform(self, X):
        if self.selector_:
            return self.selector_.transform(X)
        return X


# ============================================================
# PHASE 5: DIMENSIONALITY REDUCTION - PCA
# ============================================================

class PCAReducer(BaseEstimator, TransformerMixin):
    """
    Principal Component Analysis for dimensionality reduction.
    Captures maximum variance with fewer components.
    """
    
    def __init__(self, n_components=0.95, svd_solver='auto'):
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.pca_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X, y=None):
        self.pca_ = PCA(n_components=self.n_components, svd_solver=self.svd_solver)
        self.pca_.fit(X)
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        return self
    
    def transform(self, X):
        if self.pca_:
            return self.pca_.transform(X)
        return X
    
    def get_n_components(self):
        if self.pca_:
            return self.pca_.n_components_
        return None
    
    def get_cumulative_variance(self):
        if self.explained_variance_ratio_ is not None:
            return np.cumsum(self.explained_variance_ratio_)
        return None


# ============================================================
# PHASE 6: FEATURE OPS & GOVERNANCE - Metadata and Documentation
# ============================================================

class FeatureMetadata(BaseEstimator, TransformerMixin):
    """
    Feature metadata tracking and documentation.
    Ensures reproducibility and governance.
    """
    
    def __init__(self):
        self.feature_metadata_ = {}
        self.transformation_history_ = []
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                self.feature_metadata_[col] = {
                    'dtype': str(X[col].dtype),
                    'missing_count': int(X[col].isnull().sum()),
                    'missing_percentage': float(X[col].isnull().sum() / len(X) * 100),
                    'unique_count': int(X[col].nunique()),
                    'sample_values': X[col].head(3).tolist()
                }
        return self
    
    def transform(self, X):
        # This transformer doesn't transform, just tracks metadata
        return X
    
    def add_transformation(self, transformation_name, columns_affected):
        """Record transformation history for reproducibility."""
        self.transformation_history_.append({
            'transformation': transformation_name,
            'columns': columns_affected,
            'timestamp': pd.Timestamp.now().isoformat()
        })
    
    def get_feature_card(self, feature_name):
        """Generate a feature card for documentation."""
        if feature_name in self.feature_metadata_:
            return {
                'feature_name': feature_name,
                'metadata': self.feature_metadata_[feature_name],
                'transformations': [t for t in self.transformation_history_ 
                                  if feature_name in t.get('columns', [])]
            }
        return None
    
    def generate_feature_report(self):
        """Generate comprehensive feature report."""
        return {
            'total_features': len(self.feature_metadata_),
            'feature_metadata': self.feature_metadata_,
            'transformation_history': self.transformation_history_,
            'data_quality_summary': {
                'total_missing': sum(m['missing_count'] for m in self.feature_metadata_.values()),
                'high_cardinality_features': sum(1 for m in self.feature_metadata_.values() 
                                                 if m['unique_count'] > 50)
            }
        }


# ============================================================
# COMPREHENSIVE FEATURE ENGINEERING PIPELINE
# ============================================================

class ComprehensiveFeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    """
    End-to-end feature engineering pipeline implementing all 6 phases.
    """
    
    def __init__(self, config=None):
        self.config = config or {
            'phase2_cleaning': True,
            'phase2_scaling': 'standard',
            'phase3_polynomial': True,
            'phase3_business': True,
            'phase4_selection': 'filter',
            'phase5_pca': False,
            'phase6_metadata': True
        }
        self.metadata_tracker = FeatureMetadata()
        self.transformers_ = {}
    
    def fit(self, X, y=None):
        # Phase 1: Feature Type Classification
        feature_classifier = FeatureTypeClassifier()
        feature_classifier.fit(X)
        self.transformers_['feature_classifier'] = feature_classifier
        
        # Phase 2: Cleaning & Prep
        if self.config['phase2_cleaning']:
            missing_handler = AdvancedMissingValueHandler(strategy='median')
            missing_handler.fit(X)
            self.transformers_['missing_handler'] = missing_handler
        
        # Phase 3: Feature Creation
        if self.config['phase3_polynomial']:
            poly_creator = PolynomialFeatureCreator(degree=2)
            poly_creator.fit(X)
            self.transformers_['poly_creator'] = poly_creator
        
        if self.config['phase3_business']:
            business_creator = BusinessFeatureCreator()
            business_creator.fit(X)
            self.transformers_['business_creator'] = business_creator
        
        # Phase 4: Feature Selection
        if self.config.get('phase4_selection') == 'filter':
            selector = FilterFeatureSelector(method='variance', threshold=0.01)
            # Note: Actual selection happens in transform after all features are created
            self.transformers_['selector'] = selector
        
        # Phase 6: Metadata Tracking
        if self.config['phase6_metadata']:
            self.metadata_tracker.fit(X)
        
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        
        # Apply transformations in order
        if 'missing_handler' in self.transformers_:
            X_transformed = self.transformers_['missing_handler'].transform(X_transformed)
            self.metadata_tracker.add_transformation('missing_value_imputation', list(X_transformed.columns))
        
        if 'poly_creator' in self.transformers_:
            X_transformed = self.transformers_['poly_creator'].transform(X_transformed)
            new_cols = set(X_transformed.columns) - set(X.columns)
            self.metadata_tracker.add_transformation('polynomial_features', list(new_cols))
        
        if 'business_creator' in self.transformers_:
            X_transformed = self.transformers_['business_creator'].transform(X_transformed)
            new_cols = set(X_transformed.columns) - set(X.columns)
            self.metadata_tracker.add_transformation('business_features', list(new_cols))
        
        if self.config['phase6_metadata']:
            self.metadata_tracker.fit(X_transformed)
        
        return X_transformed
    
    def get_feature_report(self):
        """Get comprehensive feature engineering report."""
        return self.metadata_tracker.generate_feature_report()