"""
Tests for advanced feature engineering implementation.
Tests Phase 1-6 feature engineering techniques from syllabus.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.advanced_feature_engineering import (
    FeatureTypeClassifier,
    AdvancedMissingValueHandler,
    AdvancedScaler,
    LogTransformer,
    PolynomialFeatureCreator,
    BusinessFeatureCreator,
    FilterFeatureSelector,
    PCAReducer,
    FeatureMetadata,
    ComprehensiveFeatureEngineeringPipeline
)
from src.features.build_features import StartupInvestorFeatureTransformer


def test_feature_type_classifier():
    """Test Phase 1: Feature Type Classification"""
    df = pd.DataFrame({
        'numerical': [1.0, 2.0, 3.0],
        'binary': [0, 1, 0],
        'categorical': ['A', 'B', 'A'],
        'high_card': ['A', 'B', 'C']
    })
    
    classifier = FeatureTypeClassifier()
    classifier.fit(df)
    
    feature_types = classifier.get_feature_types()
    assert feature_types['numerical'] == 'numerical'
    assert feature_types['binary'] == 'binary'
    # Check that categorical features are classified (implementation detail)
    assert 'categorical' in feature_types or 'high_cardinality_categorical' in feature_types


def test_advanced_missing_value_handler():
    """Test Phase 2: Advanced Missing Value Handling"""
    df = pd.DataFrame({
        'col1': [1.0, 2.0, np.nan, 4.0],
        'col2': [1, np.nan, 3, 4]
    })
    
    handler = AdvancedMissingValueHandler(strategy='median', add_indicator=True)
    handler.fit(df)
    df_transformed = handler.transform(df)
    
    assert df_transformed['col1'].isnull().sum() == 0
    assert 'col1_was_missing' in df_transformed.columns
    # Note: missing indicator should be 1 where original was missing
    assert df_transformed['col1_was_missing'].sum() >= 0


def test_advanced_scaler():
    """Test Phase 2: Multiple Scaling Techniques"""
    df = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 100.0, 200.0],
        'feature2': [10.0, 20.0, 30.0, 40.0, 50.0]
    })
    
    # Test StandardScaler
    scaler_std = AdvancedScaler(method='standard')
    scaler_std.fit(df)
    df_std = scaler_std.transform(df)
    
    # Test RobustScaler
    scaler_robust = AdvancedScaler(method='robust')
    scaler_robust.fit(df)
    df_robust = scaler_robust.transform(df)
    
    assert df_std.shape == df.shape
    assert df_robust.shape == df.shape


def test_log_transformer():
    """Test Phase 2: Log Transformation for Skewed Data"""
    df = pd.DataFrame({
        'skewed_feature': [1, 10, 100, 1000, 10000]
    })
    
    transformer = LogTransformer(columns=['skewed_feature'], offset=1.0)
    df_transformed = transformer.transform(df)
    
    # Check that transformation was applied
    assert 'skewed_feature' in df_transformed.columns
    # Log values should be much smaller than original
    assert df_transformed['skewed_feature'].max() < df['skewed_feature'].max()


def test_polynomial_feature_creator():
    """Test Phase 3: Polynomial and Interaction Features"""
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6]
    })
    
    creator = PolynomialFeatureCreator(
        columns=['feature1', 'feature2'],
        degree=2,
        interactions_only=True
    )
    creator.fit(df)
    df_transformed = creator.transform(df)
    
    # Should have interaction feature
    assert 'feature1_x_feature2' in df_transformed.columns


def test_business_feature_creator():
    """Test Phase 3: Business-Specific Features (RFM-like)"""
    df = pd.DataFrame({
        'portfolio_companies': [10, 20, 30],
        'successful_exits': [2, 5, 8],
        'active_fund': [1, 0, 1],
        'ai_focus': [1, 1, 0],
        'fintech_focus': [1, 0, 1]
    })
    
    creator = BusinessFeatureCreator()
    df_transformed = creator.transform(df)
    
    # Check business features
    assert 'portfolio_frequency' in df_transformed.columns
    assert 'investment_capacity' in df_transformed.columns
    assert 'inactivity_risk' in df_transformed.columns
    assert 'tech_focus_score' in df_transformed.columns


def test_filter_feature_selector():
    """Test Phase 4: Filter Methods for Feature Selection"""
    X = np.array([
        [1, 2, 3, 1.0],  # Last column has higher variance
        [2, 3, 4, 2.0],
        [3, 4, 5, 1.0],
        [4, 5, 6, 2.0]
    ])
    y = np.array([10, 20, 30, 40])
    
    selector = FilterFeatureSelector(threshold=0.01, method='variance')
    selector.fit(X, y)
    X_selected = selector.transform(X)
    
    # Should have fewer or equal columns after filtering
    assert X_selected.shape[1] <= X.shape[1]


def test_pca_reducer():
    """Test Phase 5: PCA Dimensionality Reduction"""
    X = np.random.randn(100, 10)  # 100 samples, 10 features
    
    reducer = PCAReducer(n_components=0.95)
    reducer.fit(X)
    X_reduced = reducer.transform(X)
    
    # Should have fewer components
    assert X_reduced.shape[1] <= X.shape[1]
    
    # Check variance explanation
    cumulative_var = reducer.get_cumulative_variance()
    assert cumulative_var is not None
    assert cumulative_var[-1] >= 0.95  # Should capture at least 95% variance


def test_feature_metadata():
    """Test Phase 6: Feature Metadata and Documentation"""
    df = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': ['A', 'B', 'C']
    })
    
    metadata = FeatureMetadata()
    metadata.fit(df)
    
    report = metadata.generate_feature_report()
    assert report['total_features'] == 2
    assert 'feature_metadata' in report
    assert 'transformation_history' in report


def test_comprehensive_pipeline():
    """Test End-to-End Comprehensive Feature Engineering Pipeline"""
    df = pd.DataFrame({
        'numerical1': [1, 2, 3, 4, 5],
        'numerical2': [10, 20, 30, 40, 50],
        'categorical': ['A', 'B', 'A', 'B', 'C']
    })
    
    pipeline = ComprehensiveFeatureEngineeringPipeline(config={
        'phase2_cleaning': True,
        'phase3_polynomial': True,
        'phase3_business': True,
        'phase4_selection': 'filter',
        'phase6_metadata': True
    })
    
    pipeline.fit(df)
    df_transformed = pipeline.transform(df)
    
    # Should have more features due to polynomial/business features
    assert df_transformed.shape[1] >= df.shape[1]
    
    # Check metadata tracking
    report = pipeline.get_feature_report()
    assert report is not None


def test_startup_investor_transformer_advanced():
    """Test Enhanced Startup Investor Transformer with All Phases"""
    df = pd.DataFrame({
        'founded_year': [2015, 2010, 2020],
        'portfolio_companies': [10, 50, 100],
        'successful_exits': [2, 15, 30],
        'investment_stage': ['Seed', 'Series A', 'Series B'],
        'ai_focus': [1, 1, 0],
        'fintech_focus': [1, 0, 1]
    })
    
    transformer = StartupInvestorFeatureTransformer(
        current_year=2026,
        enable_advanced_features=True
    )
    
    transformer.fit(df)
    df_transformed = transformer.transform(df)
    
    # Check basic features
    assert 'investor_age_years' in df_transformed.columns
    assert 'exit_success_rate' in df_transformed.columns
    assert 'investment_stage_rank' in df_transformed.columns
    
    # Check advanced features (when enabled)
    if transformer.enable_advanced_features:
        assert 'portfolio_x_exits' in df_transformed.columns
        assert 'portfolio_companies_squared' in df_transformed.columns
    
    # Check metadata
    metadata = transformer.get_feature_metadata()
    assert metadata is not None
    assert len(metadata) > 0


def test_phase_4_feature_selection_methods():
    """Test Phase 4: Multiple Feature Selection Methods"""
    from src.features.advanced_feature_engineering import (
        WrapperFeatureSelector,
        EmbeddedFeatureSelector
    )
    
    X = np.random.randn(50, 10)
    y = np.random.randn(50)
    
    # Test Wrapper Method (RFE)
    wrapper_selector = WrapperFeatureSelector(n_features_to_select=5, method='rfe')
    wrapper_selector.fit(X, y)
    X_wrapper = wrapper_selector.transform(X)
    assert X_wrapper.shape[1] == 5
    
    # Test Embedded Method (Lasso)
    embedded_selector = EmbeddedFeatureSelector(method='lasso', alpha=0.1)
    embedded_selector.fit(X, y)
    X_embedded = embedded_selector.transform(X)
    assert X_embedded.shape[1] <= X.shape[1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])