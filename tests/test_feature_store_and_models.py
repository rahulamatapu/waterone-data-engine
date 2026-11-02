"""Automated test suite for Gold Feature Store and ML Anomaly Detection."""
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import pytest

from src.transformation.feature_store import GoldFeatureStoreBuilder
from src.models.anomaly_detector import WaterQualityAnomalyDetector


def test_feature_store_calculations():
    builder = GoldFeatureStoreBuilder()
    df_daily = builder.build_daily_water_quality_mart()

    assert not df_daily.empty
    assert "blended_turbidity_ntu" in df_daily.columns
    assert "total_upstream_flow_cfs" in df_daily.columns
    assert "turbidity_7d_ma" in df_daily.columns
    assert "turbidity_volatility_7d" in df_daily.columns
    assert "treatment_stress_ratio" in df_daily.columns
    assert "water_safety_index" in df_daily.columns

    # Verify values are positive and realistic
    assert (df_daily["blended_turbidity_ntu"] > 0).all()
    assert (df_daily["total_upstream_flow_cfs"] > 1000).all()
    assert (df_daily["water_safety_index"] >= 0).all()
    assert (df_daily["water_safety_index"] <= 100).all()


def test_ml_feature_store_structure():
    builder = GoldFeatureStoreBuilder()
    df_features = builder.build_ml_feature_store()

    assert not df_features.empty
    assert "turbidity_deviation_from_30d" in df_features.columns
    assert "flow_rate_of_change_7d" in df_features.columns
    assert len(df_features) > 10


def test_isolation_forest_anomaly_detector():
    detector = WaterQualityAnomalyDetector()
    df_anom = detector.train_and_score()

    assert not df_anom.empty
    assert "is_anomaly" in df_anom.columns
    assert "severity_index" in df_anom.columns
    assert "alert_level" in df_anom.columns
    assert "primary_contributing_factor" in df_anom.columns
    assert "recommended_action" in df_anom.columns

    # Check severity index boundaries [0, 100]
    assert (df_anom["severity_index"] >= 0.0).all()
    assert (df_anom["severity_index"] <= 100.0).all()

    # Verify alert level categories
    valid_levels = {"NORMAL", "ELEVATED_WATCH", "PRE_BREACH_WARNING", "CRITICAL_ALERT"}
    assert set(df_anom["alert_level"].unique()).issubset(valid_levels)

    # Verify primary contributing factors are valid features
    assert set(df_anom["primary_contributing_factor"].unique()).issubset(set(detector.FEATURE_COLS))


def test_lakehouse_gold_views():
    with duckdb.connect("database/water_lakehouse.duckdb") as con:
        tables = [t[0] for t in con.execute("SHOW TABLES;").fetchall()]
        assert "mart_daily_water_quality" in tables
        assert "mart_ml_feature_store" in tables
        assert "mart_anomaly_events" in tables

        # Test querying top anomaly events
        top_anom = con.execute("""
            SELECT date, severity_index, alert_level, primary_contributing_factor
            FROM mart_anomaly_events
            ORDER BY severity_index DESC
            LIMIT 5;
        """).df()
        assert len(top_anom) == 5

