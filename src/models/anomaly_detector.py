"""Isolation Forest multi-parameter anomaly detection engine for WaterOne Data Engine."""
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import yaml


class WaterQualityAnomalyDetector:
    """Trains and scores multi-parameter Isolation Forest for pre-breach anomaly detection."""

    FEATURE_COLS = [
        "blended_turbidity_ntu",
        "turbidity_7d_ma",
        "turbidity_volatility_7d",
        "river_discharge_cfs",
        "river_temp_c",
        "river_conductance_us_cm",
        "treatment_stress_ratio",
    ]

    def __init__(
        self,
        gold_dir: str = "data/gold",
        config_path: str = "config/pipeline_config.yaml",
        db_path: str = "database/water_lakehouse.duckdb",
    ):
        self.gold_dir = Path(gold_dir)
        self.gold_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = Path(config_path)
        self.db_path = Path(db_path)

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        anom_cfg = self.cfg.get("anomaly_detection", {})
        self.contamination = anom_cfg.get("contamination", 0.05)
        self.random_state = anom_cfg.get("random_state", 42)
        self.pre_breach_thresh = anom_cfg.get("pre_breach_warning_threshold", 65.0)
        self.critical_thresh = anom_cfg.get("critical_alert_threshold", 85.0)

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100,
            bootstrap=False,
        )
        self.scaler = StandardScaler()

    def train_and_score(
        self,
        df_features: Optional[pd.DataFrame] = None,
        custom_contamination: Optional[float] = None,
        persist_to_duckdb: bool = True,
    ) -> pd.DataFrame:
        """
        Trains Isolation Forest on feature store data and outputs scored anomaly events.
        """
        if df_features is None:
            feat_pq = self.gold_dir / "mart_ml_feature_store.parquet"
            if not feat_pq.exists():
                raise FileNotFoundError(
                    f"Prerequisite Gold feature store not found at {feat_pq}. "
                    "Please run 'poetry run waterpipe transform' or 'poetry run waterpipe run' first."
                )
            df_features = pd.read_parquet(feat_pq)

        if len(df_features) < 10:
            raise ValueError(f"Insufficient records ({len(df_features)}) to train Isolation Forest.")

        contamination = custom_contamination if custom_contamination is not None else self.contamination
        model = IsolationForest(
            contamination=contamination,
            random_state=self.random_state,
            n_estimators=100,
        )

        X = df_features[self.FEATURE_COLS].copy()
        # Impute any missing with median
        X = X.fillna(X.median())
        X_scaled = self.scaler.fit_transform(X)

        # Fit and predict
        model.fit(X_scaled)
        pred = model.predict(X_scaled)  # -1 = anomaly, 1 = normal
        scores = model.decision_function(X_scaled)  # higher = more normal, lower = anomalous

        # Compute continuous Severity Index (0 to 100)
        # Decision function is typically between -0.3 and 0.3
        # Lower scores map to higher severity
        raw_severity = (0.20 - scores) * 160.0
        severity_index = np.clip(raw_severity, 0.0, 100.0).round(1)

        # Identify primary contributing feature factor using standardized z-scores
        z_scores = np.abs(X_scaled)
        top_feature_indices = np.argmax(z_scores, axis=1)
        top_features = [self.FEATURE_COLS[i] for i in top_feature_indices]

        # Determine Alert Level and Operator Recommendation
        alert_levels = []
        recommendations = []
        for sev, top_feat, is_anom in zip(severity_index, top_features, pred == -1):
            if sev >= self.critical_thresh:
                alert_levels.append("CRITICAL_ALERT")
                recommendations.append(
                    f"CRITICAL: Extreme anomaly detected in {top_feat}. Engage rapid flocculant dosing at Hansen Plant and initiate secondary basin rerouting."
                )
            elif sev >= self.pre_breach_thresh or is_anom:
                alert_levels.append("PRE_BREACH_WARNING")
                recommendations.append(
                    f"WARNING: Upstream stress detected via {top_feat}. Increase coagulant monitoring and check intake caisson screens."
                )
            elif sev >= 45.0:
                alert_levels.append("ELEVATED_WATCH")
                recommendations.append(f"WATCH: Minor hydrological deviation in {top_feat}. Standard operation.")
            else:
                alert_levels.append("NORMAL")
                recommendations.append("Nominal operating conditions across both river intakes.")

        df_anom = pd.DataFrame({
            "date": df_features["date"],
            "is_anomaly": pred == -1,
            "decision_function_score": scores.round(4),
            "severity_index": severity_index,
            "alert_level": alert_levels,
            "primary_contributing_factor": top_features,
            "recommended_action": recommendations,
            "blended_turbidity_ntu": df_features["blended_turbidity_ntu"],
            "river_discharge_cfs": df_features["river_discharge_cfs"],
            "treatment_stress_ratio": df_features["treatment_stress_ratio"],
        })

        output_path = self.gold_dir / "mart_anomaly_events.parquet"
        if persist_to_duckdb:
            df_anom.to_parquet(output_path, index=False, engine="pyarrow")
            try:
                with duckdb.connect(str(self.db_path)) as con:
                    con.execute(f"CREATE OR REPLACE VIEW mart_anomaly_events AS SELECT * FROM read_parquet('{output_path}');")
            except Exception:
                pass

        return df_anom

