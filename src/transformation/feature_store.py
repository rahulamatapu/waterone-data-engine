"""Gold layer Feature Store and Analytical Marts builder for WaterOne Data Engine."""
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import duckdb
import numpy as np
import pandas as pd
import yaml


class GoldFeatureStoreBuilder:
    """Builds Gold analytical marts and ML feature stores."""

    def __init__(
        self,
        silver_dir: str = "data/silver",
        gold_dir: str = "data/gold",
        config_path: str = "config/pipeline_config.yaml",
        db_path: str = "database/water_lakehouse.duckdb",
    ):
        self.silver_dir = Path(silver_dir)
        self.gold_dir = Path(gold_dir)
        self.gold_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = Path(config_path)
        self.db_path = Path(db_path)

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.weights = self.cfg.get("source_blending_weights", {
            "missouri_river": 0.514,
            "wolcott_collector": 0.275,
            "kansas_river": 0.196,
            "wells": 0.016,
        })

    def build_daily_water_quality_mart(self) -> pd.DataFrame:
        """
        Builds mart_daily_water_quality by aggregating hourly river telemetry,
        computing seasonal source blending, rolling trends, and treatment stress ratios.
        """
        tel_pq = self.silver_dir / "fact_river_telemetry.parquet"
        if not tel_pq.exists():
            return pd.DataFrame()

        df_tel = pd.read_parquet(tel_pq)
        df_tel["date"] = pd.to_datetime(df_tel["timestamp"]).dt.date

        # Aggregate hourly readings to daily per site
        daily_agg = df_tel.groupby(["date", "site_id"]).agg(
            mean_streamflow_cfs=("streamflow_cfs", "mean"),
            max_streamflow_cfs=("streamflow_cfs", "max"),
            mean_turbidity_fnu=("turbidity_fnu", "mean"),
            max_turbidity_fnu=("turbidity_fnu", "max"),
            mean_temp_c=("water_temperature_c", "mean"),
            mean_conductance_us_cm=("specific_conductance_us_cm", "mean"),
        ).reset_index()

        # Pivot Kansas (06892350) and Missouri (06893000) rivers
        pivoted = daily_agg.pivot(index="date", columns="site_id")
        pivoted.columns = [f"{col[0]}_{col[1]}" for col in pivoted.columns]
        df_daily = pivoted.reset_index().sort_values("date").reset_index(drop=True)

        # Fallback column handling if one site is missing
        ks_id = "06892350"
        mo_id = "06893000"

        flow_ks = df_daily[f"mean_streamflow_cfs_{ks_id}"] if f"mean_streamflow_cfs_{ks_id}" in df_daily else 4500.0
        flow_mo = df_daily[f"mean_streamflow_cfs_{mo_id}"] if f"mean_streamflow_cfs_{mo_id}" in df_daily else 45000.0
        turb_ks = df_daily[f"mean_turbidity_fnu_{ks_id}"] if f"mean_turbidity_fnu_{ks_id}" in df_daily else 25.0
        turb_mo = df_daily[f"mean_turbidity_fnu_{mo_id}"] if f"mean_turbidity_fnu_{mo_id}" in df_daily else 45.0
        temp_ks = df_daily[f"mean_temp_c_{ks_id}"] if f"mean_temp_c_{ks_id}" in df_daily else 14.0
        temp_mo = df_daily[f"mean_temp_c_{mo_id}"] if f"mean_temp_c_{mo_id}" in df_daily else 13.0
        cond_ks = df_daily[f"mean_conductance_us_cm_{ks_id}"] if f"mean_conductance_us_cm_{ks_id}" in df_daily else 650.0
        cond_mo = df_daily[f"mean_conductance_us_cm_{mo_id}"] if f"mean_conductance_us_cm_{mo_id}" in df_daily else 550.0

        # Compute WaterOne Source Blending
        w_mo = self.weights.get("missouri_river", 0.514)
        w_wolcott = self.weights.get("wolcott_collector", 0.275)
        w_ks = self.weights.get("kansas_river", 0.196)
        w_wells = self.weights.get("wells", 0.016)

        # Wolcott collector draws alluvial groundwater influenced by MO River with ~80% natural filtration
        wolcott_turbidity = turb_mo * 0.20
        wells_turbidity = 0.5  # baseline deep alluvial well turbidity

        blended_turbidity = (
            w_mo * turb_mo +
            w_wolcott * wolcott_turbidity +
            w_ks * turb_ks +
            w_wells * wells_turbidity
        )

        total_upstream_flow = flow_mo + flow_ks
        blended_temp = 0.789 * temp_mo + 0.211 * temp_ks
        blended_conductance = 0.789 * cond_mo + 0.211 * cond_ks

        df_mart = pd.DataFrame({
            "date": pd.to_datetime(df_daily["date"]),
            "kansas_river_flow_cfs": flow_ks.round(1),
            "missouri_river_flow_cfs": flow_mo.round(1),
            "total_upstream_flow_cfs": total_upstream_flow.round(1),
            "kansas_river_turbidity_fnu": turb_ks.round(2),
            "missouri_river_turbidity_fnu": turb_mo.round(2),
            "blended_turbidity_ntu": blended_turbidity.round(2),
            "blended_temp_c": blended_temp.round(2),
            "blended_conductance_us_cm": blended_conductance.round(1),
        })

        # Rolling Window Metrics
        df_mart["turbidity_7d_ma"] = df_mart["blended_turbidity_ntu"].rolling(7, min_periods=1).mean().round(2)
        df_mart["turbidity_30d_ma"] = df_mart["blended_turbidity_ntu"].rolling(30, min_periods=1).mean().round(2)
        df_mart["turbidity_volatility_7d"] = df_mart["blended_turbidity_ntu"].rolling(7, min_periods=2).std().fillna(0).round(2)
        df_mart["turbidity_volatility_30d"] = df_mart["blended_turbidity_ntu"].rolling(30, min_periods=2).std().fillna(0).round(2)

        df_mart["flow_7d_ma"] = df_mart["total_upstream_flow_cfs"].rolling(7, min_periods=1).mean().round(1)
        df_mart["flow_30d_ma"] = df_mart["total_upstream_flow_cfs"].rolling(30, min_periods=1).mean().round(1)
        df_mart["flow_volatility_7d"] = df_mart["total_upstream_flow_cfs"].rolling(7, min_periods=2).std().fillna(0).round(1)

        # Treatment Stress & Filtration Metrics
        finished_turbidity_baseline = 0.23  # From CCR 2026 report (highest level detected)
        df_mart["finished_water_turbidity_baseline"] = finished_turbidity_baseline
        df_mart["treatment_stress_ratio"] = (df_mart["blended_turbidity_ntu"] / finished_turbidity_baseline).round(1)
        df_mart["required_log_removal"] = np.log10(np.maximum(df_mart["blended_turbidity_ntu"], 1.0) / finished_turbidity_baseline).round(2)

        # Overall Water Safety Health Index (0 - 100, where 100 = optimal clarity & safety)
        # Drops during severe river spikes
        df_mart["water_safety_index"] = np.clip(
            100.0 - (df_mart["blended_turbidity_ntu"] / 100.0) * 15.0 - (df_mart["turbidity_volatility_7d"] / 20.0) * 10.0,
            60.0,
            100.0
        ).round(1)

        output_path = self.gold_dir / "mart_daily_water_quality.parquet"
        df_mart.to_parquet(output_path, index=False, engine="pyarrow")
        return df_mart

    def build_ml_feature_store(self, df_daily: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Builds mart_ml_feature_store containing normalized analytical features
        for unsupervised anomaly detection.
        """
        if df_daily is None:
            daily_pq = self.gold_dir / "mart_daily_water_quality.parquet"
            if not daily_pq.exists():
                df_daily = self.build_daily_water_quality_mart()
            else:
                df_daily = pd.read_parquet(daily_pq)

        df_features = pd.DataFrame()
        df_features["date"] = df_daily["date"]
        df_features["blended_turbidity_ntu"] = df_daily["blended_turbidity_ntu"]
        df_features["turbidity_7d_ma"] = df_daily["turbidity_7d_ma"]
        df_features["turbidity_30d_ma"] = df_daily["turbidity_30d_ma"]
        df_features["turbidity_volatility_7d"] = df_daily["turbidity_volatility_7d"]
        df_features["river_discharge_cfs"] = df_daily["total_upstream_flow_cfs"]
        df_features["river_temp_c"] = df_daily["blended_temp_c"]
        df_features["river_conductance_us_cm"] = df_daily["blended_conductance_us_cm"]
        df_features["treatment_stress_ratio"] = df_daily["treatment_stress_ratio"]

        # Engineered delta features
        df_features["turbidity_deviation_from_30d"] = (
            df_features["blended_turbidity_ntu"] - df_features["turbidity_30d_ma"]
        ).round(2)
        df_features["flow_rate_of_change_7d"] = (
            (df_daily["total_upstream_flow_cfs"] - df_daily["flow_7d_ma"]) / (df_daily["flow_7d_ma"] + 1e-5)
        ).round(4)

        output_path = self.gold_dir / "mart_ml_feature_store.parquet"
        df_features.to_parquet(output_path, index=False, engine="pyarrow")
        return df_features

    def register_duckdb_gold_views(self) -> None:
        """Registers all Gold marts as views in DuckDB lakehouse."""
        with duckdb.connect(str(self.db_path)) as con:
            for p in self.gold_dir.glob("*.parquet"):
                view_name = p.stem
                con.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{p}');")

    def run_all(self) -> Dict[str, int]:
        """Executes full Gold layer marts generation."""
        df_daily = self.build_daily_water_quality_mart()
        df_features = self.build_ml_feature_store(df_daily)
        self.register_duckdb_gold_views()

        return {
            "mart_daily_water_quality": len(df_daily),
            "mart_ml_feature_store": len(df_features),
        }

