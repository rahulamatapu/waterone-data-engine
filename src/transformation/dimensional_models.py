"""Silver layer Star Schema builder for WaterOne Data Engine."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import duckdb
import numpy as np
import pandas as pd
import yaml

from src.quality.contracts import (
    DataQualityGate,
    FacilitySchema,
    ContaminantSchema,
    WaterQualitySampleSchema,
    RiverTelemetrySchema,
    ViolationSchema,
    SiteVisitSchema,
)


class SilverDimensionalBuilder:
    """Builds clean, validated Star Schema Parquet tables for the Silver layer."""

    def __init__(
        self,
        bronze_dir: str = "data/bronze",
        silver_dir: str = "data/silver",
        config_path: str = "config/pipeline_config.yaml",
        mapping_path: str = "config/contaminant_mapping.yaml",
        db_path: str = "database/water_lakehouse.duckdb",
    ):
        self.bronze_dir = Path(bronze_dir)
        self.silver_dir = Path(silver_dir)
        self.silver_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = Path(config_path)
        self.mapping_path = Path(mapping_path)
        self.db_path = Path(db_path)
        self.quality_gate = DataQualityGate(db_path=db_path)

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.pipeline_cfg = yaml.safe_load(f)

        with open(self.mapping_path, "r", encoding="utf-8") as f:
            self.contaminant_cfg = yaml.safe_load(f)

    # -------------------------------------------------------------
    # DIMENSION BUILDERS
    # -------------------------------------------------------------

    def build_dim_facility(self) -> pd.DataFrame:
        """Transforms bronze_facilities into clean dim_facility."""
        pq_path = self.bronze_dir / "bronze_facilities.parquet"
        if not pq_path.exists():
            return pd.DataFrame()

        df_raw = pd.read_parquet(pq_path)
        
        # Select and rename columns
        df = pd.DataFrame()
        df["facility_id"] = df_raw["facility_id"].astype(str).str.strip()
        df["pws_id"] = df_raw["pws_id"].astype(str).str.strip()
        df["pws_name"] = df_raw["pws_name"].astype(str).str.strip()
        df["facility_name"] = df_raw["facility_name"].astype(str).str.strip()
        df["facility_type"] = df_raw["facility_type"].astype(str).str.strip()
        
        # Standardize facility activity
        def clean_activity(act):
            if pd.isna(act) or not str(act).strip():
                return "Unknown"
            act_str = str(act).strip().capitalize()
            return act_str if act_str in {"Active", "Inactive", "Proposed"} else "Unknown"

        df["facility_activity"] = df_raw["facility_activity"].apply(clean_activity)
        df["is_source"] = df_raw["is_source"].apply(lambda x: "Y" if str(x).upper() == "Y" else "N")
        df["water_type"] = df_raw["water_type"].fillna("Surface water")
        df["availability_description"] = df_raw["availability_description"].fillna("Permanent")
        df["filtration_status"] = df_raw["filtration_status"].fillna("None")
        df["seller_pws_name"] = df_raw["seller_pws_name"].fillna("None")
        df["deactivation_date"] = pd.to_datetime(df_raw["facility_deactivation_date"], errors="coerce")

        # Deduplicate on facility_id
        df = df.drop_duplicates(subset=["facility_id"]).sort_values("facility_id").reset_index(drop=True)

        # Validate with Pandera
        is_valid, errors = self.quality_gate.validate(df, FacilitySchema, "dim_facility")
        if not is_valid:
            print(f"Warning: dim_facility had validation warnings: {errors[:2]}")

        output_path = self.silver_dir / "dim_facility.parquet"
        df.to_parquet(output_path, index=False, engine="pyarrow")
        return df

    def build_dim_contaminant(self) -> pd.DataFrame:
        """Builds dim_contaminant from contaminant mapping configuration."""
        records = []
        contaminants = self.contaminant_cfg.get("contaminants", {})
        for cid, info in contaminants.items():
            records.append({
                "contaminant_id": cid,
                "display_name": info.get("display_name", cid.title()),
                "chemical_symbol": info.get("chemical_symbol", ""),
                "category": info.get("category", "General"),
                "unit": info.get("unit", "ppm"),
                "mcl": info.get("mcl"),
                "mclg": info.get("mclg"),
                "action_level": info.get("action_level"),
                "warning_threshold": info.get("warning_threshold"),
                "critical_threshold": info.get("critical_threshold"),
                "description": info.get("description", ""),
            })

        df = pd.DataFrame(records)
        is_valid, errors = self.quality_gate.validate(df, ContaminantSchema, "dim_contaminant")
        if not is_valid:
            print(f"Warning: dim_contaminant validation warnings: {errors[:2]}")

        output_path = self.silver_dir / "dim_contaminant.parquet"
        df.to_parquet(output_path, index=False, engine="pyarrow")
        return df

    def build_dim_monitoring_station(self) -> pd.DataFrame:
        """Builds dim_monitoring_station from USGS config."""
        records = []
        stations = self.pipeline_cfg.get("usgs_telemetry", {}).get("stations", {})
        for key, s in stations.items():
            river = "Kansas River" if "kansas" in key else "Missouri River"
            records.append({
                "site_id": s.get("site_id"),
                "site_name": s.get("site_name"),
                "river_name": river,
                "latitude": float(s.get("latitude", 0.0)),
                "longitude": float(s.get("longitude", 0.0)),
                "state": "KS" if "KS" in s.get("site_name", "") else "MO",
            })

        df = pd.DataFrame(records)
        output_path = self.silver_dir / "dim_monitoring_station.parquet"
        df.to_parquet(output_path, index=False, engine="pyarrow")
        return df

    def build_dim_date(self, start_year: int = 2000, end_year: int = 2030) -> pd.DataFrame:
        """Builds comprehensive calendar and USGS water-year dimension."""
        dates = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq="D")
        df = pd.DataFrame({"date": dates})
        df["date_id"] = df["date"].dt.strftime("%Y%m%d").astype(int)
        df["year"] = df["date"].dt.year
        df["quarter"] = df["date"].dt.quarter
        df["month"] = df["date"].dt.month
        df["month_name"] = df["date"].dt.strftime("%B")
        df["day"] = df["date"].dt.day
        df["day_of_week"] = df["date"].dt.day_of_week + 1
        df["day_name"] = df["date"].dt.strftime("%A")
        df["is_weekend"] = df["day_of_week"].isin([6, 7])

        # USGS Water Year runs from Oct 1 to Sep 30
        df["water_year"] = np.where(df["month"] >= 10, df["year"] + 1, df["year"])

        # Seasons
        def get_season(m):
            if m in [12, 1, 2]:
                return "Winter"
            elif m in [3, 4, 5]:
                return "Spring"
            elif m in [6, 7, 8]:
                return "Summer"
            else:
                return "Fall"

        df["season"] = df["month"].apply(get_season)

        output_path = self.silver_dir / "dim_date.parquet"
        df.to_parquet(output_path, index=False, engine="pyarrow")
        return df

    # -------------------------------------------------------------
    # FACT BUILDERS
    # -------------------------------------------------------------

    def build_fact_water_quality_samples(self) -> pd.DataFrame:
        """Transforms bronze_pdf_tables into fact_water_quality_samples."""
        pq_path = self.bronze_dir / "bronze_pdf_tables.parquet"
        if not pq_path.exists():
            return pd.DataFrame()

        df_raw = pd.read_parquet(pq_path)
        
        # Map parameter name to standardized contaminant_id
        mapping = {
            "lead": "lead",
            "copper": "copper",
            "fluoride": "fluoride",
            "total trihalomethanes": "tthm",
            "haloacetic acids": "haa5",
            "nitrate": "nitrate",
            "turbidity": "turbidity",
            "chloramines": "chlorine_residual",
            "barium": "barium",
            "atrazine": "atrazine",
        }

        def match_contaminant_id(param_name):
            p = param_name.lower()
            for key, cid in mapping.items():
                if key in p:
                    return cid
            return re.sub(r"[^a-zA-Z0-9]+", "_", p).strip("_")

        records = []
        for idx, row in df_raw.iterrows():
            param = str(row["parameter_clean"]).strip()
            cid = match_contaminant_id(param)
            year = int(row.get("reporting_year", 2026))
            sample_id = f"CCR_{year}_{cid}_{idx+1}"
            res_val = row.get("waterone_result_value")
            res_val = float(res_val) if pd.notnull(res_val) else None
            mcl_val = float(row.get("mcl_value")) if pd.notnull(row.get("mcl_value")) else None
            mclg_val = float(row.get("mclg_value")) if pd.notnull(row.get("mclg_value")) else None

            # Compliance check
            met_standard = True
            if res_val is not None and mcl_val is not None:
                met_standard = res_val <= mcl_val

            records.append({
                "sample_id": sample_id,
                "contaminant_id": cid,
                "parameter_clean": param,
                "category": row.get("category", "Unspecified"),
                "reporting_year": year,
                "result_value": res_val,
                "unit": row.get("unit", "ppm"),
                "mcl_value": mcl_val,
                "mclg_value": mclg_val,
                "range_min": row.get("range_min"),
                "range_max": row.get("range_max"),
                "sample_frequency": row.get("sample_frequency", "Annual"),
                "met_standard": met_standard,
                "source_description": row.get("source_description", ""),
            })

        df = pd.DataFrame(records)
        is_valid, errors = self.quality_gate.validate(df, WaterQualitySampleSchema, "fact_water_quality_samples")
        if not is_valid:
            print(f"Warning: fact_water_quality_samples validation warnings: {errors[:2]}")

        output_path = self.silver_dir / "fact_water_quality_samples.parquet"
        df.to_parquet(output_path, index=False, engine="pyarrow")
        return df

    def build_fact_river_telemetry(self) -> pd.DataFrame:
        """Transforms bronze_usgs_telemetry into clean fact_river_telemetry."""
        pq_path = self.bronze_dir / "bronze_usgs_telemetry.parquet"
        if not pq_path.exists():
            return pd.DataFrame()

        df_raw = pd.read_parquet(pq_path)
        df = df_raw.copy()
        
        # Ensure timestamp is datetime and generate date_id
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date_id"] = df["timestamp"].dt.strftime("%Y%m%d").astype(int)
        
        # Telemetry ID surrogate key
        df["telemetry_id"] = df["site_id"].astype(str) + "_" + df["timestamp"].dt.strftime("%Y%m%d%H%M")
        
        # Add hydrological regime flags
        def flag_turbidity(val):
            if pd.isna(val):
                return "UNKNOWN"
            elif val > 100:
                return "SPIKE"
            elif val > 50:
                return "ELEVATED"
            return "NORMAL"

        df["turbidity_flag"] = df["turbidity_fnu"].apply(flag_turbidity)

        # Validate with Pandera
        is_valid, errors = self.quality_gate.validate(df, RiverTelemetrySchema, "fact_river_telemetry")
        if not is_valid:
            print(f"Warning: fact_river_telemetry validation warnings: {errors[:2]}")

        output_path = self.silver_dir / "fact_river_telemetry.parquet"
        df.to_parquet(output_path, index=False, engine="pyarrow")
        return df

    def build_fact_violations(self) -> pd.DataFrame:
        """Transforms bronze_violations into fact_violations."""
        pq_path = self.bronze_dir / "bronze_violations.parquet"
        if not pq_path.exists():
            return pd.DataFrame()

        df_raw = pd.read_parquet(pq_path)
        df = pd.DataFrame()
        df["violation_id"] = df_raw["violation_id"].astype(str)
        df["pws_id"] = df_raw["pws_id"].astype(str)
        df["rule_name"] = df_raw["rule_name"].astype(str)
        df["violation_type"] = df_raw["violation_type"].astype(str)
        df["is_health_based"] = df_raw["is_health_based"].astype(str)
        df["compliance_begin_date"] = pd.to_datetime(df_raw["compliance_period_begin_date"], errors="coerce")
        df["compliance_end_date"] = pd.to_datetime(df_raw["compliance_period_end_date"], errors="coerce")
        df["compliance_status"] = df_raw["compliance_status"].astype(str)
        df["rtc_date"] = pd.to_datetime(df_raw["rtc_date"], errors="coerce")

        # Compute resolution days
        df["resolution_days"] = (df["rtc_date"] - df["compliance_begin_date"]).dt.days
        df["resolution_days"] = df["resolution_days"].apply(lambda x: max(0, x) if pd.notnull(x) else None)

        is_valid, errors = self.quality_gate.validate(df, ViolationSchema, "fact_violations")
        if not is_valid:
            print(f"Warning: fact_violations validation warnings: {errors[:2]}")

        output_path = self.silver_dir / "fact_violations.parquet"
        df.to_parquet(output_path, index=False, engine="pyarrow")
        return df

    def build_fact_site_visits(self) -> pd.DataFrame:
        """Transforms bronze_site_visits into fact_site_visits."""
        pq_path = self.bronze_dir / "bronze_site_visits.parquet"
        if not pq_path.exists():
            return pd.DataFrame()

        df_raw = pd.read_parquet(pq_path)
        df = pd.DataFrame()
        df["visit_id"] = df_raw["visit_id"].astype(str)
        df["pws_id"] = df_raw["pws_id"].astype(str)
        df["visit_date"] = pd.to_datetime(df_raw["visit_date"], errors="coerce")
        df["visit_reason"] = df_raw["visit_reason"].astype(str)
        df["agency_type"] = df_raw["agency_type"].fillna("District")
        
        # Flag if any evaluation noted recommendations or deficiencies
        def has_rec(row):
            text = " ".join([str(val) for val in row if pd.notnull(val)]).lower()
            return "recommendation" in text or "deficienc" in text

        df["has_findings"] = df_raw.apply(has_rec, axis=1)
        df["treatment_evaluation"] = df_raw["treatment_evaluation"].fillna("Not Applicable")
        df["finished_water_storage_evaluation"] = df_raw["finished_water_storage_evaluation"].fillna("Not Applicable")
        df["distribution_evaluation"] = df_raw["distribution_evaluation"].fillna("Not Applicable")
        df["visit_comments"] = df_raw["visit_comments"].fillna("")

        is_valid, errors = self.quality_gate.validate(df, SiteVisitSchema, "fact_site_visits")
        if not is_valid:
            print(f"Warning: fact_site_visits validation warnings: {errors[:2]}")

        output_path = self.silver_dir / "fact_site_visits.parquet"
        df.to_parquet(output_path, index=False, engine="pyarrow")
        return df

    # -------------------------------------------------------------
    # ORCHESTRATION & REGISTRATION
    # -------------------------------------------------------------

    def register_duckdb_silver_views(self) -> None:
        """Registers all Silver layer Star Schema tables as DuckDB views."""
        with duckdb.connect(str(self.db_path)) as con:
            for p in self.silver_dir.glob("*.parquet"):
                view_name = p.stem
                con.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{p}');")

    def run_all(self) -> Dict[str, any]:
        """Executes full Silver layer Star Schema transformations."""
        results = {}
        # Dimensions
        results["dim_facility"] = len(self.build_dim_facility())
        results["dim_contaminant"] = len(self.build_dim_contaminant())
        results["dim_monitoring_station"] = len(self.build_dim_monitoring_station())
        results["dim_date"] = len(self.build_dim_date())

        # Facts
        results["fact_water_quality_samples"] = len(self.build_fact_water_quality_samples())
        results["fact_river_telemetry"] = len(self.build_fact_river_telemetry())
        results["fact_violations"] = len(self.build_fact_violations())
        results["fact_site_visits"] = len(self.build_fact_site_visits())

        self.register_duckdb_silver_views()
        return results
