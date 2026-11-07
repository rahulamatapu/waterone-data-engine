"""Pipeline orchestrator for WaterOne Data Engine."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import duckdb
import pandas as pd
from rich.console import Console
from rich.table import Table

from src.ingestion.orchestrator import BronzeIngestionOrchestrator
from src.quality.contracts import DataQualityGate
from src.transformation.dimensional_models import SilverDimensionalBuilder
from src.transformation.feature_store import GoldFeatureStoreBuilder
from src.models.anomaly_detector import WaterQualityAnomalyDetector


@dataclass
class PipelineStatus:
    database_exists: bool = False
    database_size_bytes: int = 0
    layer_counts: Dict[str, int] = field(default_factory=dict)
    recent_audits: List[Dict] = field(default_factory=list)
    quality_checks: List[Dict] = field(default_factory=list)


class WaterOnePipeline:
    """End-to-end orchestrator for WaterOne Data Engine lakehouse."""

    def __init__(
        self,
        db_path: str = "database/water_lakehouse.duckdb",
        bronze_dir: str = "data/bronze",
        silver_dir: str = "data/silver",
        gold_dir: str = "data/gold",
        config_path: str = "config/pipeline_config.yaml",
    ):
        self.db_path = Path(db_path)
        self.bronze_dir = Path(bronze_dir)
        self.silver_dir = Path(silver_dir)
        self.gold_dir = Path(gold_dir)
        self.config_path = Path(config_path)

        self.ingestion = BronzeIngestionOrchestrator(
            bronze_dir=str(self.bronze_dir),
            db_path=str(self.db_path),
        )
        self.silver = SilverDimensionalBuilder(
            bronze_dir=str(self.bronze_dir),
            silver_dir=str(self.silver_dir),
            config_path=str(self.config_path),
            db_path=str(self.db_path),
        )
        self.gold = GoldFeatureStoreBuilder(
            silver_dir=str(self.silver_dir),
            gold_dir=str(self.gold_dir),
            config_path=str(self.config_path),
            db_path=str(self.db_path),
        )
        self.detector = WaterQualityAnomalyDetector(
            gold_dir=str(self.gold_dir),
            config_path=str(self.config_path),
            db_path=str(self.db_path),
        )
        self.quality_gate = DataQualityGate(db_path=str(self.db_path))

    def ingest(self, force: bool = False) -> Dict[str, any]:
        """Runs the Bronze ingestion engine."""
        return self.ingestion.run(force=force)

    def validate(self) -> Dict[str, any]:
        """Executes Pandera quality gates across Bronze & Silver tables."""
        results = {"status": "SUCCESS", "tables": {}}
        with duckdb.connect(str(self.db_path)) as con:
            tables = [t[0] for t in con.execute("SHOW TABLES;").fetchall()]
            if "silver_quality_audit" in tables:
                df_audit = con.execute("""
                    SELECT table_name, checked_at, records_checked, is_valid, error_count, error_details
                    FROM silver_quality_audit
                    ORDER BY checked_at DESC
                """).df()
                results["audit_log"] = df_audit.to_dict(orient="records")
            else:
                results["audit_log"] = []
        return results

    def transform(self) -> Dict[str, any]:
        """Builds Silver Star Schema and Gold Feature Marts."""
        # Check prerequisite
        if not (self.bronze_dir / "bronze_facilities.parquet").exists():
            raise FileNotFoundError(
                "Bronze layer tables missing. Please run 'waterpipe ingest' first."
            )

        silver_results = self.silver.run_all()
        gold_results = self.gold.run_all()

        return {
            "silver": silver_results,
            "gold": gold_results,
        }

    def detect(self, contamination: Optional[float] = None) -> pd.DataFrame:
        """Trains Isolation Forest and scores anomaly events."""
        return self.detector.train_and_score(custom_contamination=contamination)

    def run_full(self, force: bool = False) -> Dict[str, any]:
        """Runs complete end-to-end pipeline: Ingest -> Transform -> Detect."""
        ingest_res = self.ingest(force=force)
        transform_res = self.transform()
        detect_df = self.detect()

        return {
            "ingest": ingest_res,
            "transform": transform_res,
            "anomaly_events_scored": len(detect_df),
            "critical_alerts": int((detect_df["alert_level"] == "CRITICAL_ALERT").sum()),
            "pre_breach_warnings": int((detect_df["alert_level"] == "PRE_BREACH_WARNING").sum()),
        }

    def status(self) -> PipelineStatus:
        """Inspects pipeline health, layer row counts, and storage stats."""
        status = PipelineStatus()
        if self.db_path.exists():
            status.database_exists = True
            status.database_size_bytes = self.db_path.stat().st_size

            with duckdb.connect(str(self.db_path)) as con:
                tables = [t[0] for t in con.execute("SHOW TABLES;").fetchall()]
                for t in tables:
                    try:
                        cnt = con.execute(f"SELECT count(*) FROM {t};").fetchone()[0]
                        status.layer_counts[t] = cnt
                    except Exception:
                        status.layer_counts[t] = -1

                if "bronze_audit_log" in tables:
                    status.recent_audits = con.execute("""
                        SELECT file_name, layer, status, row_count, ingested_at
                        FROM bronze_audit_log
                        ORDER BY ingested_at DESC LIMIT 5;
                    """).df().to_dict(orient="records")

                if "silver_quality_audit" in tables:
                    status.quality_checks = con.execute("""
                        SELECT table_name, is_valid, error_count, checked_at
                        FROM silver_quality_audit
                        ORDER BY checked_at DESC LIMIT 5;
                    """).df().to_dict(orient="records")

        return status

    def execute_query(self, sql: str) -> pd.DataFrame:
        """Runs an ad-hoc SQL query against the local DuckDB lakehouse."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found at {self.db_path}. Run pipeline first.")
        with duckdb.connect(str(self.db_path), read_only=True) as con:
            return con.execute(sql).df()

    def export_mart(self, mart_name: str, output_path: str, format_type: str = "csv") -> Path:
        """Exports a Gold analytical mart to CSV, Parquet, or JSON."""
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        with duckdb.connect(str(self.db_path), read_only=True) as con:
            df = con.execute(f"SELECT * FROM {mart_name};").df()

        fmt = format_type.lower()
        if fmt == "csv":
            df.to_csv(out_p, index=False)
        elif fmt in {"parquet", "pq"}:
            df.to_parquet(out_p, index=False)
        elif fmt == "json":
            df.to_json(out_p, orient="records", date_format="iso", indent=2)
        else:
            raise ValueError(f"Unsupported format: {format_type}. Choose from 'csv', 'parquet', 'json'.")

        return out_p
