"""Unified Bronze ingestion orchestrator."""
from pathlib import Path
from typing import Dict, List, Optional
import duckdb

from src.ingestion.audit import AuditManager
from src.ingestion.csv_normalizer import CSVNormalizer
from src.ingestion.pdf_parser import CCRParser
from src.ingestion.usgs_client import USGSClient


class BronzeIngestionOrchestrator:
    """Coordinates Bronze ingestion across KDHE CSVs, CCR PDF, and USGS river telemetry."""

    def __init__(
        self,
        landing_ccr: str = "data/landing/ccr",
        landing_kdhe: str = "data/landing/kdhe",
        bronze_dir: str = "data/bronze",
        db_path: str = "database/water_lakehouse.duckdb",
    ):
        self.landing_ccr = Path(landing_ccr)
        self.landing_kdhe = Path(landing_kdhe)
        self.bronze_dir = Path(bronze_dir)
        self.db_path = Path(db_path)
        self.audit = AuditManager(db_path=db_path)
        self.csv_normalizer = CSVNormalizer(
            landing_dir=str(self.landing_kdhe),
            bronze_dir=str(self.bronze_dir),
            audit_manager=self.audit,
        )
        self.pdf_parser = CCRParser(
            landing_dir=str(self.landing_ccr),
            bronze_dir=str(self.bronze_dir),
            audit_manager=self.audit,
        )
        self.usgs_client = USGSClient(
            bronze_dir=str(self.bronze_dir),
            audit_manager=self.audit,
        )

    def register_duckdb_views(self) -> None:
        """Registers/updates DuckDB views for all Bronze Parquet files."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(self.db_path)) as con:
            for p in self.bronze_dir.glob("*.parquet"):
                view_name = p.stem
                con.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{p}');")

    def run(self, force: bool = False) -> Dict[str, any]:
        """Runs the complete Bronze layer ingestion."""
        results = {
            "kdhe_csvs": [],
            "ccr_pdfs": [],
            "usgs_telemetry": None,
            "total_records_ingested": 0,
        }

        # 1. Ingest KDHE CSVs
        csv_results = self.csv_normalizer.process_all(force=force)
        results["kdhe_csvs"] = csv_results
        for r in csv_results:
            results["total_records_ingested"] += r.get("row_count", 0)

        # 2. Ingest CCR PDF
        pdf_results = self.pdf_parser.process_all(force=force)
        results["ccr_pdfs"] = pdf_results
        for r in pdf_results:
            results["total_records_ingested"] += r.get("row_count", 0)

        # 3. Ingest USGS River Telemetry
        usgs_result = self.usgs_client.ingest(force=force)
        results["usgs_telemetry"] = usgs_result
        results["total_records_ingested"] += usgs_result.get("row_count", 0)

        # 4. Register DuckDB views
        self.register_duckdb_views()

        return results

