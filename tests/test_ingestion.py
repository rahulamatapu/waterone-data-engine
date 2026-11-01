"""Automated test suite for Bronze Ingestion Engine."""
from pathlib import Path
import duckdb
import pandas as pd
import pytest

from src.ingestion.audit import AuditManager, compute_sha256
from src.ingestion.csv_normalizer import CSVNormalizer, detect_header_row, to_snake_case
from src.ingestion.pdf_parser import CCRParser, clean_param_name, extract_numeric_and_unit
from src.ingestion.usgs_client import USGSClient
from src.ingestion.orchestrator import BronzeIngestionOrchestrator


def test_audit_manager_record_and_check(tmp_path):
    test_db = tmp_path / "test_lakehouse.duckdb"
    test_file = tmp_path / "sample.csv"
    test_file.write_text("col1,col2\nval1,val2\n", encoding="utf-8")

    audit = AuditManager(db_path=str(test_db))
    is_ingested, file_hash = audit.is_already_ingested(test_file)
    assert not is_ingested
    assert len(file_hash) == 64  # SHA-256 hex string

    audit.record_ingestion(
        file_hash=file_hash,
        file_name="sample.csv",
        source_path=str(test_file),
        layer="BRONZE",
        status="SUCCESS",
        row_count=1,
    )

    is_ingested_now, _ = audit.is_already_ingested(test_file)
    assert is_ingested_now

    log_df = audit.get_audit_log(limit=5)
    assert len(log_df) == 1
    assert log_df.iloc[0]["file_name"] == "sample.csv"
    assert log_df.iloc[0]["status"] == "SUCCESS"


def test_snake_case_conversion():
    assert to_snake_case("PWS ID") == "pws_id"
    assert to_snake_case("# of Facilities") == "num_facilities"
    assert to_snake_case("Population Served Count") == "population_served_count"
    assert to_snake_case("Water System Site Visit") == "water_system_site_visit"


def test_csv_normalizer_kdhe_files(tmp_path):
    test_bronze = tmp_path / "bronze"
    test_db = tmp_path / "lakehouse.duckdb"
    audit = AuditManager(db_path=str(test_db))
    normalizer = CSVNormalizer(
        landing_dir="data/landing/kdhe",
        bronze_dir=str(test_bronze),
        audit_manager=audit,
    )

    results = normalizer.process_all(force=True)
    assert len(results) >= 4
    for r in results:
        assert r["status"] == "SUCCESS"
        assert r["row_count"] > 0

    facilities_pq = test_bronze / "bronze_facilities.parquet"
    assert facilities_pq.exists()
    df_fac = pd.read_parquet(facilities_pq)
    assert "facility_id" in df_fac.columns
    assert "facility_name" in df_fac.columns
    assert len(df_fac) == 134


def test_pdf_parser_ccr_extraction(tmp_path):
    test_bronze = tmp_path / "bronze"
    test_db = tmp_path / "lakehouse.duckdb"
    audit = AuditManager(db_path=str(test_db))
    parser = CCRParser(
        landing_dir="data/landing/ccr",
        bronze_dir=str(test_bronze),
        audit_manager=audit,
    )

    results = parser.process_all(force=True)
    assert len(results) == 1
    assert results[0]["status"] == "SUCCESS"
    assert results[0]["row_count"] > 0

    df_pdf = pd.read_parquet(test_bronze / "bronze_pdf_tables.parquet")
    assert "parameter_clean" in df_pdf.columns
    assert "waterone_result_value" in df_pdf.columns

    params = df_pdf["parameter_clean"].tolist()
    assert "Lead" in params
    assert "Copper" in params
    assert "Turbidity" in params
    assert "Fluoride" in params


def test_usgs_client_telemetry(tmp_path):
    test_bronze = tmp_path / "bronze"
    test_db = tmp_path / "lakehouse.duckdb"
    audit = AuditManager(db_path=str(test_db))
    client = USGSClient(bronze_dir=str(test_bronze), audit_manager=audit)

    res = client.ingest(force=True, days=7)
    assert res["status"] == "SUCCESS"
    assert res["row_count"] > 0

    df_tel = pd.read_parquet(test_bronze / "bronze_usgs_telemetry.parquet")
    assert "site_id" in df_tel.columns
    assert "streamflow_cfs" in df_tel.columns
    assert "turbidity_fnu" in df_tel.columns
    assert len(df_tel) > 0


def test_bronze_orchestrator_views():
    orchestrator = BronzeIngestionOrchestrator()
    res = orchestrator.run(force=True)
    assert res["total_records_ingested"] > 0

    with duckdb.connect("database/water_lakehouse.duckdb") as con:
        tables = [t[0] for t in con.execute("SHOW TABLES;").fetchall()]
        assert "bronze_facilities" in tables
        assert "bronze_violations" in tables
        assert "bronze_site_visits" in tables
        assert "bronze_summary" in tables
        assert "bronze_pdf_tables" in tables
        assert "bronze_usgs_telemetry" in tables

