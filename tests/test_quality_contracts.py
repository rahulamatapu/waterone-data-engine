"""Automated test suite for Data Quality Contracts and Silver Star Schema."""
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
import pytest
import pandera.pandas as pa

from src.quality.contracts import (
    DataQualityGate,
    FacilitySchema,
    ContaminantSchema,
    WaterQualitySampleSchema,
    RiverTelemetrySchema,
    ViolationSchema,
    SiteVisitSchema,
)
from src.transformation.dimensional_models import SilverDimensionalBuilder


def test_pandera_rejection_negative_telemetry(tmp_path):
    """Verifies Pandera rejects impossible physical readings (negative flow, extreme temp)."""
    gate = DataQualityGate(db_path=str(tmp_path / "lakehouse.duckdb"))

    # Invalid dataframe with negative flow and boiling water temperature
    invalid_telemetry = pd.DataFrame({
        "site_id": ["06892350"],
        "streamflow_cfs": [-500.0],  # Negative flow: INVALID
        "turbidity_fnu": [25.0],
        "water_temperature_c": [120.0],  # 120 C: INVALID
        "specific_conductance_us_cm": [500.0],
    })

    is_valid, errors = gate.validate(invalid_telemetry, RiverTelemetrySchema, "test_river_telemetry")
    assert not is_valid
    assert len(errors) >= 2


def test_pandera_rejection_duplicate_facility_pk(tmp_path):
    """Verifies Pandera rejects duplicate facility IDs."""
    gate = DataQualityGate(db_path=str(tmp_path / "lakehouse.duckdb"))

    duplicate_facilities = pd.DataFrame({
        "facility_id": ["10591", "10591"],  # Duplicate PK: INVALID
        "facility_name": ["RESERVOIR 06", "RESERVOIR 06 COPY"],
        "facility_type": ["Reservoir", "Reservoir"],
        "facility_activity": ["Active", "Active"],
        "is_source": ["Y", "Y"],
        "water_type": ["Surface water", "Surface water"],
    })

    is_valid, errors = gate.validate(duplicate_facilities, FacilitySchema, "test_facility")
    assert not is_valid
    assert any("facility_id" in err or "unique" in err for err in errors)


def test_pandera_rejection_invalid_facility_status(tmp_path):
    """Verifies Pandera rejects invalid activity status."""
    gate = DataQualityGate(db_path=str(tmp_path / "lakehouse.duckdb"))

    bad_status = pd.DataFrame({
        "facility_id": ["10591"],
        "facility_name": ["RESERVOIR 06"],
        "facility_type": ["Reservoir"],
        "facility_activity": ["DESTROYED_IN_ORBIT"],  # Invalid status: INVALID
        "is_source": ["Y"],
        "water_type": ["Surface water"],
    })

    is_valid, errors = gate.validate(bad_status, FacilitySchema, "test_facility")
    assert not is_valid


def test_silver_star_schema_generation():
    """Verifies Silver layer Star Schema tables are generated and populated."""
    builder = SilverDimensionalBuilder()
    res = builder.run_all()

    assert res["dim_facility"] > 0
    assert res["dim_contaminant"] >= 10
    assert res["dim_monitoring_station"] == 2
    assert res["dim_date"] > 1000
    assert res["fact_water_quality_samples"] > 0
    assert res["fact_river_telemetry"] > 0


def test_duckdb_star_schema_joins():
    """Verifies relational Star Schema joins in DuckDB lakehouse."""
    with duckdb.connect("database/water_lakehouse.duckdb") as con:
        # 1. Join samples with contaminant dimension
        joined_samples = con.execute("""
            SELECT s.parameter_clean, c.display_name, s.result_value, c.unit, c.critical_threshold
            FROM fact_water_quality_samples s
            JOIN dim_contaminant c ON s.contaminant_id = c.contaminant_id
            WHERE s.result_value IS NOT NULL;
        """).df()
        assert len(joined_samples) > 0

        # 2. Join telemetry with monitoring station dimension
        joined_telemetry = con.execute("""
            SELECT t.telemetry_id, m.river_name, t.streamflow_cfs, t.turbidity_fnu
            FROM fact_river_telemetry t
            JOIN dim_monitoring_station m ON t.site_id = m.site_id
            LIMIT 10;
        """).df()
        assert len(joined_telemetry) == 10
        assert "river_name" in joined_telemetry.columns

        # 3. Check quality audit log recorded in DuckDB
        audit_records = con.execute("""
            SELECT count(*) FROM silver_quality_audit WHERE is_valid = true;
        """).fetchone()[0]
        assert audit_records >= 6

