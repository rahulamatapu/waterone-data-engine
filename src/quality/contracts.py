"""Pandera data quality contracts and validation suite for Silver layer tables."""
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import duckdb
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema


# 1. Facility Contract
FacilitySchema = DataFrameSchema(
    columns={
        "facility_id": Column(str, nullable=False, unique=True, coerce=True),
        "facility_name": Column(str, nullable=False, coerce=True),
        "facility_type": Column(str, nullable=False, coerce=True),
        "facility_activity": Column(
            str,
            Check.isin(["Active", "Inactive", "Proposed", "Unknown"]),
            nullable=False,
            coerce=True,
        ),
        "is_source": Column(str, Check.isin(["Y", "N"]), nullable=True, coerce=True),
        "water_type": Column(str, nullable=True, coerce=True),
    },
    strict=False,
)

# 2. Contaminant Dimension Contract
ContaminantSchema = DataFrameSchema(
    columns={
        "contaminant_id": Column(str, nullable=False, unique=True, coerce=True),
        "display_name": Column(str, nullable=False, coerce=True),
        "unit": Column(str, nullable=False, coerce=True),
        "warning_threshold": Column(float, Check.greater_than_or_equal_to(0), nullable=True, coerce=True),
        "critical_threshold": Column(float, Check.greater_than_or_equal_to(0), nullable=True, coerce=True),
    },
    strict=False,
)

# 3. Water Quality Finished Water Sample Contract
WaterQualitySampleSchema = DataFrameSchema(
    columns={
        "sample_id": Column(str, nullable=False, coerce=True),
        "parameter_clean": Column(str, nullable=False, coerce=True),
        "result_value": Column(float, Check.greater_than_or_equal_to(0), nullable=True, coerce=True),
        "reporting_year": Column(int, Check.in_range(2000, 2050), nullable=False, coerce=True),
    },
    strict=False,
)

# 4. River Sensor Telemetry Contract
RiverTelemetrySchema = DataFrameSchema(
    columns={
        "site_id": Column(str, Check.isin(["06892350", "06893000", "06893060"]), nullable=False, coerce=True),
        "streamflow_cfs": Column(float, Check.greater_than_or_equal_to(0), nullable=True, coerce=True),
        "turbidity_fnu": Column(float, Check.greater_than_or_equal_to(0), nullable=True, coerce=True),
        "water_temperature_c": Column(float, Check.in_range(-10.0, 50.0), nullable=True, coerce=True),
        "specific_conductance_us_cm": Column(float, Check.greater_than_or_equal_to(0), nullable=True, coerce=True),
    },
    strict=False,
)

# 5. Regulatory Violation Contract
ViolationSchema = DataFrameSchema(
    columns={
        "violation_id": Column(str, nullable=False, coerce=True),
        "rule_name": Column(str, nullable=False, coerce=True),
        "compliance_status": Column(str, nullable=False, coerce=True),
    },
    strict=False,
)

# 6. Site Visit / Sanitary Survey Contract
SiteVisitSchema = DataFrameSchema(
    columns={
        "visit_id": Column(str, nullable=False, coerce=True),
        "visit_reason": Column(str, nullable=False, coerce=True),
    },
    strict=False,
)


class DataQualityGate:
    """Executes Pandera quality checks across Silver Star Schema datasets."""

    def __init__(self, db_path: str = "database/water_lakehouse.duckdb"):
        self.db_path = Path(db_path)
        self._init_audit_table()

    def _init_audit_table(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect(str(self.db_path)) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS silver_quality_audit (
                    check_id VARCHAR,
                    table_name VARCHAR,
                    checked_at TIMESTAMP,
                    records_checked BIGINT,
                    is_valid BOOLEAN,
                    error_count BIGINT,
                    error_details VARCHAR
                );
                """
            )

    def validate(self, df: pd.DataFrame, schema: DataFrameSchema, table_name: str) -> Tuple[bool, List[str]]:
        """
        Validates DataFrame against Pandera schema.
        Returns (is_valid, list_of_error_messages).
        """
        errors = []
        is_valid = True
        try:
            schema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as err:
            is_valid = False
            for _, row in err.failure_cases.iterrows():
                col = row.get("column", "DataFrame")
                check = row.get("check", "Schema")
                val = row.get("failure_case", "")
                errors.append(f"Column '{col}' failed check '{check}' on value: {val}")
        except Exception as e:
            is_valid = False
            errors.append(str(e))

        # Record in DuckDB audit table
        now = datetime.now(timezone.utc)
        import uuid
        check_id = str(uuid.uuid4())[:8]
        error_details = "; ".join(errors[:10]) if errors else None

        with duckdb.connect(str(self.db_path)) as con:
            con.execute(
                """
                INSERT INTO silver_quality_audit (
                    check_id, table_name, checked_at, records_checked, is_valid, error_count, error_details
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    check_id,
                    table_name,
                    now,
                    len(df),
                    is_valid,
                    len(errors),
                    error_details,
                ],
            )

        return is_valid, errors

