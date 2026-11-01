"""SHA-256 fingerprinting and deduplication audit manager for the Bronze layer."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import duckdb
import pandas as pd


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash digest for a given file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class AuditManager:
    """Manages Bronze ingestion audit log in DuckDB to ensure idempotency."""

    def __init__(self, db_path: str = "database/water_lakehouse.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.db_path))

    def _init_db(self) -> None:
        """Initializes the bronze_audit_log table if it does not exist."""
        with self._get_connection() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS bronze_audit_log (
                    file_hash VARCHAR PRIMARY KEY,
                    file_name VARCHAR,
                    source_path VARCHAR,
                    layer VARCHAR,
                    ingested_at TIMESTAMP,
                    status VARCHAR,
                    row_count BIGINT,
                    error_message VARCHAR
                );
                """
            )

    def is_already_ingested(self, file_path: Path) -> Tuple[bool, str]:
        """
        Checks if file has already been successfully ingested.
        Returns (is_ingested, file_hash).
        """
        file_hash = compute_sha256(file_path)
        with self._get_connection() as con:
            result = con.execute(
                """
                SELECT status FROM bronze_audit_log
                WHERE file_hash = ? AND status = 'SUCCESS'
                """,
                [file_hash],
            ).fetchone()
        return (result is not None, file_hash)

    def record_ingestion(
        self,
        file_hash: str,
        file_name: str,
        source_path: str,
        layer: str,
        status: str,
        row_count: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        """Records or updates an ingestion attempt in the audit log."""
        now = datetime.now(timezone.utc)
        with self._get_connection() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO bronze_audit_log (
                    file_hash, file_name, source_path, layer, ingested_at, status, row_count, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                [
                    file_hash,
                    file_name,
                    source_path,
                    layer,
                    now,
                    status,
                    row_count,
                    error_message,
                ],
            )

    def get_audit_log(self, limit: int = 50) -> pd.DataFrame:
        """Fetches the latest audit log records."""
        with self._get_connection() as con:
            return con.execute(
                """
                SELECT file_hash, file_name, layer, ingested_at, status, row_count, error_message
                FROM bronze_audit_log
                ORDER BY ingested_at DESC
                LIMIT ?
                """,
                [limit],
            ).df()

