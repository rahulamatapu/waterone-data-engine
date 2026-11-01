"""Header-detecting and normalizing CSV parser for KDHE state compliance reports."""
import csv
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import pandas as pd

from src.ingestion.audit import AuditManager, compute_sha256


def to_snake_case(text: str) -> str:
    """Converts a column header into standardized clean snake_case."""
    # Replace '#' with 'num_'
    text = re.sub(r"#\s*of\b", "num_", text, flags=re.IGNORECASE)
    text = text.replace("#", "num_")
    # Remove apostrophes, quotes, parens
    text = re.sub(r"['\"()\[\]]", "", text)
    # Replace non-alphanumeric with underscore
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    # Strip leading/trailing underscores and lowercase
    text = text.strip("_").lower()
    return text


def detect_header_row(file_path: Path) -> Tuple[int, List[str]]:
    """
    Dynamically detects the 0-based header row index and column list
    in a CSV file by scanning preamble rows.
    """
    known_header_anchors = {
        "pws id",
        "submission year quarter",
        "facility id",
        "visit id",
        "violation id",
        "population served count",
    }

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if not row:
                continue
            cleaned = [c.strip().lower() for c in row if c.strip()]
            # If row contains any known anchor keyword and has at least 4 non-empty columns
            if len(cleaned) >= 4 and any(
                anchor in c for anchor in known_header_anchors for c in cleaned
            ):
                return idx, [c.strip() for c in row]

    # Fallback: line with highest number of non-empty columns in first 15 lines
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        best_idx, best_cols, max_len = 0, [], 0
        for idx in range(15):
            try:
                row = next(reader)
            except StopIteration:
                break
            non_empty = [c.strip() for c in row if c.strip()]
            if len(non_empty) > max_len:
                max_len = len(non_empty)
                best_idx = idx
                best_cols = [c.strip() for c in row]
        return best_idx, best_cols


class CSVNormalizer:
    """Ingests and normalizes raw KDHE CSV files into Bronze Parquet tables."""

    def __init__(
        self,
        landing_dir: str = "data/landing/kdhe",
        bronze_dir: str = "data/bronze",
        audit_manager: Optional[AuditManager] = None,
    ):
        self.landing_dir = Path(landing_dir)
        self.bronze_dir = Path(bronze_dir)
        self.bronze_dir.mkdir(parents=True, exist_ok=True)
        self.audit = audit_manager or AuditManager()

    def _determine_table_type(self, file_name: str) -> str:
        """Determines the Bronze table name from the file name."""
        name_lower = file_name.lower()
        if "facility" in name_lower:
            return "bronze_facilities"
        elif "violation" in name_lower:
            return "bronze_violations"
        elif "site visit" in name_lower or "visit" in name_lower:
            return "bronze_site_visits"
        elif "summary" in name_lower:
            return "bronze_summary"
        else:
            clean_base = re.sub(r"[^a-zA-Z0-9]+", "_", file_name.split(".")[0]).strip("_").lower()
            return f"bronze_{clean_base}"

    def process_file(self, file_path: Path, force: bool = False) -> Dict[str, any]:
        """Processes a single KDHE CSV file into Bronze Parquet."""
        file_hash = compute_sha256(file_path)
        is_ingested, _ = self.audit.is_already_ingested(file_path)

        if is_ingested and not force:
            return {
                "file": file_path.name,
                "status": "SKIPPED",
                "reason": "Already ingested with identical SHA-256",
                "row_count": 0,
            }

        table_name = self._determine_table_type(file_path.name)
        output_parquet = self.bronze_dir / f"{table_name}.parquet"

        try:
            header_row_idx, raw_headers = detect_header_row(file_path)
            
            # Read CSV skipping header rows
            df = pd.read_csv(
                file_path,
                skiprows=header_row_idx,
                dtype=str,
                encoding="utf-8",
                on_bad_lines="skip",
            )

            # Clean and normalize column names
            snake_columns = [to_snake_case(c) for c in df.columns]
            
            # Deduplicate column names if any
            seen = {}
            deduped = []
            for col in snake_columns:
                if col in seen:
                    seen[col] += 1
                    deduped.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    deduped.append(col)
            df.columns = deduped

            # Clean sentinel dates and empty whitespace
            sentinel_dates = {"31-dec-1899", "1899-12-31", "12/31/1899", "nan", "none", "<na>", ""}
            for col in df.columns:
                df[col] = df[col].apply(
                    lambda x: None if pd.isna(x) or str(x).strip().lower() in sentinel_dates else str(x).strip()
                )

            # Add audit lineage metadata
            now_iso = datetime.now(timezone.utc).isoformat()
            df["_source_file"] = file_path.name
            df["_ingested_at"] = now_iso
            df["_file_hash"] = file_hash

            # Write to Bronze Parquet
            df.to_parquet(output_parquet, index=False, engine="pyarrow")

            row_count = len(df)
            self.audit.record_ingestion(
                file_hash=file_hash,
                file_name=file_path.name,
                source_path=str(file_path),
                layer="BRONZE",
                status="SUCCESS",
                row_count=row_count,
            )

            return {
                "file": file_path.name,
                "table": table_name,
                "output": str(output_parquet),
                "status": "SUCCESS",
                "row_count": row_count,
            }
        except Exception as e:
            self.audit.record_ingestion(
                file_hash=file_hash,
                file_name=file_path.name,
                source_path=str(file_path),
                layer="BRONZE",
                status="FAILED",
                row_count=0,
                error_message=str(e),
            )
            return {
                "file": file_path.name,
                "status": "FAILED",
                "error": str(e),
                "row_count": 0,
            }

    def process_all(self, force: bool = False) -> List[Dict[str, any]]:
        """Processes all CSV files in the landing KDHE directory."""
        results = []
        if not self.landing_dir.exists():
            return results

        csv_files = sorted(list(self.landing_dir.glob("*.csv")))
        for f in csv_files:
            results.append(self.process_file(f, force=force))
        return results
