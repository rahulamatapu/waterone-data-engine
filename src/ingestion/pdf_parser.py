"""Dynamic template-agnostic CCR water quality PDF extractor."""
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import pdfplumber
import pandas as pd

from src.ingestion.audit import AuditManager, compute_sha256


def clean_param_name(name: str) -> str:
    """Strips footnote asterisks and numbers from parameter names."""
    if not name:
        return ""
    # Remove newlines
    name = name.replace("\n", " ").strip()
    # Remove trailing asterisks and footnote numbers like 'Lead 1' or 'Aluminum*'
    name = re.sub(r"[\*\†\‡]+$", "", name).strip()
    name = re.sub(r"\s+\d+$", "", name).strip()
    return name


def extract_numeric_and_unit(text: Optional[str]) -> Tuple[Optional[float], Optional[str]]:
    """
    Extracts numeric value and unit from strings like '4.1 ppb 2', '0.02 ppm', '0.23 NTU 5'.
    """
    if not text or text.lower() in {"n/a", "na", "none", "", "tt", "nd"}:
        return None, None

    # Handle strings like 'AL=1.3 ppm' or 'MRDL=4 ppm'
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(ppb|ppm|mg/l|ug/l|ntu|pci/l|s\.i\.|%)?", text, flags=re.IGNORECASE)
    if match:
        val = float(match.group(1))
        unit = match.group(2).lower() if match.group(2) else None
        if unit == "s.i.":
            unit = "S.I."
        elif unit == "pci/l":
            unit = "pCi/L"
        elif unit:
            unit = unit.lower()
        return val, unit
    return None, None


def extract_range(text: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """
    Extracts min and max numeric values from strings like 'ND (1.0) ppb - 1.8 ppb' or '46 ppm - 83 ppm'.
    """
    if not text:
        return None, None
    numbers = re.findall(r"([0-9]+(?:\.[0-9]+)?)", text)
    if not numbers:
        return None, None
    nums = [float(n) for n in numbers]
    if len(nums) == 1:
        return nums[0], nums[0]
    return min(nums), max(nums)


class CCRParser:
    """Dynamic anchor-based table extractor for annual Consumer Confidence Reports."""

    def __init__(
        self,
        landing_dir: str = "data/landing/ccr",
        bronze_dir: str = "data/bronze",
        audit_manager: Optional[AuditManager] = None,
    ):
        self.landing_dir = Path(landing_dir)
        self.bronze_dir = Path(bronze_dir)
        self.bronze_dir.mkdir(parents=True, exist_ok=True)
        self.audit = audit_manager or AuditManager()

    def parse_pdf(self, file_path: Path, force: bool = False) -> Dict[str, any]:
        """Extracts regulated and secondary contaminant tables from a CCR PDF."""
        file_hash = compute_sha256(file_path)
        is_ingested, _ = self.audit.is_already_ingested(file_path)

        if is_ingested and not force:
            return {
                "file": file_path.name,
                "status": "SKIPPED",
                "reason": "Already ingested with identical SHA-256",
                "row_count": 0,
            }

        output_parquet = self.bronze_dir / "bronze_pdf_tables.parquet"

        try:
            records = []
            year_match = re.search(r"20\d\d", file_path.name)
            reporting_year = int(year_match.group(0)) if year_match else datetime.now().year

            with pdfplumber.open(file_path) as pdf:
                for page_idx, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    tables = page.extract_tables()
                    if not tables:
                        continue

                    # Look for regulated parameters table
                    if "REGULATED PARAMETERS" in text.upper():
                        current_category = "Regulated Contaminant"
                        for t in tables:
                            for row in t:
                                if not row or not row[0]:
                                    continue
                                first_col = str(row[0]).strip()
                                
                                # Skip header row
                                if "REGULATED PARAMETERS" in first_col.upper() or "PARAMETER" in first_col.upper():
                                    continue
                                
                                # Category divider rows (e.g. Inorganic Contaminants, Disinfectants)
                                if len(row) > 1 and all(c is None or str(c).strip() == "" for c in row[1:]):
                                    current_category = f"Regulated - {first_col}"
                                    continue

                                param_name = clean_param_name(first_col)
                                mcl_raw = str(row[1]).strip() if len(row) > 1 and row[1] else None
                                mclg_raw = str(row[2]).strip() if len(row) > 2 and row[2] else None
                                result_raw = str(row[3]).strip() if len(row) > 3 and row[3] else None
                                range_raw = str(row[4]).strip() if len(row) > 4 and row[4] else None
                                freq = str(row[5]).strip() if len(row) > 5 and row[5] else None
                                source_desc = str(row[7]).strip() if len(row) > 7 and row[7] else None

                                res_val, unit = extract_numeric_and_unit(result_raw)
                                mcl_val, mcl_unit = extract_numeric_and_unit(mcl_raw)
                                mclg_val, _ = extract_numeric_and_unit(mclg_raw)
                                r_min, r_max = extract_range(range_raw)

                                records.append({
                                    "contaminant": first_col,
                                    "parameter_clean": param_name,
                                    "category": current_category,
                                    "mcl_raw": mcl_raw,
                                    "mcl_value": mcl_val,
                                    "mclg_raw": mclg_raw,
                                    "mclg_value": mclg_val,
                                    "waterone_result_raw": result_raw,
                                    "waterone_result_value": res_val,
                                    "unit": unit or mcl_unit or "ppm",
                                    "waterone_range_raw": range_raw,
                                    "range_min": r_min,
                                    "range_max": r_max,
                                    "sample_frequency": freq,
                                    "source_description": source_desc,
                                    "reporting_year": reporting_year,
                                    "page_number": page_idx + 1,
                                })

                    # Look for Secondary / Unregulated parameters tables
                    elif "SECONDARY MCL" in text.upper() or "UNREGULATED PARAMETERS" in text.upper():
                        for t in tables:
                            for row in t:
                                if not row or not row[0]:
                                    continue
                                first_col = str(row[0]).strip()
                                if "PARAMETER" in first_col.upper():
                                    continue
                                param_name = clean_param_name(first_col)
                                
                                category = "Secondary Standards" if len(row) == 4 and "SECONDARY" in text.upper() else "Unregulated Parameters"
                                smcl_raw = str(row[1]).strip() if len(row) > 1 and row[1] else None
                                result_raw = str(row[2]).strip() if len(row) > 2 and row[2] else None
                                range_raw = str(row[3]).strip() if len(row) > 3 and row[3] else None

                                res_val, unit = extract_numeric_and_unit(result_raw)
                                smcl_val, smcl_unit = extract_numeric_and_unit(smcl_raw)
                                r_min, r_max = extract_range(range_raw)

                                records.append({
                                    "contaminant": first_col,
                                    "parameter_clean": param_name,
                                    "category": category,
                                    "mcl_raw": smcl_raw,
                                    "mcl_value": smcl_val,
                                    "mclg_raw": None,
                                    "mclg_value": None,
                                    "waterone_result_raw": result_raw,
                                    "waterone_result_value": res_val,
                                    "unit": unit or smcl_unit or "ppm",
                                    "waterone_range_raw": range_raw,
                                    "range_min": r_min,
                                    "range_max": r_max,
                                    "sample_frequency": "Periodic",
                                    "source_description": "Secondary aesthetic or unregulated monitoring",
                                    "reporting_year": reporting_year,
                                    "page_number": page_idx + 1,
                                })

            df = pd.DataFrame(records)
            now_iso = datetime.now(timezone.utc).isoformat()
            df["_source_file"] = file_path.name
            df["_ingested_at"] = now_iso
            df["_file_hash"] = file_hash

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
                "table": "bronze_pdf_tables",
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
        """Extracts tables from all CCR PDFs in landing dropzone."""
        results = []
        if not self.landing_dir.exists():
            return results
        pdf_files = sorted(list(self.landing_dir.glob("*.pdf")))
        for pdf in pdf_files:
            results.append(self.parse_pdf(pdf, force=force))
        return results

