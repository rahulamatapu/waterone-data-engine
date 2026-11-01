# WaterOne Data Engine (`waterone-data-engine`)

An end-to-end, zero-cost water quality lakehouse and telemetry platform for **Water District No. 1 of Johnson County, Kansas (PWS ID: KS2009110)**.

## Architecture

- **Storage & OLAP**: DuckDB + Apache Parquet (Medallion Architecture: Landing $\rightarrow$ Bronze $\rightarrow$ Silver $\rightarrow$ Gold)
- **Data Quality**: Pandera schema contracts and boundary enforcement
- **Real-Time Telemetry**: USGS Water Data API (Missouri River & Kansas River)
- **Unstructured / Semi-Structured**: Dynamic anchor-based `pdfplumber` extractor for CCR PDFs + regex header-detecting parser for KDHE CSVs
- **ML Anomaly Detection**: Scikit-Learn Isolation Forest multi-parameter pre-breach warning engine
- **CLI Suite**: `waterpipe` (Click + Rich)
- **Interactive UI**: Streamlit Water Quality Health Monitor

## Quickstart

```bash
# Install dependencies
poetry install

# Run the complete pipeline (Ingest -> Validate -> Transform -> Detect)
poetry run waterpipe run

# Check pipeline status and table row counts
poetry run waterpipe status

# Launch the interactive Streamlit Health Monitor
poetry run waterpipe dashboard
```

## Directory Structure

```text
waterone-data-engine/
├── config/
│   ├── pipeline_config.yaml           # Source URLs, USGS station IDs, landing paths
│   └── contaminant_mapping.yaml       # Standardized parameter IDs and EPA limit definitions
├── data/
│   ├── landing/                       # Raw source files dropzone (CCR PDF, KDHE CSVs)
│   ├── bronze/                        # Raw Parquet tables + audit log
│   ├── silver/                        # Cleaned dimensional Star Schema Parquet tables
│   └── gold/                          # Analytical marts and ML feature store
├── database/
│   └── water_lakehouse.duckdb         # DuckDB lakehouse catalog
├── src/
│   ├── cli.py                         # waterpipe CLI entrypoint
│   ├── pipeline.py                    # Pipeline orchestrator
│   ├── ingestion/                     # Audit, PDF parser, CSV normalizer, USGS client
│   ├── quality/                       # Pandera contracts
│   ├── transformation/                # Dimensional models & feature store
│   └── models/                        # Isolation Forest anomaly detector
├── dashboard/
│   └── app.py                         # Streamlit interactive application
└── tests/                             # Pytest automated test suite
```

