"""Baseline environment and configuration sanity tests."""
from pathlib import Path
import yaml


def test_pipeline_config_loads():
    config_path = Path("config/pipeline_config.yaml")
    assert config_path.exists(), "pipeline_config.yaml does not exist"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    assert cfg["water_system"]["pws_id"] == "KS2009110"
    assert "usgs_telemetry" in cfg
    assert "stations" in cfg["usgs_telemetry"]


def test_contaminant_mapping_loads():
    mapping_path = Path("config/contaminant_mapping.yaml")
    assert mapping_path.exists(), "contaminant_mapping.yaml does not exist"
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = yaml.safe_load(f)
    assert "contaminants" in mapping
    assert "lead" in mapping["contaminants"]
    assert "turbidity" in mapping["contaminants"]


def test_core_imports():
    import duckdb
    import pandas
    import pandera
    import pyarrow
    import sklearn

    assert duckdb is not None
    assert pandas is not None
    assert pandera is not None
    assert pyarrow is not None
    assert sklearn is not None

