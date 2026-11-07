"""Automated integration tests for Pipeline Orchestrator and CLI Suite."""
from pathlib import Path
from click.testing import CliRunner
import pandas as pd
import pytest

from src.pipeline import WaterOnePipeline
from src.cli import main


def test_pipeline_status():
    pipeline = WaterOnePipeline()
    stat = pipeline.status()

    assert stat.database_exists
    assert stat.database_size_bytes > 0
    assert "bronze_facilities" in stat.layer_counts
    assert "dim_facility" in stat.layer_counts
    assert "mart_daily_water_quality" in stat.layer_counts
    assert stat.layer_counts["dim_facility"] > 0


def test_pipeline_query():
    pipeline = WaterOnePipeline()
    df = pipeline.execute_query("SELECT count(*) as cnt FROM dim_facility;")
    assert not df.empty
    assert "cnt" in df.columns
    assert df.iloc[0]["cnt"] == 95


def test_pipeline_export(tmp_path):
    pipeline = WaterOnePipeline()
    csv_out = tmp_path / "daily.csv"
    json_out = tmp_path / "daily.json"

    pipeline.export_mart("mart_daily_water_quality", str(csv_out), format_type="csv")
    assert csv_out.exists()
    assert csv_out.stat().st_size > 0

    pipeline.export_mart("mart_daily_water_quality", str(json_out), format_type="json")
    assert json_out.exists()
    assert json_out.stat().st_size > 0


def test_cli_status():
    runner = CliRunner()
    result = runner.invoke(main, ["status"])
    assert result.exit_code == 0
    assert "WaterOne Data Engine Catalog" in result.output
    assert "dim_facility" in result.output


def test_cli_validate():
    runner = CliRunner()
    result = runner.invoke(main, ["validate"])
    assert result.exit_code == 0
    assert "Pandera Quality Contracts Verification" in result.output


def test_cli_detect():
    runner = CliRunner()
    result = runner.invoke(main, ["detect", "--contamination", "0.05"])
    assert result.exit_code == 0
    assert "Anomaly Detection Complete" in result.output


def test_cli_query():
    runner = CliRunner()
    result = runner.invoke(main, ["query", "SELECT count(*) FROM dim_contaminant;"])
    assert result.exit_code == 0
    assert "count_star()" in result.output
    assert "10" in result.output


def test_cli_export(tmp_path):
    runner = CliRunner()
    out_file = tmp_path / "test_mart.csv"
    result = runner.invoke(main, ["export", "mart_daily_water_quality", str(out_file), "--format", "csv"])
    assert result.exit_code == 0
    assert out_file.exists()
