"""Pytest session fixtures for WaterOne Data Engine test suite."""
import pytest

from src.ingestion.orchestrator import BronzeIngestionOrchestrator
from src.transformation.dimensional_models import SilverDimensionalBuilder
from src.transformation.feature_store import GoldFeatureStoreBuilder
from src.models.anomaly_detector import WaterQualityAnomalyDetector


@pytest.fixture(scope="session", autouse=True)
def setup_pipeline_test_environment():
    """
    Ensures Bronze, Silver, and Gold lakehouse tables are initialized
    before any unit tests run in clean CI environments.
    """
    # 1. Populate Bronze
    ingestion = BronzeIngestionOrchestrator()
    ingestion.run(force=True)

    # 2. Populate Silver
    silver = SilverDimensionalBuilder()
    silver.run_all()

    # 3. Populate Gold
    gold = GoldFeatureStoreBuilder()
    gold.run_all()

    # 4. Populate Anomaly Detector
    detector = WaterQualityAnomalyDetector()
    detector.train_and_score()

