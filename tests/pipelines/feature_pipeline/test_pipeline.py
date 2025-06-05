from unittest.mock import MagicMock

import pytest
from pandas import DataFrame

from src.pipelines.feature_pipeline.movies_client import (
    MoviesAPIClient,
    MoviesAPIConfig,
)
from src.pipelines.feature_pipeline.pipeline import (
    FeaturePipelineConfig,
    MovieFeaturePipeline,
)
from src.utils.feature_store_interface import FeatureStoreInterface

# Test constants
TEST_TOKEN: str = "dummy_token"  # noqa: S105
TEST_FEATURE_GROUP: str = "test_movies"
TEST_PAGES: int = 1


@pytest.fixture
def feature_store() -> MagicMock:
    """Create mock feature store."""
    mock = MagicMock(spec=FeatureStoreInterface)
    mock.fetch_existing_movie_ids.return_value = {1, 2, 3}  # Simulate existing movies
    return mock


@pytest.fixture
def initial_config() -> FeaturePipelineConfig:
    """Create initial load config."""
    return FeaturePipelineConfig(
        api_token=TEST_TOKEN,
        feature_group=TEST_FEATURE_GROUP,
        type="initial",
        pages=TEST_PAGES,
    )


@pytest.fixture
def incremental_config() -> FeaturePipelineConfig:
    """Create incremental load config."""
    return FeaturePipelineConfig(
        api_token=TEST_TOKEN,
        feature_group=TEST_FEATURE_GROUP,
        type="incremental",
        pages=TEST_PAGES,
    )


@pytest.fixture
def movies_df() -> DataFrame:
    """Create sample movies DataFrame."""
    return DataFrame(
        {
            "id": [1, 2],
            "title": ["Movie 1", "Movie 2"],
            "vote_average": [7.5, 8.0],
        }
    )


def test_feature_pipeline_config_validation() -> None:
    """Test pipeline configuration validation."""
    with pytest.raises(ValueError, match="Invalid load type"):
        FeaturePipelineConfig(
            api_token=TEST_TOKEN,
            feature_group=TEST_FEATURE_GROUP,
            type="invalid_type",
        )


def test_pipeline_initial_load(
    initial_config: FeaturePipelineConfig,
    feature_store: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test pipeline execution in initial load mode."""
    # Mock MoviesAPIClient
    mock_client = MagicMock(spec=MoviesAPIClient)
    mock_client.fetch_popular_movie_ids.return_value = mock_client
    mock_client.fetch_movie_extended_info.return_value = mock_client
    mock_client.store_movies_features.return_value = mock_client

    # Mock client creation
    def mock_init(config: MoviesAPIConfig, existing_ids: set[int] | None = None) -> MagicMock:
        assert existing_ids is None  # Should be None for initial load
        return mock_client

    monkeypatch.setattr(MoviesAPIClient, "__init__", mock_init)
    monkeypatch.setattr(MoviesAPIClient, "__new__", lambda cls, *args, **kwargs: mock_client)

    # Execute pipeline
    pipeline = MovieFeaturePipeline(initial_config)
    pipeline.run(feature_store)

    # Verify feature store was not queried for existing IDs
    feature_store.fetch_existing_movie_ids.assert_not_called()

    # Verify client method calls
    mock_client.fetch_popular_movie_ids.assert_called_once()
    mock_client.fetch_movie_extended_info.assert_called_once()
    mock_client.store_movies_features.assert_called_once_with(feature_store)


def test_pipeline_incremental_load(
    incremental_config: FeaturePipelineConfig,
    feature_store: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test pipeline execution in incremental load mode."""
    # Mock MoviesAPIClient
    mock_client = MagicMock(spec=MoviesAPIClient)
    mock_client.fetch_popular_movie_ids.return_value = mock_client
    mock_client.fetch_movie_extended_info.return_value = mock_client
    mock_client.store_movies_features.return_value = mock_client

    # Mock client creation
    def mock_init(config: MoviesAPIConfig, existing_ids: set[int] | None = None) -> MagicMock:
        assert existing_ids == {1, 2, 3}  # Should match feature store mock
        return mock_client

    monkeypatch.setattr(MoviesAPIClient, "__init__", mock_init)
    monkeypatch.setattr(MoviesAPIClient, "__new__", lambda cls, *args, **kwargs: mock_client)

    # Execute pipeline
    pipeline = MovieFeaturePipeline(incremental_config)
    pipeline.run(feature_store)

    # Verify feature store was queried for existing IDs
    feature_store.fetch_existing_movie_ids.assert_called_once_with(TEST_FEATURE_GROUP)

    # Verify client method calls
    mock_client.fetch_popular_movie_ids.assert_called_once()
    mock_client.fetch_movie_extended_info.assert_called_once()
    mock_client.store_movies_features.assert_called_once_with(feature_store)


def test_pipeline_config_validation(initial_config: FeaturePipelineConfig) -> None:
    """Test pipeline configuration validation."""
    # Valid config should not raise
    pipeline = MovieFeaturePipeline(initial_config)
    assert pipeline.config == initial_config

    # Invalid config should raise
    with pytest.raises(ValueError, match="Invalid load type"):
        FeaturePipelineConfig(
            api_token=TEST_TOKEN,
            feature_group=TEST_FEATURE_GROUP,
            type="invalid",
        )
