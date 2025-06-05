from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from src.pipelines.feature_pipeline.movies_client import (
    MoviesAPIClient,
    MoviesAPIConfig,
)

TEST_TOKEN: str = "dummy_token"  # noqa: S105
TEST_FEATURE_GROUP: str = "test_movies"
TEST_VOTE_AVERAGE: float = 8.5
TEST_RUNTIME: int = 120
TEST_BUDGET: int = 1_000_000


@pytest.fixture
def config() -> MoviesAPIConfig:
    return MoviesAPIConfig(
        token=TEST_TOKEN,
        feature_group=TEST_FEATURE_GROUP,
        pages=1,
        timeout=1,
    )


@pytest.fixture
def mock_base_response() -> MagicMock:
    """Mock response for base movie data."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "results": [
            {
                "id": 1,
                "adult": False,
                "original_language": "en",
                "original_title": "Test Movie",
                "overview": "Test overview",
                "popularity": 100.0,
                "vote_average": 8.5,
                "vote_count": 1000,
                "release_date": "2024-01-01",
            }
        ]
    }
    return mock


@pytest.fixture
def mock_extended_response() -> MagicMock:
    """Mock response for extended movie info."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "runtime": 120,
        "budget": 1000000,
        "revenue": 2000000,
        "status": "Released",
        "tagline": "Test tagline",
        "genres": [{"name": "Action"}],
        "spoken_languages": [{"name": "English"}],
    }
    return mock


@pytest.fixture
def mock_popular_response() -> MagicMock:
    """Mock response for popular movies API."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {
        "results": [
            {"id": 1, "title": "Popular Movie 1"},
            {"id": 2, "title": "Popular Movie 2"},
        ]
    }
    return mock


@pytest.fixture
def mock_get() -> Generator:
    """Mock requests.get calls."""
    with patch("requests.get") as mock:
        yield mock


def test_build_movie_base_success(
    mock_get: MagicMock,
    config: MoviesAPIConfig,
    mock_base_response: MagicMock,
) -> None:
    """Test successful movie base data fetch."""
    mock_get.return_value = mock_base_response

    client = MoviesAPIClient(config)
    assert client.movies is not None, "Movies DataFrame should not be None"
    assert not client.movies.empty
    assert len(client.movies) == 1
    assert client.movies.iloc[0]["original_title"] == "Test Movie"
    assert client.movies.iloc[0]["vote_average"] == TEST_VOTE_AVERAGE


def test_build_movie_base_http_error(mock_get: MagicMock, config: MoviesAPIConfig) -> None:
    """Test handling of HTTP error in movie fetch."""
    mock = MagicMock()
    mock.status_code = 404
    mock_get.return_value = mock

    with pytest.raises(ValueError, match=MoviesAPIClient.ERR_NO_MOVIES):
        MoviesAPIClient(config)


def test_fetch_extended_info(
    mock_get: MagicMock,
    config: MoviesAPIConfig,
    mock_base_response: MagicMock,
    mock_extended_response: MagicMock,
) -> None:
    """Test fetching extended movie information."""
    mock_get.side_effect = [mock_base_response, mock_extended_response]

    client = MoviesAPIClient(config)
    client.fetch_movie_extended_info()

    assert client.movies is not None, "Movies DataFrame should not be None"
    assert client.movies.iloc[0]["runtime"] == TEST_RUNTIME
    assert client.movies.iloc[0]["budget"] == TEST_BUDGET
    assert isinstance(client.movies.iloc[0]["genres"], list)


def test_fetch_popular_movie_ids(
    mock_get: MagicMock,
    config: MoviesAPIConfig,
    mock_popular_response: MagicMock,
    mock_base_response: MagicMock,
) -> None:
    """Test fetching and marking popular movies."""
    mock_get.side_effect = [mock_base_response, mock_popular_response]

    client = MoviesAPIClient(config)
    client.fetch_popular_movie_ids()

    assert client.movies is not None, "Movies DataFrame should not be None"
    assert "is_popular" in client.movies.columns


def test_fetch_popular_movie_ids_error(
    mock_get: MagicMock,
    config: MoviesAPIConfig,
    mock_base_response: MagicMock,
) -> None:
    """Test handling API errors in popular movies fetch."""
    error_response = MagicMock()
    error_response.status_code = 500
    mock_get.side_effect = [mock_base_response, error_response]

    client = MoviesAPIClient(config)
    client.fetch_popular_movie_ids()

    assert client.movies is not None, "Movies DataFrame should not be None"
    assert "is_popular" in client.movies.columns
    assert not client.movies["is_popular"].iloc[0]


def test_add_extraction_date(
    mock_get: MagicMock,
    config: MoviesAPIConfig,
    mock_base_response: MagicMock,
) -> None:
    """Test adding extraction date to movies DataFrame."""
    mock_get.return_value = mock_base_response

    client = MoviesAPIClient(config)
    client.add_extraction_date()

    assert client.movies is not None, "Movies DataFrame should not be None"
    assert "extraction_date" in client.movies.columns
    assert isinstance(client.movies.iloc[0]["extraction_date"], str)


def test_skip_existing_movies(
    mock_get: MagicMock,
    config: MoviesAPIConfig,
    mock_base_response: MagicMock,
) -> None:
    """Test skipping existing movies in incremental mode."""
    mock_get.return_value = mock_base_response
    existing_ids = {1}  # Movie ID from mock_response

    with pytest.raises(ValueError, match=MoviesAPIClient.ERR_NO_MOVIES):
        MoviesAPIClient(config, existing_ids=existing_ids)
