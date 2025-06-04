from dataclasses import dataclass
from typing import ClassVar

from loguru import logger

from src.pipelines.feature_pipeline.movies_client import (
    MoviesAPIClient,
    MoviesAPIConfig,
)
from src.utils.feature_store_interface import FeatureStoreInterface


@dataclass
class FeaturePipelineConfig:
    """Configuration for movie feature pipeline.

    Handles configuration validation and defaults for the movie feature pipeline.
    Supports both initial and incremental loading modes.

    Attributes:
        ALLOWED_TYPES: Valid load types for the pipeline.
        api_token: TMDb API authentication token.
        feature_group: Name of the feature group/table.
        type: Load type (initial/incremental).
        pages: Number of pages to fetch from API.
        timeout: API request timeout in seconds.
        HTTP_OKAY: Success HTTP status code.

    Raises:
        ValueError: If provided load type is not in ALLOWED_TYPES.
    """

    ERR_INVALID_TYPE: ClassVar[str] = "Invalid load type: {}. Must be one of: {}"

    ALLOWED_TYPES: ClassVar[list[str]] = ["initial", "incremental"]

    api_token: str
    feature_group: str
    type: list[str]
    pages: int = 400  # number of pages to read
    timeout: int = 10  # seconds
    HTTP_OKAY: int = 200

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.type not in self.ALLOWED_TYPES:
            raise ValueError(self.ERR_INVALID_TYPE.format(self.type, ", ".join(self.ALLOWED_TYPES)))


class MovieFeaturePipeline:
    """Pipeline for loading movie features from TMDb API.

    Handles fetching, transforming and storing movie data in feature store.
    Supports both initial and incremental loading patterns.

    Attributes:
        config: Pipeline configuration parameters.
    """

    def __init__(self, config: FeaturePipelineConfig):
        self.config = config

    def run(self, feature_store: FeatureStoreInterface) -> None:
        """Execute the feature pipeline.

        Orchestrates the complete pipeline execution:
        1. Checks for existing movies (incremental mode)
        2. Configures API client
        3. Fetches movies and their details
        4. Stores results in feature store

        Args:
            feature_store: Storage implementation for features.

        Raises:
            ValueError: If API configuration is invalid.
        """
        logger.info(
            f"\nStarting Feature Pipeline:\n"
            f"- Load type: {self.config.type}\n"
            f"- Pages to process: {self.config.pages}\n"
            f"- Feature group: {self.config.feature_group}"
        )

        existing_ids: set[int] | None = (
            feature_store.fetch_existing_movie_ids(self.config.feature_group)
            if self.config.type == "incremental"
            else None
        )
        api_config: MoviesAPIConfig = MoviesAPIConfig(
            token=self.config.api_token,
            feature_group=self.config.feature_group,
            pages=self.config.pages,  # Adjust as needed
        )

        _: MoviesAPIClient = (
            MoviesAPIClient(api_config, existing_ids)
            .fetch_popular_movie_ids()
            .fetch_movie_extended_info()
            .store_movies_features(feature_store)
        )
