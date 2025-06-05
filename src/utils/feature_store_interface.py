from abc import ABC, abstractmethod

from pandas import DataFrame


class FeatureStoreInterface(ABC):
    """Interface for feature store implementations."""

    @abstractmethod
    def insert(self, feature_group: str, features: DataFrame, mode: str) -> None:
        """Insert features into the store.

        Args:
            feature_group: Name of the feature group/table
            features: DataFrame containing features to store
        """
        ...

    @abstractmethod
    def fetch_existing_movie_ids(self, feature_group: str) -> set:
        """Insert features into the store.

        Args:
            feature_group: Name of the feature group/table
            features: DataFrame containing features to store
        """
        ...

    @abstractmethod
    def query_features(self, feature_group: str, columns: list[str] | None = None) -> DataFrame: ...
