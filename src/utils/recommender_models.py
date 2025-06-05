from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from pandas import DataFrame
from sklearn.pipeline import Pipeline

from src.utils.feature_store_interface import FeatureStoreInterface


@dataclass
class RecommenderModelConfig:
    model_name: str
    feature_store: FeatureStoreInterface
    training_feature_group: str  # TODO: Improve this, with feature group object
    similarity_matrix_group: str  # TODO: Improve this, with feature group object
    required_features: list[str]
    transformation_pipeline: Pipeline
    model: Callable


class RecommenderModel:
    ERR_NO_FEATURES: ClassVar[str] = "No features found for {model_name} model"
    ERR_NOT_FITTED: ClassVar[str] = (
        "Model has not been fitted yet. Call fit() before storing outputs"
    )

    def __init__(
        self,
        config: RecommenderModelConfig,
    ):
        self.config = config
        self.name = self.config.model_name
        self.similarity_matrix: DataFrame | None = None

    def __fetch_features(self) -> DataFrame:
        features: DataFrame = self.config.feature_store.query_features(
            feature_group=self.config.training_feature_group,
            columns=self.config.required_features,
        )
        if features.empty:
            raise ValueError(self.ERR_NO_FEATURES.format(model_name=self.config.model_name))

        return features

    def fit(self) -> "RecommenderModel":
        # this use of feature goups is tech debt, better to create an object for each feature group
        features: DataFrame = self.__fetch_features()
        features.drop_duplicates(
            inplace=True, subset=features.select_dtypes(exclude=["object"]).columns
        )
        dtype_optimized_features: DataFrame = features.convert_dtypes()

        preprocessed_features: DataFrame = self.config.transformation_pipeline.fit_transform(
            dtype_optimized_features
        )
        self.similarity_matrix = DataFrame(
            self.config.model(preprocessed_features, preprocessed_features)
        )

        return self

    def store_outputs(self) -> "RecommenderModel":
        if self.similarity_matrix is None:
            raise ValueError(self.ERR_NOT_FITTED)

        self.config.feature_store.insert(
            feature_group=self.config.similarity_matrix_group,
            features=self.similarity_matrix,
            mode="replace",
        )
        return self
