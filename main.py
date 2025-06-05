from sklearn.metrics.pairwise import cosine_similarity, linear_kernel

from src.pipelines.feature_pipeline.pipeline import (
    FeaturePipelineConfig,
    MovieFeaturePipeline,
)
from src.pipelines.training_pipeline.movie_feature_preprocessor import (
    MovieFeaturePreprocessor,
)
from src.pipelines.training_pipeline.pipeline import MovieTrainPipeline
from src.utils.arg_parser import ArgParser, PipelineArgs
from src.utils.recommender_models import RecommenderModel, RecommenderModelConfig
from src.utils.sqlite_conn import SQLiteConn


def main() -> None:
    ERR_MISSING_TOKEN: str = "API token must be provided for feature pipeline"  # noqa: S105

    args: PipelineArgs = ArgParser.get()

    if args.pipeline_type == "feature":
        if not args.api_token:
            raise ValueError(ERR_MISSING_TOKEN)

        config = FeaturePipelineConfig(
            api_token=args.api_token,
            feature_group="movies",
            type=args.load_type,
            pages=args.pages,
        )
        feature_pipeline = MovieFeaturePipeline(config)
        feature_store = SQLiteConn(r"data/feature_store.sqlite")
        feature_pipeline.run(feature_store)

    elif args.pipeline_type == "train":
        cosine_config = RecommenderModelConfig(
            model_name="cosine-smilarity-movies",
            feature_store=SQLiteConn(r"data/feature_store.sqlite"),
            training_feature_group="movies",
            similarity_matrix_group="cosine_similarity_movies",
            required_features=[
                "original_title",
                "original_language",
                "popularity",
                "vote_average",
                "vote_count",
                "is_popular",
                "runtime",
                "budget",
                "revenue",
                "genres",
                "spoken_languages",
            ],
            transformation_pipeline=MovieFeaturePreprocessor.get_preprocessor(),
            model=cosine_similarity,
        )
        cosine_model: RecommenderModel = RecommenderModel(cosine_config)
        linear_kernel_config = RecommenderModelConfig(
            model_name="linear-kernel-movies",
            feature_store=SQLiteConn(r"data/feature_store.sqlite"),
            training_feature_group="movies",
            similarity_matrix_group="linear_kernel_similarity_movies",
            required_features=[
                "original_title",
                "original_language",
                "popularity",
                "vote_average",
                "vote_count",
                "is_popular",
                "runtime",
                "budget",
                "revenue",
                "genres",
                "spoken_languages",
            ],
            transformation_pipeline=MovieFeaturePreprocessor.get_preprocessor(),
            model=linear_kernel,  # Replace with actual model type
        )
        linear_kernel_model: RecommenderModel = RecommenderModel(linear_kernel_config)
        (
            MovieTrainPipeline()
            .add_training_step(cosine_model)
            .add_training_step(linear_kernel_model)
            .save_model_outputs()
        )


if __name__ == "__main__":
    main()
