from src.pipelines.feature_pipeline.pipeline import (
    FeaturePipelineConfig,
    MovieFeaturePipeline,
)
from src.utils.arg_parser import ArgParser, PipelineArgs
from src.utils.sqlite_conn import SQLiteConn


def main() -> None:
    ERR_MISSING_TOKEN: str = "API token must be provided for feature pipeline"  # noqa: S105

    args: PipelineArgs = ArgParser.get()

    if args.pipeline_type == "feature":
        if not args.api_token:
            raise ValueError(ERR_MISSING_TOKEN)

        config = FeaturePipelineConfig(
            api_token=args.api_token,
            feature_group="movies_test",
            type=args.load_type,
            pages=args.pages,
        )
        pipeline = MovieFeaturePipeline(config)
        feature_store = SQLiteConn(r"data/feature_store.sqlite")
        pipeline.run(feature_store)


if __name__ == "__main__":
    main()
