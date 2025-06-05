from argparse import ArgumentParser, Namespace
from dataclasses import dataclass


@dataclass
class PipelineArgs:
    pipeline_type: str
    load_type: str
    pages: int
    api_token: str | None


class ArgParser:
    @staticmethod
    def get() -> PipelineArgs:
        parser: ArgumentParser = ArgumentParser(description="ML Movie Recommender Pipeline")

        parser.add_argument(
            "--pipeline",
            type=str,
            choices=["feature"],
            required=True,
            help="Pipeline to execute (feature)",
        )

        parser.add_argument(
            "--type",
            type=str,
            choices=["initial", "incremental"],
            required=False,
            help="Load type (initial/incremental) - Required for feature pipeline",
        )

        parser.add_argument(
            "--pages",
            type=int,
            default=400,
            help="Number of pages to fetch (default: 400)",
        )

        parser.add_argument(
            "--api-token",
            type=str,
            required=False,
            help="TMDb API token - Required for feature pipeline",
        )

        args: Namespace = parser.parse_args()
        if args.pipeline == "feature":
            if not args.type:
                parser.error("--type is required when pipeline is 'feature'")
            if not args.api_token:
                parser.error("--api-token is required when pipeline is 'feature'")

        return PipelineArgs(
            pipeline_type=args.pipeline,
            load_type=args.type,
            pages=args.pages,
            api_token=args.api_token,
        )
