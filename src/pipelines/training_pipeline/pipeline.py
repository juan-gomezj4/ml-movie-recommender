from typing import ClassVar

from src.utils.recommender_models import RecommenderModel


class MovieTrainPipeline:
    ERR_NO_STEPS: ClassVar[str] = "No training steps have been added to the pipeline"

    def __init__(self) -> None:
        self.steps: dict[str, RecommenderModel] = {}

    def add_training_step(self, model: RecommenderModel) -> "MovieTrainPipeline":
        self.steps[model.name] = model
        return self

    def save_model_outputs(self) -> "MovieTrainPipeline":
        if not self.steps:
            raise ValueError(self.ERR_NO_STEPS)
        for _, model in self.steps.items():
            model.fit()
            model.store_outputs()
        return self
