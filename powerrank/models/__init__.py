from powerrank.models.base import RatingModel
from powerrank.models.elo import EloModel
from powerrank.models.ridge import RidgeMarginModel

MODELS: dict[str, type[RatingModel]] = {
    EloModel.name: EloModel,
    RidgeMarginModel.name: RidgeMarginModel,
}

__all__ = ["MODELS", "EloModel", "RatingModel", "RidgeMarginModel"]
