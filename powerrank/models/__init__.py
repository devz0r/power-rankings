from powerrank.models.base import RatingModel
from powerrank.models.elo import EloModel
from powerrank.models.epa import EpaModel
from powerrank.models.ridge import RidgeMarginModel

MODELS: dict[str, type[RatingModel]] = {
    EloModel.name: EloModel,
    RidgeMarginModel.name: RidgeMarginModel,
    EpaModel.name: EpaModel,
}

__all__ = ["MODELS", "EloModel", "EpaModel", "RatingModel", "RidgeMarginModel"]
