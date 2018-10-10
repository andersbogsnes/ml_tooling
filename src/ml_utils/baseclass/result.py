from functools import total_ordering
import numpy as np

@total_ordering
class Result:
    """
    Data class for holding results of model testing.
    Also implements comparison operators for finding max mean score
    """

    def __init__(self,
                 model,
                 model_name,
                 viz=None,
                 model_params=None,
                 cross_val_scores=None,
                 cross_val_mean=None,
                 cross_val_std=None,
                 metric=None
                 ):
        self.model = model
        self.model_name = model_name
        self.cross_val_scores = cross_val_scores
        self.cross_val_mean = cross_val_mean
        self.cross_val_std = cross_val_std
        self.model_params = model_params
        self.metric = metric
        self.plot = viz

    def __eq__(self, other):
        return self.cross_val_mean == other.cross_val_mean

    def __lt__(self, other):
        return self.cross_val_mean < other.cross_val_mean

    def __repr__(self):
        return f"<Result {self.model_name}: " \
               f"Cross-validated {self.metric}: {np.round(self.cross_val_mean, 2)} " \
               f"± {np.round(self.cross_val_std, 2)}>"
