import numpy as np
import pandas as pd

from ml_tooling.result import Result


class CVResult(Result):
    """
    Data class for holding results of model testing.
    Also implements comparison operators for finding max mean score
    """

    def __init__(self,
                 model,
                 viz=None,
                 cv=None,
                 cross_val_scores=None,
                 metric=None,
                 ):
        self.cv = cv
        self.cross_val_scores = cross_val_scores
        self.cross_val_mean = np.mean(cross_val_scores)
        self.cross_val_std = np.std(cross_val_scores)
        super().__init__(model=model,
                         viz=viz,
                         score=self.cross_val_mean,
                         metric=metric,
                         )

    def to_dataframe(self, params: bool = True, cross_val_score: bool = False) -> pd.DataFrame:
        """
        Output result as a DataFrame for ease of inspecting and manipulating.
        Defaults to including model params, which can be toggled with the params flag.
        This is useful if you're comparing different models. Additionally includes
        the standard deviation of the score and number of cross validations.

        If you want to inspect the cross-validated scores, toggle cross_val_score and the resulting
        DataFrame will have one row per fold.
        :param params:
            Boolean - whether to include model parameters as columns in the DataFrame or not
        :param cross_val_score:
            Boolean - whether to have one row per fold in the DataFrame or not
        :return:
            DataFrame of results
        """
        df = super().to_dataframe(params).assign(cross_val_std=self.cross_val_std, cv=self.cv)
        if cross_val_score:
            return pd.concat([df.assign(score=score)
                              for score in self.cross_val_scores], ignore_index=True)
        return df

    def __repr__(self):
        cross_val_type = f"{self.cv}-fold " if isinstance(self.cv, int) else ''
        return f"<Result {self.model_name}: " \
            f"{cross_val_type}Cross-validated {self.metric}: {np.round(self.score, 2)} " \
            f"± {np.round(self.cross_val_std, 2)}>"
