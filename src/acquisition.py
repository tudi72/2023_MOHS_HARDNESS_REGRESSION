
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


def acquisition_pipeline(path=None,drop_cols=None,fill_NA_col=None):
    try:
        pipeline_steps = []
        X = pd.DataFrame()

        if path is not None:
            pipeline_steps.append(('read',read_csv_errorless(path)))

        if drop_cols is not None:
            pipeline_steps.append(('drop', drop_redundant_cols(drop_cols=drop_cols)))

        if fill_NA_col is not None:
            pipeline_steps.append(('fill NA', add_fill_NA_column(fill_NA_col=fill_NA_col)))

        pip = Pipeline(steps=pipeline_steps)

        return pip.fit_transform(X)
    
    except Exception as e:
        print(f"[ERROR.acquisition.acquisition_pipeline]: ",e)

class read_csv_errorless(BaseEstimator,TransformerMixin):
    def __init__(self, path = None):
        self.path = path 
    def fit(self, X):
        return self 
    def transform(self, X):
        try:
            X = pd.read_csv(self.path)
            print(f"[INFO.acquisition.read_csv_errorless]: Dataframe {X.shape} ...")

            return X
        except Exception as e:
            print(f"[ERROR.acquisition.read_csv_errorless]: ",e)
            return None

class add_fill_NA_column(BaseEstimator, TransformerMixin):
   
    def __init__(self, fill_NA_col):
        self.fill_NA_col = fill_NA_col
    
    def fit(self,X):
        return self 
    
    def transform(self, X):
        try:
            if self.fill_NA_col in X.columns:
                X[self.fill_NA_col] = X[self.fill_NA_col].fillna('not known')
                print(f"[INFO.acquisition.add_fill_NA_col]: Dataframe {X.shape} ...")

            else: 
                raise Exception(f"Column {self.fill_NA_col} not in dataframe")
            
            return X
        except Exception as e:
            print(f"[ERROR.acquisition.add_fill_NA_col]: ",e) 
            return None   
  
class drop_redundant_cols(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols):
        self.drop_cols = drop_cols

    def fit(self, X,):
        return self
    
    def transform(self, X):
        try:
            if self.drop_cols in X.columns:
                X = X.drop(self.drop_cols, axis=1)
                print(f"[INFO.acquisition.drop_redundant_cols]: Dataframe {X.shape} ...")

            else: 
                raise Exception(f"column {self.drop_cols} not in dataframe")
            return X
        except Exception as e:
            print(f"[ERROR.acquisition.drop_redundant_cols]: ",e)
            return None
   
