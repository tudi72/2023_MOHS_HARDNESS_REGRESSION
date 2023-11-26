import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np

seed = np.random.seed(578)

def feature_engineering_pipeline(X,to_merge=None,ignore_outlier_features=None,outlier_col=None,std_features=None,std_method='zscore',kmeans_features=None,k_clusters=5):
    try:
        pipeline_steps = []

        if to_merge is not None:
            pipeline_steps.append(('merge /WO duplicates', merge_without_duplicates(to_merge=to_merge)))

        if outlier_col is not None:
            pipeline_steps.append(('outliers',add_outliers_col(ignore_outlier_features,outlier_col)))

        if std_features is not None:
            pipeline_steps.append(('standardize',standardize_features(std_features,std_method)))

        if kmeans_features is not None:
            pipeline_steps.append(('kmeans feature',create_kmeans_features(kmeans_features,k_clusters)))

        pip = Pipeline(steps=pipeline_steps)
        X = pip.fit_transform(X)
        return X 
    except Exception as e:
        print(f"[ERROR.feature_engineering.feature_engineering_pipeline]: ",e)
        return None

class merge_without_duplicates(BaseEstimator,TransformerMixin):
    def __init__(self,to_merge):
        self.to_merge = to_merge
    
    def fit(self, X):
        return self 
    
    def transform(self, X):
        try:
            X = pd.concat([X,self.to_merge]).reset_index(drop=True)

            X_duplicates = X[X.duplicated()]
            if not X_duplicates.empty:
            
                counts = len(X_duplicates)
                X = X.drop_duplicates()
                X = X.reset_index(drop=True)

                print(f"[INFO.feature_engineering.merge_without_duplicates]: Found {counts} duplicates")
            else:
                print(f"[INFO.feature_engineering.merge_without_duplicates]: No duplicates")

            return X

        except Exception as e: 
            print(f"[ERROR.feature_engineering.merge_without_duplicates]: ",e)

class add_outliers_col(BaseEstimator,TransformerMixin):
    
    def __init__(self,ignore_features=None,outlier_col=None):
        self.ignore_features = ignore_features 
        self.outlier_col = outlier_col
    
    def fit(self, X):
        return self 
    
    def transform(self, X):
        try: 
            features = [f for f in X.columns if f not in self.ignore_features]
            X_subset = X[features]

            clf = IsolationForest(contamination='auto')
            outliers_pred = clf.fit_predict(X_subset)

            outliers_counted_X = pd.DataFrame({
                self.outlier_col: [(1 if (pred == -1) else 0) for pred in outliers_pred]
            })

            total_outliers = outliers_counted_X[self.outlier_col].sum()
            print(f"[INFO.feature_engineering.add_outliers_col]: Found {total_outliers} outliers")
            X[self.outlier_col] = outliers_counted_X
            
            return X

        except Exception as e:
            print(f"[ERROR.feature_engineering.add_outliers_col]: ",e)
            return None

class standardize_features(BaseEstimator, TransformerMixin):
    def __init__(self, features, method='zscore'):

        self.features = features

        if method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()

    def fit(self, X):
        return self 
    
    def transform(self, X):
        try:
            X_copy = X.copy()
            X_subset = X_copy[self.features]
            X_subset_standardized = self.scaler.fit_transform(X_subset)
            X[self.features] = X_subset_standardized
            return X

        except Exception as e:
            print(f"[ERROR.feature_engineering.standardize_features]: ",e)

class create_kmeans_features(BaseEstimator,TransformerMixin):

    def __init__(self,features,k_clusters):
        self.k_clusters = k_clusters
    
    def fit(self,X):
        return self 
    
    def transform(self, X):
        try:
            
            ignore_features = ['id','Hardness','is_original']
            features = [f for f in X.columns if f not in ignore_features]

            X_subset = X[features]
            kmeans = KMeans(n_clusters=self.k_clusters, random_state=seed).fit(X_subset)
            X['cluster'] = kmeans.labels_

            for i, center in enumerate(kmeans.cluster_centers_):
                X[f'dist_to_center_{i}'] = ((X_subset - center) ** 2).sum(axis=1) ** 0.5

            print(f"[INFO.feature_engineering.create_kmeans_features]: Created {self.k_clusters} cluster cols")    
            return X
        
        except Exception as e:
            print(f"[ERROR.feature_engineering.create_kmeans_features]: ",e)
            return None








