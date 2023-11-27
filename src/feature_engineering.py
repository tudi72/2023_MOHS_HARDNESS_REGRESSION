import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import numpy as np

seed = np.random.seed(578)

def feature_engineering_pipeline(X,to_merge=None,ignore_outlier_features=None,outlier_col=None,
                                std_features=None,std_method='zscore',kmeans_features=None,k_clusters=5,
                                new_column=None,to_merge_NN=None,n_neighbors=None,
                                encode_method='label',encode_columns=None):
    try:
        pipeline_steps = []

        if to_merge is not None:
            print(to_merge.shape)
            pipeline_steps.append(('merge /WO duplicates', merge_without_duplicates(to_merge=to_merge)))

        if outlier_col is not None:
            pipeline_steps.append(('outliers',add_outliers_col(ignore_outlier_features,outlier_col)))

        if std_features is not None:
            pipeline_steps.append(('standardize',standardize_features(std_features,std_method)))


        if kmeans_features is not None:
            pipeline_steps.append(('kmeans feature',create_kmeans_features(kmeans_features,k_clusters)))


        if to_merge_NN is not None:
            pipeline_steps.append(('merge with NN values', merge_with_NN(new_column,to_merge_NN,n_neighbors)))

        if encode_columns is not None:
            pipeline_steps.append(('column encoder',column_encoder(encode_method,encode_columns)))

        if len(pipeline_steps) == 0:
            return X
        else:
            pip = Pipeline(steps=pipeline_steps)
            X = pip.fit_transform(X)

        return X 
    except Exception as e:
        print(f"[ERROR.feature_engineering.feature_engineering_pipeline]:\t\t ",e)
        return None

class merge_without_duplicates(BaseEstimator,TransformerMixin):
    def __init__(self,to_merge):
        self.to_merge = to_merge
    
    def fit(self, X):
        return self 
    
    def transform(self, X):
        try:
            X = pd.concat([X,self.to_merge])
            X.reset_index(drop=True,inplace=True)
            
            X_duplicates = X[X.duplicated()]
            if not X_duplicates.empty:
                
                # entirely duplicated rows 
                counts = len(X_duplicates)
                X = X.drop_duplicates()
                X.reset_index(drop=True,inplace=True)

                print(f"[INFO.feature_engineering.merge_without_duplicates]:\t\t Found {counts} duplicates")
            
            
            
            else:
                print(f"[INFO.feature_engineering.merge_without_duplicates]:\t\t {X.shape} ...")

            return X

        except Exception as e: 
            print(f"[ERROR.feature_engineering.merge_without_duplicates]:\t\t ",e)

class merge_with_NN(BaseEstimator,TransformerMixin):
    def __init__(self,new_column=None,to_merge_NN=None,n_neighbors=1):
        self.new_column = new_column
        self.to_merge_NN= to_merge_NN
        self.n_neighbors= n_neighbors
    
    def fit(self, X):
        return self 

    def transform(self, X):
        try:
            if self.new_column not in self.to_merge_NN.columns:
                raise Exception(f" column {self.new_column} not existent in dataset")
            
            ignore_features = ['id','Hardness']
            shared_features = [col for col in X.columns if ((col in self.to_merge_NN.columns) and (col not in ignore_features))]

            x_train = X[shared_features]
            to_merge_train = self.to_merge_NN[shared_features]

            model = NearestNeighbors(n_neighbors=1)
            model.fit(to_merge_train)
            _, idx_df =  model.kneighbors(x_train)

            X[self.new_column] = self.to_merge_NN.iloc[idx_df.flatten()][self.new_column].values
            
            print(f"[INFO.feature_engineering.merge_with_NN]:\t\t Created {self.new_column} column with NearestNeighbor")
            return X

        except Exception as e:
            print(f"[ERROR.feature_engineering.merge_with_column]:\t\t ",e)
            return None 
        
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
            print(f"[INFO.feature_engineering.add_outliers_col]:\t\t Found {total_outliers} outliers")
            X[self.outlier_col] = outliers_counted_X
            
            return X

        except Exception as e:
            print(f"[ERROR.feature_engineering.add_outliers_col]:\t\t ",e)
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
            print(f"[ERROR.feature_engineering.standardize_features]:\t\t ",e)

class create_kmeans_features(BaseEstimator,TransformerMixin):

    def __init__(self,features,k_clusters):
        self.k_clusters = k_clusters
    
    def fit(self,X):
        return self 
    
    def transform(self, X):
        try:
            
            ignore_features = ['id','Hardness']
            features = [f for f in X.columns if f not in ignore_features]

            X_subset = X[features]
            kmeans = KMeans(n_clusters=self.k_clusters, random_state=seed).fit(X_subset)
            X['cluster'] = kmeans.labels_

            for i, center in enumerate(kmeans.cluster_centers_):
                X[f'dist_to_center_{i}'] = ((X_subset - center) ** 2).sum(axis=1) ** 0.5
            
            print(f"[INFO.feature_engineering.create_kmeans_features]:\t\t Created {self.k_clusters} cluster cols")    
            return X
        
        except Exception as e:
            print(f"[ERROR.feature_engineering.create_kmeans_features]:\t\t ",e)
            return None

class column_encoder(BaseEstimator,TransformerMixin):
    def __init__(self,method='label',columns=None):
        self.method = method 
        self.columns = columns

        if method == 'label':
            self.encoder = LabelEncoder()
        else:
            self.encoder = None
    
    def fit(self, X):
        return self 
    
    def transform(self, X):
        try:
            
            for column in self.columns:
                if column not in X.columns:
                    raise Exception(f" column {column} not existent")

                X[column] = self.encoder.fit_transform(X[column])
            print(f"[INFO.feature_engineering.categoric_encoder]:\t\t {X.shape} encoded")
            return X

        except Exception as e:
            print(f"[ERROR.feature_engineering.categoric_encoder]:\t\t ",e)
            return None    






