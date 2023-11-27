
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt 
from prettytable import PrettyTable
import numpy as np 
import seaborn as sns 
import pandas as pd 


def view_dataframe_info(X):

    print("DataFrame Information:")
    print("--"* 40)
    print(X.info(verbose=True, show_counts=True))
    print("--"* 40)
    print("DataFrame Values:")
    print("--"* 40)
    print(X.head(5).T)
    print("--"* 40)
    print("DataFrame Description:")
    print("--"* 40)
    print(X.describe().T)
    print("--"* 40)
    print("Number of Null Values:")
    print("--"* 40)
    print(X.isnull().sum())
    print("--"* 40)

    print("Number of Duplicated Rows:")
    print("--"* 40)
    print(X.duplicated().sum())
    print("--"* 40)

    print("Number of Unique Values:")
    print("--"* 40)
    print(X.nunique())
    print("--"* 40)

    print("DataFrame Shape:")
    print("--"* 40)
    print(f"Rows: {X.shape[0]}, Columns: {X.shape[1]}")
    print("--"* 40)


def view_mutual_info_regression(X,label):
    y = X[label]
    X = X.drop(label, axis=1)

    mi_scores = mutual_info_regression(X, y)
    mi_scores = pd.Series(mi_scores, name="Mutual Information", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)

    pretty_table = PrettyTable()
    pretty_table.field_names = ["Feature", "Mutual Information"]

    for feature, mi_score in mi_scores.items():
        pretty_table.add_row([feature, round(mi_score, 3)])

    print(pretty_table)


def view_upper_corr_matrix(X,method='spearman'):
    
    corr = X.corr(method)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(10, 8))

    heatmap = sns.heatmap(corr, mask=mask, cmap='pink_r', vmax=.3, center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    title = heatmap.set_title("Correlation Heatmap", weight='bold', size=16)
    title.set_position([0.45, 1.1])
    plt.show()