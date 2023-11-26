
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt 
from prettytable import PrettyTable
import pandas as pd 


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