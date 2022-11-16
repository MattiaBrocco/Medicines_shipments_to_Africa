import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def lasso_report(model, column_names, multiclass = False):
    
    if multiclass == False:
        print(pd.DataFrame(np.c_[column_names, model.coef_],
                           columns = ["Feature", "Coefficient"]).sort_values("Coefficient",
                                                                             key = lambda v: abs(v),
                                                                             ascending = False,
                                                                             ignore_index = True).head(10))

        plt.figure(figsize = (10, 5))
        plt.errorbar(np.log(model.alphas_), model.mse_path_.mean(axis = 1),
                     yerr = model.mse_path_.std(axis = 1), fmt = "o", color = "red")
        plt.axvline(np.log(model.alpha_), ls = "--", color = "grey")
        plt.title("CV loss")
        plt.xlabel("$log(\lambda)$")
        plt.show()
    else:
        # Analysis of Logistic Regression
        
        first_class = model.classes_[0]
        print(pd.DataFrame(np.c_[column_names, model.coef_.T],
                           columns = ["Feature"] +\
                           [f"Coeff. {c}"
                            for c in model.classes_]).sort_values(f"Coeff. {first_class}",
                                                                  key = lambda v: abs(v),
                                                                  ascending = False,
                                                                  ignore_index = True).head(10))

        plt.figure(figsize = (10, 5))
        plt.errorbar(np.log(model.Cs_), model.scores_[first_class].mean(axis = 0),
                     yerr = model.scores_[first_class].std(axis = 0), fmt = "o", color = "red")
        plt.axvline(np.log(model.C_.mean()), ls = "--", color = "grey")
        plt.title("CV loss")
        plt.xlabel("$log(\lambda)$")
        plt.show()