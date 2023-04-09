import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from scipy.optimize import curve_fit



def revenue_optimization(data, sku, plot = False):
    
    price_range = np.linspace(data[(data["PO / SO #"] == sku) &
                                   (data["Unit Price"] > 0)]["Unit Price"].min(),
                              data[(data["PO / SO #"] == sku) &
                                   (data["Unit Price"] > 0)]["Unit Price"].max()*2, 50)
    
    transactions = [len(data[(data["PO / SO #"] == sku) &
                             (data["Unit Price"] >= p)])
                    for p in price_range]
    
    if len(data[(data["PO / SO #"] == sku) &
                (data["Unit Price"] > 0)]) >= 3:
    
        try:
            logit = lambda x, a, b, c: c*(np.exp(-a-b*x))/(1+np.exp(-a-b*x))

            popt, pcov = curve_fit(logit,
                                   price_range, transactions,
                                   method = "trf",
                                   bounds = ([-np.inf, 1e-10, 1e-10], np.inf),
                                   max_nfev = 3000)

            f_prime = [-derivative(logit, x, dx = 1e-6, args = popt) for x in price_range]

            optimal_price = price_range[np.argmax(f_prime)]

            if plot == True:
                plt.subplots(1, 1)
                plt.plot(price_range, transactions, marker = "o", color = "grey", label = "data")
                plt.plot(price_range, logit(price_range, *popt),
                         linestyle = "--",
                         color = "black", label = "NLLS")
                plt.plot(price_range, f_prime, label = "derivative",
                         color = "black", alpha = .7, linewidth = 1)    
                plt.axvline(price_range[np.argmax(f_prime)], label = "New Price")
                plt.xlabel("Unit Price")
                plt.ylabel("Num of transactions")
                plt.ylim(-0.1, max(transactions)*2)
                plt.title(sku)
                plt.legend()
                plt.show()
            
            ### Check against historical prices
            hist_ref = np.quantile(data[(data["PO / SO #"] == sku) &
                                        (data["Unit Price"] > 0)]["Unit Price"], .75)
            
            optimal_price = max(optimal_price, hist_ref)
            ###
            
            
            return f_prime, price_range, optimal_price, popt

        except RuntimeError:
            print(f"*** RuntimeError for {sku}")
            optimal_price = np.average(price_range, weights = transactions)
        
            return price_range, price_range, optimal_price, [0, 0, 0]
    else:
        print("Less than 3")
        optimal_price = np.mean(data[(data["PO / SO #"] == sku) &
                                     (data["Unit Price"] > 0)]["Unit Price"])
        return price_range, price_range, optimal_price, [0, 0, 0]
    
    