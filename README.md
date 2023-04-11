# Medicines shipments to Africa
Comparison of several supervised learning techniques for regression, and multi-class classification from a [Kaggle dataset](https://www.kaggle.com/divyeshardeshana/supply-chain-shipment-pricing-data)

Classification entails the prediction of the *Shipment Mode*, while regression regards *Price* prediction.

Best performance on classification is $0.91$ accuracy, while for regression a $29.8/%$ MAPE.

Moreover, coming from the insights found while writing my Bachelor Thesis on Price Optimization following the [Pricing and Revenue Optimization](https://www.sup.org/books/title/?id=31628) (R. Philips, 2006), I have tried to implement the same logics for this problem.

Using non-detrimental pricing strategy, where the price resulting from the optimization procedure (`PRO_approach.py`) is the maximum value between the "optimal" price and the 75th percentile of past prices, I have obtained a revenue increase of $18/%$.

An example is shown below, where we see the data points, their interpolation via non-linear least squares with a prior of a logit function, and the derivative of the latter function (a bell-shaped curve). Finally, the new price point is the value that maximizes that derivative.

![image](https://user-images.githubusercontent.com/61026948/231292473-c83493dc-3f03-4340-b1d6-99812e6e16f2.png)


The choice of the topic made me understand, above all to gain
"domain knowledge" useful to understand how to handle the data at hand,
how medicines are classified (different types of experimentation before use - PQ, PO, ...),
and transportation in terms of insurance and liability of the transported cargo.
From the following analysis I could also see how the insurance cost can be calculated
starting from the data at hand with a good degree of approximation,
and also how the choice of means of transportation can be predicted with precision.

Finally, from these data I could see how many medicines destined to African Countries
are produced in India (almost 65% of the instances), and even more important a greater
awareness of how malaria and HIV by 2015 were still diseases that scourge the continent.


_*RMK*_: the two `.txt` files have been created manually from maps.google.com coordinates,
because this seemed the most logical way to exploit the information of the columns
of Country of destination and Manufacturing Site (after an API caller failed to work).
