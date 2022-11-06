# Medicines shipments to Africa
Comparison of several supervised learning techniques for regression,
multi-class classification and binary classification from a Kaggle dataset.
(https://www.kaggle.com/divyeshardeshana/supply-chain-shipment-pricing-data)

I have always been passionate about understanding the dynamics of supply chains, especially
those that include intermodal transportation and global scales. In Kaggle I found this
dataset on shipments to different countries in Africa, which gave me the opportunity to try
and exploit different techniques especially with regard to the cleaning and arrangement
of data for the prediction of different aspects separately (supervised learning).

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
