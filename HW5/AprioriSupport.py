import numpy as np
import pandas as pd
from apyori import apriori

store_data = pd.read_csv('transactions.csv', header=None)
records=[]
for i in range(0, 6):
    records.append([str(store_data.values[i,j]) for j in range(0,6)])
association_rules = apriori(records, min_support=0.50, min_confidence=0.7, min_lift=1.2)
association_rules = list(association_rules)

print(association_rules)