import numpy as np
import numpy.random as random
import pandas as pd
import matplotlib.pyplot as plt


marksList1 = [47, 63, 71, 39, 47, 49, 43, 37, 81, 69, 38, 13, 29, 61, 49, 53, 57, 23, 58, 17, 73, 33, 29]
marksList2 = [20, 49, 85, 17, 33, 62, 93, 64, 37, 81, 22, 18, 45, 42, 14, 39, 67, 47, 53, 73, 58, 84, 21]
marksList1.sort()
marksList2.sort()
df = pd.DataFrame(marksList1)
norm1=random.normal(0,2,len(marksList1))
print(norm1)
norm1.sort()

norm2=random.normal(0,2,len(marksList2))
print("Norm2",norm2)
norm2.sort()

sectionA = plt.figure(figsize=(6,4),facecolor='1.0')

plt.plot(norm1,marksList1,"o")


z1 = np.polyfit(norm1, marksList1, 1)

p1 = np.poly1d(z1)

plt.plot(norm1, p1(norm1),"k--", linewidth=2)

plt.title("Normal Q-Q plot for Section A", size=15)
plt.xlabel("Theoretical quantiles", size=10)
plt.ylabel("Expreimental quantiles", size=10)
plt.tick_params(labelsize=10)
# plt.show()

sectionB = plt.figure(figsize=(6,4),facecolor='1.0')
plt.plot(norm2,marksList2,"o", color='red')
z2 = np.polyfit(norm2, marksList2, 1)
p2 = np.poly1d(z2)
plt.plot(norm2, p2(norm2), "k--", linewidth=2)

plt.title("Normal Q-Q plot for Section B", size=15)
plt.xlabel("Theoretical quantiles", size=10)
plt.ylabel("Expreimental quantiles", size=10)
plt.tick_params(labelsize=10)
plt.show()