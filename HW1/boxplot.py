import matplotlib.pyplot as plt

plt.interactive(False)
datapoints = [197,199,234,267,269,276,281,289,299,301,339]
fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.boxplot(datapoints, notch=False)
plt.show()
