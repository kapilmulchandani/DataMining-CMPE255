import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

marksList1 = [47, 63, 71, 39, 47, 49, 43, 37, 81, 69, 38, 13, 29, 61, 49, 53, 57, 23, 58, 17, 73, 33, 29]
marksList2 = [20, 49, 85, 17, 33, 62, 93, 64, 37, 81, 22, 18, 45, 42, 14, 39, 67, 47, 53, 73, 58, 84, 21]
marks1array = np.array(marksList1)
marks2array = np.array(marksList2)
# marks = [marksList1, marksList2]
# df = pd.DataFrame(marksList1)
# dfnew = pd.DataFrame({'mean': df.mean(), 'median': df.median(),
#                    '25%': df.quantile(0.25), '50%': df.quantile(0.5),
#                    '75%': df.quantile(0.75)})
#
# compareBoxes = plt.boxplot(marks, showmeans=True)
# plt.boxplot(marksList2)
# plt.setp(compareBoxes['boxes'][0], color='green')
# plt.setp(compareBoxes['caps'][0], color='green')
# plt.setp(compareBoxes['whiskers'][0], color='green')
#
# plt.setp(compareBoxes['boxes'][1], color='orange')
# plt.setp(compareBoxes['caps'][1], color='orange')
# plt.setp(compareBoxes['whiskers'][1], color='orange')
#
# plt.ylim([20, 95])
# plt.grid(True, axis='y')
# plt.title('Marks of students in different sections', fontsize=18)
# plt.ylabel('Marks of Students')
# plt.xticks([1,2], ['Section A','Section B'])


sm.qqplot(marks1array, line='s')
sm.qqplot(marks2array, line='s')
# sm.qqplot_2samples(marks1array, marks2array)
plt.show()
