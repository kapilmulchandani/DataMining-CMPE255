import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as pl

grades = [47, 63, 71, 39, 47, 49, 43, 37, 81, 69, 38, 13, 29, 61, 49, 53, 57, 23, 58, 17, 73, 33, 29]
print(np.mean(grades))

fit = stats.norm.pdf(grades, np.mean(grades), np.std(grades))  #this is a fitting indeed
grade_diff = np.std(grades)/3
print(grade_diff)

# mean - std dev = B
pl.plot(grades,fit,'-o')

# pl.hist(grades,normed=True)      #use this to draw histogram of your data



def check_mapping(p):
    mapping = [(np.mean(grades)-(5*np.std(grades)/3), "F"), (np.mean(grades)-(4*np.std(grades)/3), "D"), (np.mean(grades)-(3*np.std(grades)/3), "C-"), (np.mean(grades)-(2*np.std(grades)/3), "C"), (np.mean(grades)- (np.std(grades)/3), "C+"), (np.mean(grades)+ (np.std(grades)/3), "B+"),(np.mean(grades) + (2*np.std(grades)/3), "B+"), (np.mean(grades) + (3*np.std(grades)/3), "A-"), (np.mean(grades) + (4*np.std(grades)/3), "A"), (100 , "A+")] # Add all your values and returns here
    print(np.mean(grades)-(5*np.std(grades)/3))
    for check, value in mapping:
        if p <= check:
            return value

print( check_mapping(90) )
pl.show()