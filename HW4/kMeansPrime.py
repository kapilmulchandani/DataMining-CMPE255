from scipy.spatial import distance

firstTenPrimes = [[2,3], [5,7], [11,13], [17,19], [23,29], [31,37], [41,43], [47, 53], [59,61], [67, 71]]

#First we make 2 clusters -->
# 1st : [2,3]
# 2nd : [5,7], [11,13], [17,19], [23,29], [31,37], [41,43], [47, 53], [59,61], [67, 71]
cluster1 = [[2,3]]
cluster2 = [[5,7], [11,13], [17,19], [23,29], [31,37], [41,43], [47, 53], [59,61], [67, 71]]
newCentroids1 = [2/1, 3/1]
newCentroids2 = [0, 0]
print(firstTenPrimes[0])
for i in range(1, len(firstTenPrimes)):
    newCentroids2[0] = newCentroids2[0] + firstTenPrimes[i][0]
    newCentroids2[1] = newCentroids2[1] + firstTenPrimes[i][1]

newCentroids2[0] = newCentroids2[0]/len(firstTenPrimes) - 1
newCentroids2[1] = newCentroids2[1]/len(firstTenPrimes) - 1

print(newCentroids2)
print(newCentroids1)

#Now compute Eucleadian distance of points to both the centroids
distancesFrom1 = []
for i in range(0, len(firstTenPrimes)):
    distancesFrom1.append(distance.euclidean(firstTenPrimes[i], newCentroids1))

distancesFrom2 = []
for i in range(0, len(firstTenPrimes)):
    distancesFrom2.append(distance.euclidean(firstTenPrimes[i], newCentroids2))

print(distancesFrom1)
j = 0
k = 0
cluster1=[]
cluster2=[]
for i in range(0, len(firstTenPrimes)):
    if distancesFrom1[i] < distancesFrom2[i]:
        cluster1.append(firstTenPrimes[i])
    else:
        cluster2.append(firstTenPrimes[i])

print('wow')
# Now Change the centroids and clusters accordingly

