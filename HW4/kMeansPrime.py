from scipy.spatial import distance

firstTenPrimes = [[2,3], [5,7], [11,13], [17,19], [23,29], [31,37], [41,43], [47, 53], [59,61], [67, 71]]

# Just change the centroid for the second part [[2,3], [67,71]]
givenCentroid1 = [2,3]
givenCentroid2 = [5,7]



cluster1=[]
cluster2=[]

distancesFrom1 = []
for i in range(0, len(firstTenPrimes)):
    distancesFrom1.append(distance.euclidean(firstTenPrimes[i], givenCentroid1))

distancesFrom2 = []
for i in range(0, len(firstTenPrimes)):
    distancesFrom2.append(distance.euclidean(firstTenPrimes[i], givenCentroid2))

for i in range(0, len(firstTenPrimes)):
    if distancesFrom1[i] < distancesFrom2[i]:
        cluster1.append(firstTenPrimes[i])
    else:
        cluster2.append(firstTenPrimes[i])

count = 8
while count != 0:
    newCentroids1 = [0, 0]
    newCentroids2 = [0, 0]
    for i in range(0, len(cluster1)):
        newCentroids1[0] = newCentroids1[0] + cluster1[i][0]
        newCentroids1[1] = newCentroids1[1] + cluster1[i][1]

    newCentroids1[0] = newCentroids1[0]/len(cluster1)
    newCentroids1[1] = newCentroids1[1]/len(cluster1)

    for i in range(0, len(cluster2)):
        newCentroids2[0] = newCentroids2[0] + cluster2[i][0]
        newCentroids2[1] = newCentroids2[1] + cluster2[i][1]

    newCentroids2[0] = newCentroids2[0]/len(cluster2)
    newCentroids2[1] = newCentroids2[1]/len(cluster2)

    #Now compute Eucleadian distance of points to both the centroids
    distancesFrom1 = []
    for i in range(0, len(firstTenPrimes)):
        distancesFrom1.append(distance.euclidean(firstTenPrimes[i], newCentroids1))

    distancesFrom2 = []
    for i in range(0, len(firstTenPrimes)):
        distancesFrom2.append(distance.euclidean(firstTenPrimes[i], newCentroids2))

    # Now Change the centroids and clusters accordingly
    cluster1=[]
    cluster2=[]
    for i in range(0, len(firstTenPrimes)):
        if distancesFrom1[i] < distancesFrom2[i]:
            cluster1.append(firstTenPrimes[i])
        else:
            cluster2.append(firstTenPrimes[i])
    count = count-1

print(cluster1)
print(cluster2)
