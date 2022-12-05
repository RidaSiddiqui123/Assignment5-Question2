#-------------------------------------------------------------------------
# AUTHOR: Rida Siddiqui
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #5
# TIME SPENT: 2 hrs
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

# Complete  the  Python  program  (clustering.py)  that  will  read  the  file  training_data.csv  to
# cluster  the  data.  Your  goal  is  to  run  k-means  multiple  times  and  check  which  k  value  maximizes  the
# Silhouette  coefficient.  You  also  need  to  plot  the  values  of  k  and  their  corresponding  Silhouette
# coefficients so that we can visualize and confirm the best k value found.

df = pd.read_csv('training_data.csv', sep=',', header=None) #reading the data by using Pandas library

#assign your training data to X_training feature matrix
X_training = np.array(df.values)

#run kmeans testing different k values from 2 until 20 clusters
     #Use:  kmeans = KMeans(n_clusters=k, random_state=0)
     #      kmeans.fit(X_training)
     #--> add your Python code

k = 2
max_sil_score = -1
best_k_list = []
k_list = []
sil_score_list = []
while k < 21:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_training)

    #for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
    sil_score = silhouette_score(X_training, kmeans.labels_)
    #find which k maximizes the silhouette_coefficient
    #--> add your Python code here
    k_list.append(k)
    sil_score_list.append(sil_score)
    if sil_score > max_sil_score:
        best_k_list.append(k)
        max_sil_score = sil_score

    best_k = best_k_list[-1]
    # print("current k: " + str(k))
    # print("current sil score: " + str(sil_score))
    print("Highest silhouette score so far: " + str(max_sil_score))
    print("Best k: " + str(best_k))
    k+=1


#plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
#--> add your Python code here

plt.plot(k_list, sil_score_list)
plt.xlabel('k')
plt.ylabel('silhouette score')
plt.show()


#reading the test data (clusters) by using Pandas library
#--> add your Python code here

df = pd.read_csv('testing_data.csv', sep=',', header=None)

#assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
# do this: np.array(df.values).reshape(1,<number of samples>)[0]
#--> add your Python code here

labels = np.array(df.values).reshape(1,3823)[0]
#--> add your Python code here
kmeans = KMeans(n_clusters=best_k, random_state=0)
kmeans.fit(X_training)

#Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
