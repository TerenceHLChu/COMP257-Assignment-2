# Student name: Terence Chu
# Student number: 301220117

from sklearn.datasets import fetch_olivetti_faces
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
import numpy as np

olivetti = fetch_olivetti_faces()

print('Olivetti faces data shape', olivetti.data.shape) # 64 × 64
print('Olivetti faces target shape', olivetti.target.shape)

X = olivetti.data
y = olivetti.target

print('\nPixel values:\n', X)
print('Pixel maximum:', X.max())
print('Pixel minimum:', X.min())
print('Data is already normalized')

plt.figure(figsize=(7,7))

for i in range(12):
    plt.subplot(3, 4, i+1) # 3 rows, 4 columns
    plt.imshow(X[i].reshape(64,64), cmap='gray')
    plt.xticks([])
    plt.yticks([])

plt.show()

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=17)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, stratify=y_train_full, test_size=0.25, random_state=17)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('X_valid shape:', X_valid.shape)

# Train an SVC classifier 
svc_classifier = SVC(kernel='linear')

# Define the cross-validation 
stratified_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=17) 

cross_val_sore = cross_val_score(svc_classifier, X_train, y_train, cv=stratified_k_fold, scoring='accuracy')

print('\nCross validation score of the 5 folds:', cross_val_sore)

# Train the classifier
svc_classifier.fit(X_train, y_train)

# Assess the classifer with the validation data
score_valid = svc_classifier.score(X_valid, y_valid)
print('\nModel accuracy on validation set', score_valid)

list_of_silhouette_scores = []

print("\nNumber of Clusters : Silhouette Scores")

for num_of_clusters in range(2, X_train.shape[0], 15): # X_train.shape[0] = 240

    # Carry out KMeans clustering
    kmeans = KMeans(n_clusters=num_of_clusters, random_state=17)
    y_pred = kmeans.fit_predict(X_train)

    sil_score = silhouette_score(X_train, y_pred) # Calculate Silhouette score
    list_of_silhouette_scores.append((num_of_clusters, sil_score)) # Append with tuple of cluster number and Silhouette score
    print(num_of_clusters, ': ', sil_score)

highest_silhouette_score = 0
highest_silhouette_score_number_of_clusters = 0

# Determine highest silhouette score 
for k, v in list_of_silhouette_scores:
    if v > highest_silhouette_score:
        highest_silhouette_score = v
        
# Determine the number of clusters corresponding to the highest silhouette score 
for k, v in list_of_silhouette_scores:
    if v == highest_silhouette_score:
        highest_silhouette_score_number_of_clusters = k

print('\nHighest silhouette score:', highest_silhouette_score)
print("Highest silhouette score's number of clusters:", highest_silhouette_score_number_of_clusters) 

# Reduce dimensionality to 77 dimensions
kmeans_reduced_dims = KMeans(n_clusters=highest_silhouette_score_number_of_clusters, random_state=17)

X_train_reduced_dims = kmeans_reduced_dims.fit_transform(X_train) # Number of dimensions reduced to 77

X_valid_reduced_dims = kmeans_reduced_dims.transform(X_valid) # Number of dimensions reduced to 77

X_test_reduced_dims = kmeans_reduced_dims.transform(X_test) # Number of dimensions reduced to 77

print('\nShape of X_train_reduced_dims:', X_train_reduced_dims.shape) 
print('Shape of X_valid_reduced_dims:', X_valid_reduced_dims.shape)
print('Shape of X_test_reduced_dims:', X_test_reduced_dims.shape)

cross_val_sore_reduced = cross_val_score(svc_classifier, X_train_reduced_dims, y_train, cv=stratified_k_fold, scoring='accuracy')

print('\nCross validation score of the 5 folds:', cross_val_sore_reduced)

svc_classifier.fit(X_train_reduced_dims, y_train)

score_valid = svc_classifier.score(X_valid_reduced_dims, y_valid)
print('\nModel accuracy on validation set', score_valid)

# Try different eps and min_samples value
for epsilon in np.arange(0.5, 6.0, 1.0): 
    for minimum_samples in (3, 5):
        dbscan = DBSCAN(eps=epsilon, min_samples=minimum_samples, metric='euclidean')
        dbscan.fit(X) # No conversion needed. X is already a feature vector
        
        print('eps =', epsilon, 'min_samples =', minimum_samples, 'dimensions = 4096 metric = euclidean')
        print("Labels:", dbscan.labels_)
        print("Indices of the core instances:", dbscan.core_sample_indices_)
        print('------------------------------')

from sklearn.decomposition import PCA

n_components = 50

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

for epsilon in np.arange(0.5, 6.0, 1.0): 
    for minimum_samples in (3, 5):
        dbscan = DBSCAN(eps=epsilon, min_samples=minimum_samples, metric='euclidean')
        dbscan.fit(X_pca)
        
        print('eps =', epsilon, 'min_samples =', minimum_samples, 'dimensions =', n_components, 'metric = euclidean')
        print("Labels:", dbscan.labels_)
        print("Indices of the core instances:", dbscan.core_sample_indices_)
        print(len(dbscan.core_sample_indices_))
        print('------------------------------')

for epsilon in np.arange(0.1, 0.5, 0.05): 
    for minimum_samples in (3, 5):
        dbscan = DBSCAN(eps=epsilon, min_samples=minimum_samples, metric='cosine')
        dbscan.fit(X_pca)
        
        print('eps =', epsilon, 'min_samples =', minimum_samples, 'dimensions =', n_components, 'metric = cosine')
        print("Labels:", dbscan.labels_)
        print("Indices of the core instances:", dbscan.core_sample_indices_)
        print(len(dbscan.core_sample_indices_))
        print('------------------------------')
        
for epsilon in np.arange(0.5, 6.0, 1.0): 
    for minimum_samples in (3, 5):
        dbscan = DBSCAN(eps=epsilon, min_samples=minimum_samples, metric='manhattan')
        dbscan.fit(X_pca)
        
        print('eps =', epsilon, 'min_samples =', minimum_samples, 'dimensions =', n_components, 'metric = manhattan')
        print("Labels:", dbscan.labels_)
        print("Indices of the core instances:", dbscan.core_sample_indices_)
        print(len(dbscan.core_sample_indices_))
        print('------------------------------')