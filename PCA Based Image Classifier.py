#!/usr/bin/env python
# coding: utf-8

# In[1206]:

'''Shreyash Shrivastava
    1001397477'''

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob


def open_imgs_set(path, matrix):
    count = 0
    for filename in glob.glob(path):
        # 3-D array from an image
        img_data_3d = np.asarray(Image.open(filename))
        if img_data_3d.shape == (25, 25, 3):
            # From 3-D array to 1-D array: converting single image into 1-D array of size n (int this case n=154587)
            img_data_1d = img_data_3d.ravel()
            #  Preparing matrix n_photo*154587
            matrix.append(img_data_1d)
            if count == 400:
                break
            count += 1
    return matrix


def get_X_Y_values(filename, label):  # PCA Dimesnsion Reduction for each dataset
    pca_data = []
    x = []

    x = open_imgs_set(filename, x)

    pca_data.append(x)

    pca_data = np.array(pca_data)

    # print(len(pca_data[0][0,:])) # Extracting a row (20 X 20 X 3)
    # print(len(pca_data[0][:,0])) # Extracting a col (number of images)

    row_length = len(pca_data[0][0, :])  # 1200
    col_length = len(pca_data[0][:, 0])  # images = 3

    # Computing the d dimensional mean vector
    mean_vector = []
    for y in range(len(pca_data[0][:, 0])):
        mean_vector.append(np.mean(pca_data[0][y, :]))

    # Computing the scatter matrix
    scatter_matrix = np.zeros((row_length, row_length))
    for i in range(pca_data.shape[1]):
        scatter_matrix += (pca_data[:, i].reshape(row_length, 1) - mean_vector).dot(
            (pca_data[:, i].reshape(row_length, 1) - mean_vector).T)
    # print('Scatter Matrix:\n', scatter_matrix)

    # Computing the eigenvectors and eigenvalues from the scatter matrix
    eig_val_sc, eig_vec_sc = np.linalg.eigh(scatter_matrix)

    # Getting eigen pairs and sorting them
    eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:, i]) for i in range(len(eig_val_sc))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # Choosing K eigen vectors with largest values (3PC's)
    matrix_w = np.hstack((eig_pairs[0][1].reshape(row_length, 1), eig_pairs[1][1].reshape(row_length, 1),
                          eig_pairs[2][1].reshape(row_length, 1)))

    # Transforming the samples onto a new subspace
    # print(matrix_w.shape)
    transformed = matrix_w.T.dot(pca_data[0].transpose())
    data = transformed.transpose()

    dataset = pd.DataFrame({'X': data[:, 0], 'Y': data[:, 1], 'Z': data[:, 2], 'Class': label})
    return dataset


# Getting the datasets for diffrent fruits (dataset)
# dataset_apple = get_X_Y_values('resized_fruits/Apple/*.jpg','A')
# dataset_lemon = get_X_Y_values('resized_fruits/Lemon/*.jpg','L')
# dataset_grape = get_X_Y_values('resized_fruits/Grape/*.jpg','G')
# dataset_pineapple = get_X_Y_values('resized_fruits/Pineapple/*.jpg','P')


# print(pca2.explained_variance_ratio_.cumsum())
# print(x_t_2)

dataset_dog = get_X_Y_values('resized/cane/*.jpg', 'D')
dataset_elephant = get_X_Y_values('resized/elefante/*.jpg', 'E')
dataset_horse = get_X_Y_values('resized/cavallo/*.jpg', 'H')
dataset_butterfly = get_X_Y_values('resized/farfalla/*.jpg', 'B')

# In[1207]:


print(dataset_butterfly)

# In[1208]:


# Principle Component Vizualization
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def plot(data, string):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    x = data['X']
    y = data['Y']
    z = data['Z']

    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title(string)
    plt.show()

    return [x, y, z]


## Fruit Coordinate Data Plot
# apple_coordinates = plot(dataset_apple,'Apple')
# lemon_coordinates = plot(dataset_lemon,'Lemon')
# grape_coordinates = plot(dataset_grape,'Grape')
# pineapple_coordinates = plot(dataset_pineapple,'Pineapple')

## Animal Coordinate Data Plot
dog_coordinates = plot(dataset_dog, 'Dog')
elephant_coordinates = plot(dataset_elephant, 'Elephant')
horse_coordinates = plot(dataset_horse, 'Horse')
butterfly_coordinates = plot(dataset_butterfly, 'Butterfly')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

## Animal Data Combined Plot
ax.scatter(dog_coordinates[0], dog_coordinates[1], dog_coordinates[2], c='b', marker='x')
ax.scatter(elephant_coordinates[0], elephant_coordinates[1], elephant_coordinates[2], c='r', marker='p')
ax.scatter(butterfly_coordinates[0], butterfly_coordinates[1], butterfly_coordinates[2], c='y', marker='x')
ax.scatter(horse_coordinates[0], horse_coordinates[1], horse_coordinates[2], c='g', marker='x')

## Fruit Data Combined Plot
# ax.scatter(apple_coordinates[0], apple_coordinates[1], apple_coordinates[2], c='b', marker='x')
# ax.scatter(lemon_coordinates[0], lemon_coordinates[1], lemon_coordinates[2], c='r', marker='p')
# ax.scatter(grape_coordinates[0], grape_coordinates[1], grape_coordinates[2], c='y', marker='x')
# ax.scatter(pineapple_coordinates[0], pineapple_coordinates[1], pineapple_coordinates[2], c='g', marker='x')


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_title('All Datasets')
plt.show()

# In[1210]:


# Dataset concatnation for fruits
# final_df = dataset_apple.append(dataset_lemon).append(dataset_grape).append(dataset_pineapple)
# Dataset concatnation for animals
final_df = dataset_dog.append(dataset_elephant).append(dataset_horse).append(dataset_butterfly)

# Shuffling the dataset
final_df = final_df.sample(frac=1)

# Index correction
idx_arr = []
idx = 0
for x in range(len(final_df)):
    idx_arr.append(idx)
    idx += 1

final_df = final_df.set_index([idx_arr])

# final_df_train  = final_df.iloc[:(int)(len(final_df)*.6)] # Training Data
final_df_train = final_df.iloc[:]  # Training Data

# Train Indexing correction
idx_arr = []
idx = 0
for x in range(len(final_df_train)):
    idx_arr.append(idx)
    idx += 1

final_df_train = final_df_train.set_index([idx_arr])  # index correction for training data

final_df_test = final_df.iloc[(int)(len(final_df) * .6):]  # Testing Data

# Testing index correction
idx_arr = []
idx = 0
for x in range(len(final_df_test)):
    idx_arr.append(idx)
    idx += 1

final_df_test = final_df_test.set_index([idx_arr])  # index correction for testing data

count = 0
for x in final_df_train['X']:
    if x in final_df_test['X']:
        # print(x)
        count += 1
print(count)
print(len(final_df_test['X']))
print(len(final_df_train['X']))

print(final_df)

# In[1213]:


# Using KNN for clasification
import math


# Cartesian distance calculator function
def euclideanDistance(item1, item2, length):
    cal_dist = 0
    for x in range(length):
        cal_dist += pow((item1[x] - item2[x]), 2)
    return math.sqrt(cal_dist)


# Function to check the nearest points from the given input
def checkNeighbors(trainingData, test, k):
    distance_measure = []
    length = len(test)

    # calling the distance calculator function to measure each distance and add to a list
    for x in range(len(trainingData)):
        dist = euclideanDistance(test, trainingData[x], length)
        distance_measure.append((dist, trainingData[x]))

    sort_arr = []
    for x in range(len(distance_measure)):
        sort_arr.append(distance_measure[x][0])

    sort_arr.sort()

    sort_arr = (sort_arr[::-1][:k])

    distance_measure = sorted(distance_measure, key=lambda x: x[0])

    neighbors = []
    for x in range(k):
        neighbors.append(distance_measure[x][1])

    return neighbors


# Function to see the majority class near the point and predict the class
def determineClass(neighbors):
    classMajority = {}

    # Determining the classes majority nearest to the given data point
    for x in range(len(neighbors)):
        classification = neighbors[x][-1]
        if classification in classMajority:
            classMajority[classification] += 1
        else:
            classMajority[classification] = 1

    lis = []
    for x in classMajority:
        lis.append(x)

    xis = []
    for x in classMajority.values():
        xis.append(x)

    max = -1
    counter = 0
    max_counter = 0
    for y in xis:
        counter += 1
        if (y > max):
            max = y
            max_counter = counter

    num = ''
    for x in range(len(lis)):
        if x == max_counter - 1:
            num = lis[x]
    return num


KNN_arr = []  # Classification array for KNN
for x in final_df_train.values:
    KNN_arr.append(x)

# In[1214]:

## Testing Data (Animals)
#dataset_dog_test = get_X_Y_values('resized_animals_test/cane/*.jpg','D')
#dataset_elephant_test = get_X_Y_values('resized_animals_test/elefante/*.jpg','E')
#dataset_horse_test = get_X_Y_values('resized_animals_test/cavallo/*.jpg','H')
#dataset_butterfly_test = get_X_Y_values('resized_animals_test/farfalla/*.jpg','B')

# Testing Data (fruits)
# dataset_apple_test = get_X_Y_values('resized_fruit_test/Apple/*.jpg','A')
# dataset_lemon_test = get_X_Y_values('resized_fruit_test/Lemon/*.jpg','L')
# dataset_grape_test = get_X_Y_values('resized_fruit_test/Grape/*.jpg','G')
# dataset_pineapple_test = get_X_Y_values('resized_fruit_test/Pineapple/*.jpg','P')

#testing_final_df = dataset_dog_test.append(dataset_elephant_test).append(dataset_horse_test).append(dataset_butterfly_test)
#testing_final_df = dataset_apple_test.append(dataset_lemon_test).append(dataset_grape_test).append(dataset_pineapple_test)
#testing_final_df = testing_final_df.sample(frac=1)

#print(testing_final_df)



# Validation
correct = 0
total = 0
for x in final_df_test.values:

    result = checkNeighbors(KNN_arr, x[0:3], 3)
    result_final = determineClass(result)
    if result_final == x[3]:
        correct += 1
    total += 1

print(correct / total)




