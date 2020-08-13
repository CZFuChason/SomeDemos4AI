from tkinter import Toplevel, Label
import matplotlib.pyplot as plt #plot import
import matplotlib.colors  #color import
import numpy as np  #importing numpy
from PIL import Image #importing PIL to read all kind of images
from PIL import ImageTk
import os


def displaying_faces_grid(displaying_faces):
    size=100, 100
    fig1, axes_array = plt.subplots(5, 5)
    fig1.set_size_inches(5,5)
    count=0
    for x in range(5):
        for y in range (5):
            draw_image = displaying_faces[count]
            draw_image.thumbnail(size)
            draw_image= np.asarray(draw_image,dtype=float)/255.0
            image_plot=axes_array[x][y].imshow(draw_image,cmap=plt.cm.gray)
            axes_array[x][y].axis('off')
            count=count+1
    fig1.canvas.set_window_title('Display all faces')
    plt.show()
    
def display_mean_face(face_array):
    mean = np.mean(face_array, 0)
    fig2, axes_array = plt.subplots(1, 1)
    fig2.set_size_inches(5, 5)
    image_plot = axes_array.imshow(mean, cmap=plt.cm.gray)
    fig2.canvas.set_window_title('mean faces')
    plt.show()
    return mean

def performing_pca(face_array):
    print("mean face")
    mean = display_mean_face(face_array)
    flatten_Array = []
    for x in range(len(face_array)):
        flat_Array = face_array[x].flatten()
        flatten_Array.append(flat_Array)
    flatten_Array = np.asarray(flatten_Array)
    mean = mean.flatten()
    return mean,flatten_Array

def display_all(images):
    fig3, axes_array = plt.subplots(5, 5)
    fig3.set_size_inches(5, 5)
    count = 0
    for x in range(5):
        for y in range(5):
            draw_image = images[count]
            image_plot = axes_array[x][y].imshow(draw_image, cmap=plt.cm.gray)
            axes_array[x][y].axis('off')
            count = count + 1
    fig3.canvas.set_window_title('eigen faces')
    plt.show()
    
def reading_faces_and_displaying():
    face_array = []
    displaying_faces = []
    for face_images in os.listdir('./faces/Train/'):
        face_images = './faces/Train/'+face_images
        face_image=Image.open(face_images)
        displaying_faces.append(face_image)
        face_image = np.asarray(face_image,dtype=float)/255.0
        face_array.append(face_image)
    print("original faces")
    displaying_faces_grid(displaying_faces)
    face_array=np.asarray(face_array)
    return face_array

def display_reconstruction(images, k):
    fig4, axes_array = plt.subplots(5, 5)
    fig4.set_size_inches(5, 5)
    count = 0
    for x in range(5):
        for y in range(5):
            draw_image = np.reshape(images[count,:],(425,425))
            image_plot = axes_array[x][y].imshow(draw_image, cmap=plt.cm.gray)
            axes_array[x][y].axis('off')
            count = count + 1
    fig4.canvas.set_window_title('Reconstructed faces for k='+str(k))
    plt.show()
    
def reconstructing_faces(k,mean,substract_mean_from_original,V):
    weights=np.dot(substract_mean_from_original, V.T)
    reconstruction = mean + np.dot(weights[:,0:k], V[0:k,:])
    display_reconstruction(reconstruction, k)
    
def class_face(k,test_from_mean,test_flat_images,V,substract_mean_from_original,face_array):
    eigen_weights = np.dot(V[:k, :],substract_mean_from_original.T)
    threshold = 6000
    for i in range(test_from_mean.shape[0]):
        test_weight = np.dot(V[:k, :],test_from_mean[i:i + 1,:].T)
        distances_euclidian = np.sum((eigen_weights - test_weight) ** 2, axis=0)
        image_closest = np.argmin(np.sqrt(distances_euclidian))
        fig, axes_array = plt.subplots(1, 2)
        fig.set_size_inches(5, 5)
        to_plot=np.reshape(test_flat_images[i,:], (425,425))
        axes_array[0].imshow(to_plot, cmap=plt.cm.gray)
        axes_array[0].axis('off')
        if (distances_euclidian[image_closest] <= threshold):
            axes_array[1].imshow(face_array[image_closest,:,:], cmap=plt.cm.gray)
        axes_array[1].axis('off')
    plt.show()

def returning_vector(test_images):
    flat_test_Array = []
    for x in range(len(test_images)):
        flat_Array = test_images[x].flatten()
        flat_test_Array.append(flat_Array)
    flat_test_Array = np.asarray(flat_test_Array)
    return flat_test_Array

def reading_test_images():
    test_images=[]
    for images in os.listdir('./faces/Test/'):
        if not images.endswith('.jpg'):
            continue
        images = './faces/Test/'+images
        test_faces = Image.open(images)
        test_faces = np.asarray(test_faces, dtype=float) / 255.0
        test=(425,425,3)
        if test_faces.shape == test:
            test_faces=test_faces[:,:,0]
            test_images.append(test_faces)
        else:
            test_images.append(test_faces)
    print('test samples',len(test_images))
    flat_test_Array=returning_vector(test_images)
    test_images=np.asarray(test_images)
    return flat_test_Array,test_images

def error_for_k(k,test_from_mean,V,substract_mean_from_original,train_list,test_list):
    count=0
    eigen_weights = np.dot(V[:k, :],substract_mean_from_original.T)
    threshold = 6000
    for x in range(test_from_mean.shape[0]):
        test_weight = np.dot(V[:k, :],test_from_mean[x:x + 1,:].T)
        distances_euclidian = np.sum((eigen_weights - test_weight) ** 2, axis=0)
        image_closest = np.argmin(np.sqrt(distances_euclidian))
        x=test_list[x]
        z=int(x[1:])
        if (distances_euclidian[image_closest] <= threshold):
            y=train_list[image_closest]
        else:
            y=0000

        if (x == y) or (z < 89 and y == 0000):
            count = count
        else:
            count = count + 1

    error_rate=count/len(test_list)*100
    return error_rate,count