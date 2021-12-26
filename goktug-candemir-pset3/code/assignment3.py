# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 17:36:38 2021

@author: gcand
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.preprocessing import normalize
import pickle

# Grayscale
def BGR2GRAY(img):
	# Grayscale
	gray = 0.2126 * img[..., 2] + 0.7152 * img[..., 1] + 0.0722 * img[..., 0]
	return gray


def getImages(category,path):
    images=[]
    allImages= os.listdir(path)
    i=0
    print(path)
    for img2 in allImages:
            ##img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # convert to array
        #img2 = cv2.imread(os.path.join(path,img2),0).astype(np.uint8) #gri okumak için
        img2 = cv2.imread(os.path.join(path,img2)).astype(np.uint8)
        res = cv2.resize(img2, dsize=(30, 30))
        res=BGR2GRAY(res).astype(np.float32)
        res = normalize(res, axis=0, norm='max')
        images.append([res,category])
        i=i+1
        """
        if(i==1000):
            imgplot = plt.imshow(res)
            plt.show()
            return images
        """
    return images


def readTestData():
    test_path = 'seg_dev/seg_dev'
    
    training_images_len = []
    for category in os.listdir(test_path):
        num_images = len(os.listdir(os.path.join(test_path, category)))
        training_images_len.append(num_images)
        print(f'Total {category} images:', num_images)
    
    
    images_to_show = []
    for category in os.listdir(test_path):
        images_to_show.append([os.path.join(test_path, category, fname) 
                               for fname in os.listdir(os.path.join(test_path, category))[0:5]])
    
    
    builds_path = os.path.join(test_path,"buildings")
    forest_path = os.path.join(test_path,"forest")
    glacier_path = os.path.join(test_path,"glacier")
    mountain_path = os.path.join(test_path,"mountain")
    sea_path = os.path.join(test_path,"sea")
    street_path = os.path.join(test_path,"street")
    
    buildings_images=getImages("buildings",builds_path)
    forest_images=getImages("forest",forest_path)
    glacier_images=getImages("glacier",glacier_path)
    mountain_images=getImages("mountain",mountain_path)
    sea_images=getImages("sea",sea_path)
    street_images=getImages("street",street_path)
    
    feature_set = np.vstack([buildings_images, forest_images, glacier_images,mountain_images,sea_images,street_images])
    
    return splitData(feature_set)


def splitData(feature_set):
    images = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
    enumerate_images = list(enumerate(images,1))
    
    for i in range(5):
        np.random.shuffle(feature_set)
    
    target=[]
    for i in range(len(feature_set)):
        feature_set[i][0]=np.reshape(feature_set[i][0], (900))
        category=feature_set[i][1]
        liste=[]
        for j in range(6):
            if(enumerate_images[j][1] == category):
                liste.append(1)
            else:
                liste.append(0)
        target.append(liste)  
                
    return feature_set , target    
    
def prepareDataSet(choice='seg_train/seg_train'):
    train_path = 'seg_train/seg_train'
    test_path = 'seg_test/'
    
    
    training_images_len = []
    for category in os.listdir(train_path):
        num_images = len(os.listdir(os.path.join(train_path, category)))
        training_images_len.append(num_images)
        print(f'Total {category} images:', num_images)
    
    
    images_to_show = []
    for category in os.listdir(train_path):
        images_to_show.append([os.path.join(train_path, category, fname) 
                               for fname in os.listdir(os.path.join(train_path, category))[0:5]])
    
    
    builds_path = os.path.join(train_path,"buildings")
    forest_path = os.path.join(train_path,"forest")
    glacier_path = os.path.join(train_path,"glacier")
    mountain_path = os.path.join(train_path,"mountain")
    sea_path = os.path.join(train_path,"sea")
    street_path = os.path.join(train_path,"street")
    
    buildings_images=getImages("buildings",builds_path)
    forest_images=getImages("forest",forest_path)
    glacier_images=getImages("glacier",glacier_path)
    mountain_images=getImages("mountain",mountain_path)
    sea_images=getImages("sea",sea_path)
    street_images=getImages("street",street_path)
    
    feature_set = np.vstack([buildings_images, forest_images, glacier_images,mountain_images,sea_images,street_images])

    return splitData(feature_set)


def get_accuracy(y,y_pred):
    total = 0
    nTruePredict=0
    for i,j in enumerate(y):
        max_y = max(j)
        max_y_pred= max(y_pred[i])
        
        max_index_y = j.index(max_y)
        max_index_y_pred = y_pred[i].index(max_y_pred)
        if(max_index_y==max_index_y_pred):
            nTruePredict += 1
        total+=1

    accuracy=100 * nTruePredict / total
    return accuracy 

def plot_parameters(weight, size1, size2):
    plt.figure()
    plt.imshow(weight.reshape(size1,size2)) 
    plt.show() 


class ArtificialNeuralNetwork(object):
    
    def __init__(self, num_inputs=900, hidden_layers=[10], num_outputs=6):

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        
        self.bias = np.random.rand(1, num_outputs) - 0.5
        # layers show number of neuron each layer
        layers = [num_inputs] + hidden_layers + [num_outputs]
        self.layers=layers
        # initialize random weights for the layers
        
        weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])- 0.5
            weights.append(w)
        self.weights = weights
        
        
        # initialize derivatives per layer with 0
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives
        
        # initialize activations per layer with 0
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations
        
    
    def activation_function(self,x,choice):
        if(choice=="sigmoid"):
            result = 1.0 / (1.0 + np.exp(-x))
            #print(result)
            return result
        elif(choice=="relu"):
            return np.maximum(0,x)
        elif(choice=="softmax"):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()
        elif(choice=="tanh"):
            t=np.tanh(x)
            #t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
            return t
        
    def derivativeOfActivationsFunction(self,x,choice):
        if(choice=="sigmoid"):
            return x * (1.0 - x)
        elif(choice=="relu"):
            return np.greater(x, 0).astype(int)
        elif(choice=="softmax"):
            s=x
            jacobian_m = np.diag(s)
            for i in range(len(jacobian_m)):
                for j in range(len(jacobian_m)):
                    if i == j:
                        jacobian_m[i][j] = s[i] * (1-s[i])
                    else: 
                        jacobian_m[i][j] = -s[i]*s[j]
                        
            s=jacobian_m.reshape(-1,1)
            return np.diagflat(s) - np.dot(s, s.T)
        elif(choice=="tanh"):
            t=np.tanh(x)
            #t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
            dt=1-t**2
            return dt
        
            
    #this function returns output layers output. We will implement softmax for that   
    def forwardPass(self,inputs):
        #The first layer is input layers so activations equal to inputs
        self.activations[0]=inputs
        counter=0
        for w in (self.weights):
            o_net =np.dot(inputs,w)
            #if we are in the output layer , use softmax
            if(counter==len(self.weights)-1):
                o_out=self.activation_function(o_net,"softmax")
                
            else:
                o_out= self.activation_function(o_net,"sigmoid")
            self.activations[counter+1]=o_out
            inputs=o_out
            counter+=1
        
        return inputs
    
    def backPropagate(self,error):
        #dE / dW_i =  (y- a_[i+1]) s'(h_[i+1])) a[i]
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            # we can change activation function and related derivative here.
            if(i==len(self.derivatives)-1):
                derivativeOfActivationsFunction= self.derivativeOfActivationsFunction(activations,"sigmoid")
            else:
                derivativeOfActivationsFunction= self.derivativeOfActivationsFunction(activations,"sigmoid")
            delta = derivativeOfActivationsFunction * error
            #make one row matrix
            delta_reshaped = delta.reshape(delta.shape[0],-1).T
            # convert 1d vector to 2d array
            behind_activation = self.activations[i]
            behind_activation_reshaped= behind_activation.reshape(behind_activation.shape[0],-1)

            self.derivatives[i] = np.dot(behind_activation_reshaped,delta_reshaped)
            error = np.dot(delta,self.weights[i].T)
            
        return error
    
    def train(self,inputs,y_actual,epochs,learning_rate,batch_size):
        list_error=[]
        for i in range(epochs):
            sum_errors=0
            counter=1
            err=np.array([0,0,0,0,0,0])
            for input,target in zip(inputs,y_actual):
                output=self.forwardPass(input)
                #print(output)
                err = err + target - output                
                if(counter==batch_size):
                    error = err / batch_size                    
                    self.backPropagate(error)
                    self.gradientDescent(learning_rate)
                    sum_errors += self.meanSquareError(target, output)
                    err=[0,0,0,0,0,0]
                    counter=1
                else:
                    counter+=1
                    sum_errors += self.meanSquareError(target, output)
                    continue

            list_error.append(sum_errors/len(inputs))
            
            #to get accuracy value each epoch
            """
            y_pred=[]
            for t in range(f1.shape[0]):
                y_pred.append((annMultilayer.forwardPass(f1[t])).tolist())
            
            accuracy.append(get_accuracy(t1, y_pred))
            """
            print("Error: {} at epoch {}".format(sum_errors/len(inputs), i+1))
            #print("Error: {} at epoch {}".format(sum_errors/len(inputs), i+1))
        #print("Training complete!")
        print("=================== Training completed! =================== ")
        return list_error
    
    
    def gradientDescent(self,learning_rate):
        steps=len(self.weights)
        for i in range(steps):
            weights=self.weights[i]
            #print("Before:", weights,"\n")
            #print(self.weights[i])
            #print("--------")
            derivatives = self.derivatives[i]
            weights += (learning_rate * derivatives)
            self.weights[i]=weights
            #print("after:", weights,"\n")
            #print(self.weights[i])

    # p : prediction
    # y : expected output
    def meanSquareError(self,p,y):
        return np.average((p-y)**2)   

    def delta_cross_entropy(self,y,y_pred):
        result = y-y_pred
        return result    
    
    def neg_log_likelihood(self,target,outputs):
        #model output
        scores = outputs
        nll = np.sum(target*scores - np.log(1 + np.exp(scores)))
        return nll * -1  
       
    def cross_entropy(self,out,y):       
        #E = – ∑ ci . log(pi) + (1 – ci ). log(1 – pi)
        log_likelihood =-((y*np.log10(out) ) + ((1-y) * np.log10(1-out)))
        return log_likelihood.sum()

       

    def delta_cross_entropy(self,y_pred, y):
        result = y-y_pred
        return result    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('ann', type=str ,help='Singlelayer or Multilayer')
    parser.add_argument('choice',type=str,help='Predict from test images or validation images')
    parser.add_argument('Test_images_path',type=str,help='Path of test data')
    
    args = parser.parse_args()
    
    #read images
    #feature_set , target = prepareDataSet()
    
    #To change test data you can change the pat below
    
    
    #test_images=getImages("none","seg_test/")
    if(args.choice=="test"):
        print("=======Test images are being read.=======")
        test_images=getImages("none",args.Test_images_path+"/")
        
        for i in range(len(test_images)):
            test_images[i][0]=np.reshape(test_images[i][0], (900))
            
        test_images=np.array(test_images)[:,0]

    """
    with open('test_images','wb') as f:
        pickle.dump(test_images, f)    
    """
    
    """  
    with open('test_images','wb') as f:
        pickle.dump(test_images, f)
      
    with open('target','wb') as f:
        pickle.dump(target, f)    
    """

    
    """
    filename = 'feature_set'
    infile = open(filename,'rb')
    feature_set = pickle.load(infile)
    infile.close()
    """
    
    filename = 'target'
    infile = open(filename,'rb')
    target = pickle.load(infile)
    infile.close()
    
    #if the test images same , we can directly getting the array
    """
    filename = 'test_images'
    infile = open(filename,'rb')
    test_images = pickle.load(infile)
    infile.close()
    """
    #feature_set, target = shuffle(feature_set, target)
    """
    data=feature_set[:,0]
    target=np.array(target)
    """
    #To create and train a model you can use this code :
    
    #annMultilayer= ArtificialNeuralNetwork(900,[20,10],6)
    #annSinlelayer= ArtificialNeuralNetwork(900,[],6)       
    
    filename = 'multilayer'
    infile = open(filename,'rb')
    annMultilayer = pickle.load(infile)
    infile.close()
    
    filename = 'singlelayer'
    infile = open(filename,'rb')
    annSinlelayer = pickle.load(infile)
    infile.close()
    
    #list_error=ann.train(data,target,5,0.03,1)
    prefered_model=annMultilayer
    
    if(args.ann=="singlelayer"):
        prefered_model=annSinlelayer
        
        
    
    """
    with open('multilayer','wb') as f:
        pickle.dump(ann, f)  
    """
    #To get predict from validation data
    if(args.choice=="validation"):
        f1,t1 = readTestData()
        f1=f1[:,0]    
    
    """
    f1,t1 = readTestData()
    f1=f1[:,0]
    """
    
    
    accuracy=[]
    #items = np.array([[random()/2 for _ in range(900)] for _ in range(1)])
    y_pred=[]
    
    if(args.ann=="singlelayer"):
        print("Single Layer Model [900,6] start to predict the images")
    else:
        print("Multi Layer Model [900,20,10,6] start to predict the images")
    """
    for t in range(test_images.shape[0]):
        y_pred.append((prefered_model.forwardPass(test_images[t])).tolist())  
    """
    if(args.choice =="test"):
        print("Starting to predict from test images")
        for t in range(test_images.shape[0]):
            y_pred.append((prefered_model.forwardPass(test_images[t])).tolist()) 
    else:
        print("Starting to predict from validation images")
        for t in range(f1.shape[0]):
            y_pred.append((prefered_model.forwardPass(f1[t])).tolist()) 
        print("Accuracy: ",get_accuracy(t1,y_pred))
    print("Done !")
    #print(get_accuracy(t1,y_pred))
    
    #To save trained model
    """
    with open('multilayer','wb') as f:
        pickle.dump(ann, f)   
        
    with open('singlelayer','wb') as f:
        pickle.dump(ann2, f)  
    
    """