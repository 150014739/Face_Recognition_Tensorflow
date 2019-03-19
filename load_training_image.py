# -*- coding: utf-8 -*-
"""
Created March 1, 2019
@author: Erik Morales;
@Description: Load all data for face recognition application;
"""

import os
import glob
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np
import re
import cv2

class LoadTrainingImage:
    def __init__(self, path, classes):
        self.path = path;
        self.classes = classes;
        self.image_list = [] 
        self.label_list = []  
        self.image_data = []
        self.label_data = []
        self.default_image_width = 28 #Default value;
        self.default_image_height = 28 #Default value;

    def print_path(self):
        print(self.path);
        print(self.image_list);
        print(self.label_list);
        print(self.image_data)
        print(self.label_data)
        print(self.default_image_width)
        print(self.default_image_height)
        print("Info: Printing finished.")
    
    def next_batch(self, batch_size):  
        batch_data = []; 
        batch_target = []; 
        
        if (batch_size <= 0):
            print("Error: batch size must be bigger than 0.")
            return batch_data, batch_target
        
        train_data = self.image_data;
        train_target = self.label_data;
        index = [ i for i in range(0,len(train_target)) ]  
        np.random.shuffle(index);  
        for i in range(0,batch_size):  
            batch_data.append(train_data[index[i]]);  
            batch_target.append(train_target[index[i]])  
            
#        temp = np.array(batch_data)
#        print(temp)
        return batch_data, batch_target  
    
    def show_example(self, index):
        image_list_len = len(self.image_list)
        if (index >= len(self.image_list)):
            print("Error: Current index is %d but image_list size is %d."%(index, image_list_len));
            return
        
        #Show one of the image as the example;
        image_example = index;        
        print("Info: Show one image(index %d) as an example."%image_example)        
        test_image = self.image_list[image_example];
        img = mpimg.imread(test_image) # 
        img.shape
        plt.imshow(img)
        plt.axis('off')
        plt.show()
     
    def one_hot(self, classes, index):
        if (classes<=0 or index<0):
            print("Error: %d or %d should be all large than 0"%(classes, index))
        if (classes <= index):
            print("Error: classes %d should be larger than index %d"%(classes, index))
        
        data = [0 for i in range(classes)]        
        data[index] = 1  
        ret_data = np.array(data)
        return ret_data;
    
    def load(self, display):
        classes = self.classes;
        path = self.path;
            
        #Step 1 - Load all image names to image_list;
        #        Load all image labels to label_list;
        filepaths=[os.path.join(path,"s%d"%i)for i in range (1,classes+1)]#
        for one_folder in filepaths:
            print("Info: Start to load files in %s folder."%one_folder);
            for filename in glob.glob(one_folder+'/*.jpg'):
                self.image_list.append(os.path.join(filename))#
                temp_str = one_folder.split('\\');
                length = len(temp_str);
                self.label_list.append(temp_str[length-1])#
        print("Info: Load %d images from %d folders successfully."%(len(self.image_list), len(filepaths)));
        
        #Step 2 - Read image and store pixel data into image_data;
        #         Load image labels to label_data;
        for index in range(len(self.image_list)):    
            
            #Save data to iamge_data and label_data; These data is for tensorflow training;
            image_name = self.image_list[index];


            img_color = cv2.imread(image_name)
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)            
            img_size = img_gray.shape
                        
            img_gray_1d = np.array(img_gray).reshape(img_size[0]*img_size[1]);
            self.image_data.append(img_gray_1d)

            #Convert folder name from string to int, the face name only contains digital number from folder name;
            #For example, folder name is string S5, then face name for that folder is int 5;
            folder_name = self.label_list[index]
            face_name =  re.findall("\d+", folder_name)[0]
            
            d0 = self.one_hot(classes, (int)(face_name) - 1) #The folder start from S1, so we should substract 1;            
            self.label_data.append(d0)         
            
            #Check images size is same or not; Comparing with first image;
            if (index == 0):
                #Update default value based on first image's size;
                self.default_image_width = img_size[0]
                self.default_image_height = img_size[1]
            else:
                image_width = img_size[0]
                image_height = img_size[1]
                if (image_width!=self.default_image_width or image_height!=self.default_image_height):
                    print("Error: Image %s, current size is (%d,%d), desired size is (%d,%d)"%
                          (image_name, image_width, image_height, self.default_image_width, self.default_image_height));
                    break;
                    
            #Display images according to user input;
            if (display == 1):
                plt.imshow(img_gray)
                plt.axis('on')
                plt.show()            
                print("Info: Show converted gray image %s."%image_name)

        self.show_example(2)
        
if __name__=='__main__': 
    path = ".\\att_faces"
    classes = 40
    display = 0
    face_time = LoadTrainingImage(path, classes)
    face_time.load(display)
#    face_time.print_path()
    batch = face_time.next_batch(10)

    face_time.one_hot(classes, 5)