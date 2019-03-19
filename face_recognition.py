# -*- coding: utf-8 -*-
"""
Created March 1, 2019
@author: Erik Morales;
@Description: Run face recognition with tensorflow;
"""
#import parameters
from multiprocessing import Process, Queue
import os
import cv2
import dlib
import numpy as np
import tensorflow as tf

#User cound edit;
MODE = 0 # 1-Read data from Camera, 0-Read data from AVI file
VIDEO_FILE = ".\\avi\\camera_testing_video.mp4"
FACE_LANDMARK_FILE = ".\\dlib\\shape_predictor_68_face_landmarks.dat"
DISPLAY_SCALE = 0.5
FACE_POINT_RADIUS = 2
FACE_POINT_THICKNESS = 1
FACE_RECTANGLE_THICKNESS = 3
FACE_TEXT_THICKNESS = 3
CV_WAITKEY_MS = 10 #million seconds
RESIZED_WIDTH = 92 #Make sure that the width and height are same as training image;
RESIZED_HEIGHT = 112
CLASSES_NUMBER = 41 #Make sure that CLASS_NUMBER is same as traing classes;

#Do not touch
QUIT_THREAD_FLAG = -1000

#To read video from avi or camera; This thread must be fast, otherwise stream from camera will be corrupted;
def proc_video(q_video2algo, q_algo2video):
        
    cap = cv2.VideoCapture(VIDEO_FILE)    
    
    fp = open('.\\logs\\log_video_thread.txt', 'w')
    fp.write('Process(%s) is writing...\n' % os.getpid())

    window_name = 'Face Recognition by Tensorflow'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    scale = DISPLAY_SCALE

    frame_index = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
               
        if ret == False:
            break;

        if cv2.waitKey(CV_WAITKEY_MS) & 0xFF == ord("q"):
            break
        
        #Until here, we have gotten video frame successfully; Send frame to algorithm thread to process;        
        if q_video2algo.empty()==True:
            q_video2algo.put((frame_index, frame))                        
        
        #Assign initial state to labeled frame and index;
        if (frame_index == 0):
            frame_index_labeled = 0
            frame_labeled = frame;

        #Read labeled frame and index infomration from Queue;
        if q_algo2video.empty()==False:
            frame_index_labeled, frame_labeled = q_algo2video.get(True)
            fp.write('frame_index_labeled %d..\n' % frame_index_labeled)
        
        #Concatenate two images together; One is original image, the other is image with faces detected
        plot_frame = np.concatenate((frame, frame_labeled), axis=0)
        size = plot_frame.shape
        resized_w = (int)(size[0]*scale)
        resized_h = (int)(size[1]*scale)
        
        resized_frame = cv2.resize(plot_frame, (resized_w, resized_h), interpolation=cv2.INTER_CUBIC)
        cv2.imshow(window_name, resized_frame)            
        frame_index = frame_index + 1;
        
    cap.release()
    cv2.destroyAllWindows()    
            
    q_video2algo.put((QUIT_THREAD_FLAG, frame))
      
    fp.close()
        
#This thread is to read image from rooc_video thread and rn algorithm;
def proc_algorithm(q_video2algo, q_algo2video):
    fp = open('.\\logs\\log_algorithm_thread.txt', 'w')
#    logging.debug('Process(%s) is reading...' % os.getpid())
    fp.write('Process(%s) is reading...\n' % os.getpid())
    fp.flush()

    predictor_path = FACE_LANDMARK_FILE
    predictor = dlib.shape_predictor(predictor_path)
    detector = dlib.get_frontal_face_detector()
    fp.write("Showing detections and predictions on the images in the faces folder...\n")    

    width = RESIZED_WIDTH
    height = RESIZED_HEIGHT
    classes = CLASSES_NUMBER
    
    tf.reset_default_graph()
    ##################################################################################################################
    #CNN Model below, do not tough it. Make sure  that this mode is the same as training model;
    #The reason why same piece of code in traning and validation is to implement save/load function;
    #There is one other solution to implement save/load function without same piece of code but not implemented here;
    #CNN Model start here:
    ##################################################################################################################
    input_x = tf.placeholder(tf.float32,[None, width*height])/255.
    output_y=tf.placeholder(tf.int32,[None, classes])
    input_x_images = tf.reshape(input_x,[-1, width, height, 1])
    
    conv1=tf.layers.conv2d(
        inputs=input_x_images,
        filters=32,
        kernel_size=[5,5],
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )
    print(conv1)
    
    pool1=tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2,2],
        strides=2
    )
    print(pool1)
    
    conv2=tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        strides=1,
        padding='same',
        activation=tf.nn.relu
    )
    
    pool2=tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2,2],
        strides=2
    )
    
    w0 = int(width/4);
    h0 = int(height/4);
    flat=tf.reshape(pool2,[-1,w0*h0*64])
    
    dense=tf.layers.dense(
        inputs=flat,
        units=1024,
        activation=tf.nn.relu
    )
    print(dense)
    
    dropout=tf.layers.dropout(
        inputs=dense,
        rate=0.5
    )
    print(dropout)
    
    logits=tf.layers.dense(
        inputs=dropout,
        units=classes
    )
    print(logits)
    ##################################################################################################################
    #CNN Model end here:
    ##################################################################################################################    
    
    sess=tf.Session()
    saver=tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('.\\models'))    
        
    face_list = []    
    while True:
        
        if q_video2algo.empty()==True:
            continue
        
        #q_video2algo is not empty        
        frame_index, frame = q_video2algo.get(True)
        
        if (frame_index == QUIT_THREAD_FLAG):
            break;
        #Copy original frame    
        frame_without_label = frame.copy()            

        img2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dets = detector(img2, 1)
        face_number = len(dets);
        fp.write("Info: Frame %d received, %d face(s) detected.\n"%(frame_index, face_number))
    
        #No face detected;
        if (face_number == 0):
            continue
        
        #Some faces are detected                  
        face_list = plot_labels_on_frame(frame, dets, predictor)
        crop_image = []
        for single_face in face_list:  
            fp.write("Info: [top_left_x, top_left_y, bottom_right_x, bottom_right_y] = [%d, %d, %d, %d]\n"%(single_face[0], single_face[1], single_face[2], single_face[3]))
            crop_image = frame_without_label[single_face[1]:single_face[3], single_face[0]:single_face[2]] 
            resized_crop_image = cv2.resize(crop_image, (width, height), interpolation=cv2.INTER_CUBIC)           
            face_saved_file = '.\\logs\\face_frame{}.jpg'.format(frame_index)            
            cv2.imwrite(face_saved_file, resized_crop_image)

            #Align image and use tensorflow to do evaluation;            
            resized_crop_gray = cv2.cvtColor(resized_crop_image, cv2.COLOR_BGR2GRAY)                
            resized_crop_1d = np.array(resized_crop_gray).reshape(width*height);
            test_x = []
            test_x.append(resized_crop_1d)
            test_x.append(resized_crop_1d)
            test_output = sess.run(logits, {input_x:test_x[0:1]})
            inferenced_y = np.argmax(test_output, 1)
            fp.write('Info: Inferenced face %d\n'%inferenced_y[0])#Reconized face index
            cv2.putText(frame, 's%d'%(inferenced_y[0]+1), (single_face[0], single_face[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), FACE_TEXT_THICKNESS)

        #Send back frame with labels to video thread for displaying
        if q_algo2video.empty()==True:
            q_algo2video.put((frame_index, frame))
            
    
    sess.close()
    fp.flush()
    fp.close()

def plot_labels_on_frame(frame, dets, predictor):
    face_list = [];
    [top_left_x, top_left_y, bottom_right_x, bottom_right_y] = [0, 0, 0, 0]        
    for index, face in enumerate(dets):

        shape = predictor(frame, face)

        for index, pt in enumerate(shape.parts()):
            pt_pos = (pt.x, pt.y)
            cv2.circle(frame, pt_pos, FACE_POINT_RADIUS, (0, 255, 0), FACE_POINT_THICKNESS)  
            if index == 0:
                top_left_x = pt.x
            elif index == 16:
                bottom_right_x = pt.x
            elif index == 19:
                top_left_y = pt.y
            elif index == 8:
                bottom_right_y = pt.y        

        updated_top_left_x = top_left_x - (int)((bottom_right_x-top_left_x)*0.1)
        updated_top_left_y = top_left_y - (int)((bottom_right_y-top_left_y)*0.6)
        updated_bottom_right_x = bottom_right_x + (int)((bottom_right_x-top_left_x)*0.1)
        updated_bottom_right_y = bottom_right_y + (int)((bottom_right_y-top_left_y)*0.05)
        single_face_coordination = [updated_top_left_x, updated_top_left_y, updated_bottom_right_x, updated_bottom_right_y]  
        face_list.append(single_face_coordination)        
        cv2.rectangle(frame,(updated_top_left_x,updated_top_left_y),(updated_bottom_right_x, updated_bottom_right_y),(255,255,0), FACE_RECTANGLE_THICKNESS)
    return face_list

if __name__=='__main__':
    #Queue to communicate between two threads
    q_video2algo = Queue()
    q_algo2video = Queue()
    video = Process(target=proc_video, args=(q_video2algo, q_algo2video, ))
    algorithm = Process(target=proc_algorithm, args=(q_video2algo, q_algo2video, ))

    #Start two threads
    print('Info: Start thread to read video successfully.');
    video.start()
    print('Info: Start thread to recognize face successfully.');
    algorithm.start()

    #Wait until video and algorithm thread finishing.
    video.join()
    algorithm.join()

    #Terminate these two threads
    video.terminate()
    algorithm.terminate()
    
    print('Info: Finished!');
    
    