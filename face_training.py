# -*- coding: utf-8 -*-
"""
Created March 18, 2019
@author: Erik Morales;
@Description: Train dataset using tensorflow
"""

import numpy as np
import load_training_image as fi
import tensorflow as tf

#The image size of all_faces is 92*112;
path = ".\\att_faces";
classes = 41
display = 0
face_image = fi.LoadTrainingImage(path, classes)
face_image.load(display)
width = face_image.default_image_width
height = face_image.default_image_height
print("Info: Image size (w:%d, h:%d)"%(width, height))
test_x = face_image.image_data
test_y = face_image.label_data

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
################################################################################################################


loss=tf.losses.softmax_cross_entropy(onehot_labels=output_y,logits=logits)
print(loss)
train_op=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

accuracy_op=tf.metrics.accuracy(
    labels=tf.argmax(output_y,axis=1),
    predictions=tf.argmax(logits,axis=1)
)[1] 

tf.device('/gpu:0')
sess=tf.Session()
init=tf.group(tf.global_variables_initializer(),
              tf.local_variables_initializer())
sess.run(init)

ckpt_file_path = "./models/face"

saver = tf.train.Saver()
for i in range(30000):
    [batch_data, batch_target]= face_image.next_batch(100)
    train_loss,train_op_ = sess.run([loss,train_op],{input_x:batch_data,output_y:batch_target})
    if i%100==0:
        test_accuracy=sess.run(accuracy_op,{input_x:test_x,output_y:test_y})
        print("Step=%d, Train loss=%.4f,[Test accuracy=%.2f]"%(i,train_loss,test_accuracy))

x=55;
y=75;        
test_output = sess.run(logits,{input_x:test_x[x:y]})
inferenced_y = np.argmax(test_output,1)
print(inferenced_y,'Inferenced numbers')#推测的数字
print(np.argmax(test_y[x:y],1),'Real numbers')

saver.save(sess, ckpt_file_path)

sess.close()
