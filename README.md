# Face_Recognition_Tensorflow
This project shows face recognition by Tensorflow. The purpose of this project is to demonstrate how to use Tensorflow to traing a module and use that module to recognize people.

We first train a CNN module with several layers and then save this CNN module to disk. When real time data stream comes from Network camera. We first use DLIB to detect 68 points face and do face alignment. Then we feed this aligned face to CNN module to do detection.

Currently, this Python program could train a module and detect face from real time camera stream.
