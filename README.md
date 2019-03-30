# Face_Recognition_Tensorflow
This project shows face recognition by Tensorflow. The purpose of this project is to demonstrate how to use Tensorflow to traing a module and use that module to recognize people.

We first train a CNN module with several layers and then save this CNN module to disk. When real time data stream comes from Network camera. We first use DLIB to detect 68 points face and do face alignment. Then we feed this aligned face to CNN module to do detection.

Development Environment:
1. Windows 10 with NVIDI Quardro M2200
2. Anaconda3-5.0.0-Windows-x86_64 with Python 3.7
3. Tensorflow 1.4.0 version

Training Steps:
1. Open Spyder under Anaconda3
2. Open face_training.py and click "run"
3. If you want to add more faces, please go to "att_faces" folder and create one new folder for your new faces
