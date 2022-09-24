#Image processing
import cv2 
import numpy as np
#Deep learning 
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D
from keras.models import Sequential
PATH = os.getcwd()
# VIDEO_FOLDERS_PATH = '/media/dolan/Backup Plus/Valorant'
VIDEO_FOLDERS_PATH = PATH


model = keras.models.load_model(PATH + '/kill_cam.h5')


#Writes all the kill frames into one folder 
def write_compilation_frames(video_frames_path, compilation_frames_path, starting_index, threshold):
    directory = os.listdir(video_frames_path)
    n = len(directory)
    frame_counter = 0
    
    j = 0
    i = starting_index
    
#     for i in range(starting_index, len(directory)):
    while i < len(directory):
        frame =  cv2.imread(video_frames_path + 'frame' + str(i) + '.jpg')
        # region_of_interest = cv2.resize(sample_img, (320, 180))
        region_of_interest = frame[520:590, :][:,600:685]#[130:150, :][:,150:170]
        region_of_interest = cv2.resize(region_of_interest, (40, 40))
        # input_ = cv2.resize(region_of_interest, )
        input_ = np.expand_dims(region_of_interest, axis=0)
        prediction = model.predict(input_)
#         print(prediction[0][0])
        if i % 100 == 0:
            print('Still lookin for dem killz...')
        if prediction[0][0] > threshold:
            print('Ladies and gentlemen, we gottem...')
            for j in range(150):
                kill_cam_frame = cv2.imread(video_frames_path + 'frame' + str(i + j) + '.jpg')
                cv2.imwrite(compilation_frames_path + 'kill' + str(frame_counter) + '.jpg', kill_cam_frame)
                frame_counter += 1
#                 i += 1#Update i pointer to skip ahead 
            i += 150 #Update i upon exiting the for loop so you don't go thru this data twice
        i += 1 #Iterate through while loop data
            
          
            
    print('Finished')




compilation_frames_path = VIDEO_FOLDERS_PATH +'/compilation_images/'
video_frames_path = VIDEO_FOLDERS_PATH + '/frames/'
#We know that there is a kill at frame3850.jpg, so let's start a few frames before that...
#write_compilation_frames(video_frames_path, compilation_frames_path, 38300, 0.95)
write_compilation_frames(video_frames_path, compilation_frames_path, 0, 0.95)
#Floating point threshold indicates our preference for how confident we want the model
#to be before considering it's prediction valid.
#The aim of this is to avoid Type I errors (i.e. falsely rejecting the null hypothesis
#that the image is in fact not likely a kill, when it is!) aka false positives 
#That's one way to do this guy