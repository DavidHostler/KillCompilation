#Image processing
import cv2 
import numpy as np
#Deep learning 
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D
from keras.models import Sequential
PATH = os.getcwd()
image_folder_path = PATH + '/gameplay_frames/'


def save_frames_to_image_folder(PATH, video_frames_path=image_folder_path):
    folder_found = False
    for folder in os.listdir(PATH):
        if folder == video_frames_path:
            folder_found = True
    if folder_found is True:
        cap = cv2.VideoCapture(PATH + 'valorant1.mp4')
        i= 0
        while(cap.isOpened()):
            ret, frame_ = cap.read()
            if ret == False:
                break
            if i >= num_frames:
                
                frame = cv2.resize(frame_, (320, 180))#This is an ok res
                cv2.imwrite(video_frames_path + 'frame'+ str(i)+'.jpg',frame)
            print('Created frame no. ' + str(i) + '...')
            i+=1
        #     else:
        #         break

        cap.release()
        # Closes all the windows currently opened.
        cv2.destroyAllWindows()
    else:
        print("No such folder was found. Please create it :)")
        return 

def generate_data(arr, dataset_size, threshold=19):
    X_, y_ = [], []
    for i in range(dataset_size):
        
        skull = add_noise(arr, threshold)
        no_skull = add_noise(arr, threshold, False)
        #Containing the skull 
        X_.append(skull)
        y_.append(1)
        #Without the skull 
        X_.append(no_skull)
        y_.append(0)
        
    return np.array(X_), np.array(y_)


def generate_augmented_data(arr, dataset_size, has_skull):
    
        data = []
        for num in range(dataset_size):
            if num % 100 == 0:
                print("Oh, we're halfway there, oooowoooahh!")
                print("Livin' on a prayer!: ")
                print("Data added: ", num)
                
            video_sample = cv2.imread(video_frames_path + 'frame' + str(num) + '.jpg')
            video_sample = cv2.resize(video_sample, (320, 180))
            no_skull = video_sample[130:150, :][:,150:170] #Get the region where the skull would appear
            no_skull = cv2.resize(no_skull, (40,40))
            clone = arr
            with_skull = no_skull
            if has_skull is True:
                for i in range(len(clone)):
                    for j in range(len(clone[i])):

                        if clone[i][j].any() > 0.0:
                            with_skull[i][j] = clone[i][j]


                data.append([with_skull, 1])
                   
            else:
                data.append([no_skull, 0])
        
        return data
                
        
# X, y = generate_data(arr, 10000)        

# X, y = generate_augmented_data(arr, 1000)



images_with_skull = generate_augmented_data(arr, 5000, True)
images_without_skull = generate_augmented_data(arr, 5000, False)

data = images_with_skull + images_without_skull

# images_without_skull = generate_augmented_data(arr, 100, False)

def preprocess(data):
    
    X_, y_ = [], []
    for i in range(len(data)):
        
        X_.append(data[i][0])
        y_.append(data[i][1])
    
    return np.array(X_), np.array(y_)
        
for _ in range(5):
    random.shuffle(data)

training_data = data[:int(0.75 * len(data))]  
test_data = data[int(0.75 * len(data)):]

X_train, y_train = preprocess(training_data)
X_test, y_test = preprocess(test_data)




'''Train the deep learning model with Tensorflow'''

model.add(Conv2D(filters=1, kernel_size=(3,3), input_shape=X_train.shape[1:]))

model.add(Dropout(0.1))
model.add(MaxPool2D(2,2))
#Downsize the data to 20 * 20 convolutional layer
model.add(Conv2D(filters=1, kernel_size=(3,3),input_shape=(20,20,3), activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPool2D(2,2))

model.add(Conv2D(filters=1, kernel_size=(3,3),input_shape=(20,20,3), activation='relu'))
model.add(Dropout(0.1))
model.add(MaxPool2D(2,2))

model.add(Flatten())



model.add(Dense(400, activation='relu' ))

model.add(Dense(1, activation='sigmoid'))


from keras.optimizers import Adam

opt_ = Adam(learning_rate = 0.0001)

EPOCHS = 10

model.compile(optimizer='adam', loss='binary_crossentropy' ,  metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=EPOCHS  , validation_data=(X_test, y_test))

