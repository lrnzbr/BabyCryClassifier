
# coding: utf-8

# # Baby Cry Analysis & Testing

# ## Step 1:  Grab the audio file and its label

# In[2]:

#Store all audio files in dictionary where key: filename, value: label
import os
raw_audio = dict()


directory = 'Full_hunger'
for filename in os.listdir(directory):
    if filename.endswith(".wav"): 
        raw_audio[os.path.join(directory, filename)] = 'hungry'
    else:
        continue

directory = 'Full_pain'
for filename in os.listdir(directory):
    if filename.endswith(".wav"): 
        raw_audio[os.path.join(directory, filename)] = 'pain'
    else:
        continue
        
directory = 'Full_asphyxia'
for filename in os.listdir(directory):
    if filename.endswith(".wav"): 
        raw_audio[os.path.join(directory, filename)] = 'asphyxia'
    else:
        continue


#print raw_audio


# #If you are having issues with python being able to make the different directories, try doing it manually or with the os.system commands
# 
# ```os.system("sudo mkdir /audio")```
# 

# ## Step 2:  Chop the audio file into 1 sec. snippets and save them in separate folders

# In[4]:

import wave 
import math

def chop_song(filename, folder):
    handle = wave.open(filename, 'rb')
    frame_rate = handle.getframerate()
    n_frames = handle.getnframes()
    window_size = 2 * frame_rate
    num_secs = int(math.ceil(n_frames/frame_rate))


    #Slicing Audio file
    for i in xrange(num_secs):
        filename = 'audio/' + folder + '/snippet'+ str(i+1) + '.wav'
        snippet = wave.open(filename ,'wb')
        snippet.setnchannels(2)
        snippet.setsampwidth(handle.getsampwidth())
        snippet.setframerate(frame_rate)
        snippet.writeframes(handle.readframes(window_size))
        handle.setpos(handle.tell() - 1 * frame_rate)
        snippet.close()

    handle.close()

for audio_file in raw_audio:
    chop_song(audio_file, raw_audio[audio_file])


# ## Step 3:  Transform .wav to frequency spectrum
# Some files had some sample rate issues and caused errors in this step. If this happens, print out the file name at each pass and if one of the files is causing an error, go ahead and delete it for now

# In[158]:

import pandas as pd
import librosa 
import numpy as np
'''Chop and Transform each track'''
X = pd.DataFrame(columns = np.arange(45), dtype = 'float32').astype(np.float32)
j = 0
k = 0
for i, filename in enumerate(os.listdir('audio/pain/')):
    if filename.endswith(".wav"):
        audiofile, sr = librosa.load("audio/pain/" + filename)
        fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=1)
        x = pd.DataFrame(fingerprint, dtype = 'float32')
        x[44] = 'pain'
        X.loc[i] = x.loc[0]
        j = i 
        

for i, filename in enumerate(os.listdir('audio/hungry/')):
    if filename.endswith(".wav"):
        audiofile, sr = librosa.load("audio/hungry/" + filename)
        fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=1)
        x = pd.DataFrame(fingerprint, dtype = 'float32')
        x[44] = 'hungry'
        X.loc[i+j] = x.loc[0] 
        k = i 
        
for i, filename in enumerate(os.listdir('audio/asphyxia/')):
    if filename.endswith(".wav"):
        #print filename
        audiofile, sr = librosa.load("audio/asphyxia/" + filename)
        fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=1)
        x = pd.DataFrame(fingerprint, dtype = 'float32')
        x[44] = 'asphyxia'
        X.loc[i+j+k] = x.loc[0]
        
#Do something with missing values
X = X.fillna(0)


# ## Step 4:  Make a Test-Train-Split of the data

# In[160]:

from sklearn.cross_validation import train_test_split


y = X[44]
del X[44]
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)


# ## Step 5:  Fit the training data to a model & Check the models performance against the test data¶

# In[164]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, precision_score, recall_score


def get_scores(classifier, X_train, X_test, y_train, y_test, **kwargs):
        model = classifier(**kwargs)
        model.fit(X_train, y_train)
        y_predict = model.predict(X_test)
        return model.score(X_test, y_test),                precision_score(y_test, y_predict),                recall_score(y_test, y_predict)

print "    Model, Accuracy, Precision, Recall"
print "    Random Forest:", get_scores(RandomForestClassifier, X_train, X_test, y_train, y_test, n_estimators=25, max_features=5)
print "    Logistic Regression:", get_scores(LogisticRegression, X_train, X_test, y_train, y_test)
print "    Decision Tree:", get_scores(DecisionTreeClassifier, X_train, X_test, y_train, y_test)
print "    SVM:", get_scores(SVC, X_train, X_test, y_train, y_test)
#print "    Naive Bayes:", get_scores(MultinomialNB, X_train, X_test, y_train, y_test)


# ## Results Model: (Accuracy, Precision, Recall)
#     Random Forest: (0.967741935483871, 0.97235023041474655, 0.967741935483871)
#     Logistic Regression: (0.83870967741935487, 0.83136200716845876, 0.83870967741935487)
#     Decision Tree: (1.0, 1.0, 1.0)
#     SVM: (0.5161290322580645, 0.26638917793964617, 0.5161290322580645)

# ## After you are satisfied with the results of your model, you can save the model into a .pkl file that you can quickly use to make predictions of new data

# In[ ]:

def pickle_model(model, modelname):
    with open('../models/' + str(modelname) + '.pkl', 'wb') as f:
        return cPickle.dump(model, f)
    


# ## After you pkl a model you can open it up later on as so.

# In[ ]:

def getModel(pickle_path):
    with open(pickle_path, 'rb') as f:
        return cPickle.load(f)


# # Making Actual Predictions on new sounds

# You can decide how you want to set up your system to receive and process the audio files it receives. I chose to use concurrent queues in order to make multiple predictions and take the most common result of multiple 1-second slices of audio input.  You may choose to do something else, but as of right now, your algorithm makes a prediction based upon just one second of sound.  Here is some code that I used to make predictions after the audio was already chopped, MFCC transformed into an audio "fingerprint" , and placed in a queue.

# In[ ]:

def predict(fingerprint_queue,prediction_queue, model):
    while True:
        if not fingerprint_queue.empty():
            print "Predictor Worker waking up...\n"
            fingerprint = fingerprint_queue.get()

            X = fingerprint[0].reshape(1, -1)
            prediction = model.predict(X)
            print "PREDICTION: ", prediction
            prediction_queue.put([prediction, fingerprint[1]])

        else:
            #print "Predictor worker waiting....\n"
            sleep(.2)


# I hope this is useful!  Let me know if you have any other questions or concerns and I will be more than happy to help!
