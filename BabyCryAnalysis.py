
# coding: utf-8

# # Baby Cry Analysis & Testing

# ## Part 1: Let's train a machine learning algorithm and test it's performance

# We will begin by collecting all of the sample audio files we have, chopping them into smaller audio snippets and training a collection of machine learning algorithms with part of this data.  With another part of the data we will test and see how well the algorithms predict data they have never seen before and then choose the best algorithm for our project

# ### Step 1:  Grab the audio file and its label (we have 3 labels: hungry, pain, and asphyxia)

# In[1]:

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

# ## Step 2:  Chop the audio file into 1 sec. snippets and save them in corresponding folders

# In[3]:

import wave 
import math

def chop_song(filename, folder):
    handle = wave.open(filename, 'rb')
    frame_rate = handle.getframerate()
    n_frames = handle.getnframes()
    window_size = 2 * frame_rate
    num_secs = int(math.ceil(n_frames/frame_rate))
    #print filename
    last_number_frames = 0
    #Slicing Audio file
    for i in xrange(num_secs):
        
        shortfilename = filename.split("/")[1].split(".")[0]
        snippetfilename = 'audio/' + folder + '/' + shortfilename + 'snippet' + str(i+1) + '.wav'
        #print snippetfilename
        snippet = wave.open(snippetfilename ,'wb')
        snippet.setnchannels(2)
        snippet.setsampwidth(handle.getsampwidth())
        snippet.setframerate(frame_rate)
        #snippet.setsampwidth(2)
        #snippet.setframerate(11025)
        snippet.setnframes(handle.getnframes())
        snippet.writeframes(handle.readframes(window_size))
        handle.setpos(handle.tell() - 1 * frame_rate)
        #print snippetfilename, ":", snippet.getnchannels(), snippet.getframerate(), snippet.getnframes(), snippet.getsampwidth()
        
        #The last audio slice might be less than a second, if this is the case, we don't want to include it because it will not fit into our matrix 
        if last_number_frames < 1:
            last_number_frames = snippet.getnframes()
        elif snippet.getnframes() != last_number_frames:
            #print "this file doesnt have the same frame size!, remaming file"
            os.rename(snippetfilename, snippetfilename+".bak")
        snippet.close()

    #handle.close()

for audio_file in raw_audio:
    chop_song(audio_file, raw_audio[audio_file])


# ## Step 3:  Transform .wav files to frequency spectrum "fingerprints" using MFCC algorithm

# In[5]:

import pandas as pd
import librosa 
import numpy as np
'''Chop and Transform each track'''
X = pd.DataFrame(columns = np.arange(45), dtype = 'float32').astype(np.float32)
j = 0
k = 0
for i, filename in enumerate(os.listdir('audio/pain/')):
    last_number_frames = -1
    if filename.endswith(".wav"):
        #print filename
        audiofile, sr = librosa.load("audio/pain/" + filename)
        fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=1)
        x = pd.DataFrame(fingerprint, dtype = 'float32')
        x[44] = 'pain'
        X.loc[i] = x.loc[0]
        j = i 
        

for i, filename in enumerate(os.listdir('audio/hungry/')):
    if filename.endswith(".wav"):
        #print filename
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
        
#Do something with missing values. you might want to do something more sophisticated with missing values later
X = X.fillna(0)


# In[16]:

X.head()


# ## Step 4:  Make a Test-Train-Split of the data

# In[6]:

from sklearn.cross_validation import train_test_split


y = X[44]
del X[44]
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)


# ## Step 5:  Fit the training data to a model & Check the models performance against the test data¶

# In[7]:

#Code to hide deprication warnings

from IPython.display import HTML
HTML('''<script>
code_show_err=false; 
function code_toggle_err() {
 if (code_show_err){
 $('div.output_stderr').hide();
 } else {
 $('div.output_stderr').show();
 }
 code_show_err = !code_show_err
} 
$( document ).ready(code_toggle_err);
</script>
To toggle on/off output_stderr, click <a href="javascript:code_toggle_err()">here</a>.''')


# In[8]:

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
# We tested four popular machine learning algorithms to see which ones had the most accuracte predictions with our test data, here are the results:
# #### Random Forest: ~80% Accuracy
# #### Logistic Regression: ~70% Accuracy
# #### Decision Tree: ~77% Accuracy
# #### Support Vector Machines: ~47% Accuracy
# 
# I am glossing over a lot of details here.  Different algorithms have different performance speeds and settings that we can tweak to improve their accuracy, precision, and recall.  Random Forests usually perform best with little tweaking although they aren't the fastest in most cases. For this experiment however, I think random forests are fine for building the basic version of our application

# ## After you are satisfied with the results of your model, you can save the model into a .pkl file that you can quickly use to make predictions of new data.  I will fit a new random forest model that uses all of the data I have and save it as 'myRandomForest.pkl'

# In[9]:

import cPickle

def pickle_model(model, modelname):
    with open('models/' + str(modelname) + '.pkl', 'wb') as f:
        return cPickle.dump(model, f)

model = RandomForestClassifier()
model.fit(X,y)
pickle_model(model, "myRandomForest")


# ## After you pkl a model you can open it up later on as so.

# In[10]:

def getModel(pickle_path):
    with open(pickle_path, 'rb') as f:
        return cPickle.load(f)


# # Part 2
# ## Let's see if it works! Making Actual Predictions on new sounds
# I downloaded from YouTube an audio file of a hungry baby crying. (https://www.youtube.com/watch?v=n87mdkR4kIY) I know our dataset probably isn't big enough to make a strong prediction but let's see if we can get an algorithm working that makes a prediction.
# 

# ## Step 1: Load the model from disk into Python

# In[ ]:

model = getModel("models/myRandomForest.pkl")


# ## Step 2:  Chop the wav file and store it in a folder 
# I should have done a better job making the old chop_songs method more reusable, oh well.

# In[11]:

def chop_new_audio(filename, folder):
    handle = wave.open(filename, 'rb')
    frame_rate = handle.getframerate()
    n_frames = handle.getnframes()
    window_size = 1 * frame_rate
    num_secs = int(math.ceil(n_frames/frame_rate))
    #print filename
    last_number_frames = 0
    #Slicing Audio file
    for i in xrange(num_secs):
        
        shortfilename = filename.split(".")[0]
        snippetfilename = folder + '/' + shortfilename + 'snippet' + str(i+1) + '.wav'
        #print snippetfilename
        snippet = wave.open(snippetfilename ,'wb')
        snippet.setnchannels(2)
        snippet.setsampwidth(handle.getsampwidth())
        snippet.setframerate(frame_rate)
        #snippet.setsampwidth(2)
        #snippet.setframerate(11025)
        snippet.setnframes(handle.getnframes())
        snippet.writeframes(handle.readframes(window_size))
        handle.setpos(handle.tell() - 1 * frame_rate)
        #print snippetfilename, ":", snippet.getnchannels(), snippet.getframerate(), snippet.getnframes(), snippet.getsampwidth()
        
        #The last audio slice might be less than a second, if this is the case, we don't want to include it because it will not fit into our matrix 
        if last_number_frames < 1:
            last_number_frames = snippet.getnframes()
        elif snippet.getnframes() != last_number_frames:
            #print "this file doesnt have the same frame size!, remaming file"
            os.rename(snippetfilename, snippetfilename+".bak")
        snippet.close()

    #handle.close()


chop_new_audio("babycryingformilk.wav", "babycryingformilk")


# ## Step 3:  Transform the chopped snippets into MFCC fingerprints and make a prediction

# In[12]:

predictions = []
for i, filename in enumerate(os.listdir('babycryingformilk/')):
    last_number_frames = -1
    if filename.endswith(".wav"):
        #print filename
        audiofile, sr = librosa.load("babycryingformilk/"+filename)
        fingerprint = librosa.feature.mfcc(y=audiofile, sr=sr, n_mfcc=1)
        x = pd.DataFrame(fingerprint, dtype = 'float32')
        prediction = model.predict(fingerprint)
        #print prediction
        predictions.append(prediction[0])


# ## Step 4: Take the mode of the predictions to come up with a final predition

# In[13]:

from collections import Counter
data = Counter(predictions)
print data.most_common()   # Returns all unique items and their counts
print data.most_common(1) 


# So our algorithm worked! (But maybe we just got lucky)  I think when you build a bigger dataset and test with more sound files you will be able to see how well the algorithm really performs.  There are also many ways to adjust the algorithm parameters and test how that affects the accuracy of predictions.  I hope this is useful for getting started!  Let me know if you have any other questions or concerns and I will be more than happy to help!
