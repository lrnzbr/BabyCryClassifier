
# Baby Cry Analysis & Testing

## Step 1:  Grab the audio file and its label


```python
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
```

#If you are having issues with python being able to make the different directories, try doing it manually or with the os.system commands

```os.system("sudo mkdir /audio")```


## Step 2:  Chop the audio file into 1 sec. snippets and save them in separate folders


```python
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
```

    ('chopping: Full_pain/17.wav', 'pain')
    ('chopping: Full_pain/24.wav', 'pain')
    ('chopping: Full_hunger/39.wav', 'hungry')
    ('chopping: Full_hunger/51.wav', 'hungry')
    ('chopping: Full_pain/22.wav', 'pain')
    ('chopping: Full_hunger/42.wav', 'hungry')
    ('chopping: Full_pain/7.wav', 'pain')
    ('chopping: Full_asphyxia/63.wav', 'asphyxia')
    ('chopping: Full_pain/72.wav', 'pain')
    ('chopping: Full_pain/10a.wav', 'pain')
    ('chopping: Full_hunger/47.wav', 'hungry')
    ('chopping: Full_pain/10.wav', 'pain')
    ('chopping: Full_pain/73.wav', 'pain')
    ('chopping: Full_hunger/37.wav', 'hungry')
    ('chopping: Full_hunger/83.wav', 'hungry')
    ('chopping: Full_hunger/4.wav', 'hungry')
    ('chopping: Full_asphyxia/64.wav', 'asphyxia')
    ('chopping: Full_asphyxia/67.wav', 'asphyxia')
    ('chopping: Full_pain/13.wav', 'pain')
    ('chopping: Full_hunger/80.wav', 'hungry')
    ('chopping: Full_hunger/18.wav', 'hungry')
    ('chopping: Full_hunger/88a.wav', 'hungry')
    ('chopping: Full_hunger/36.wav', 'hungry')
    ('chopping: Full_hunger/5.wav', 'hungry')
    ('chopping: Full_hunger/43.wav', 'hungry')
    ('chopping: Full_hunger/23.wav', 'hungry')
    ('chopping: Full_pain/14.wav', 'pain')
    ('chopping: Full_hunger/35.wav', 'hungry')
    ('chopping: Full_hunger/75.wav', 'hungry')
    ('chopping: Full_pain/50.wav', 'pain')
    ('chopping: Full_pain/20.wav', 'pain')
    ('chopping: Full_hunger/44.wav', 'hungry')
    ('chopping: Full_pain/6.wav', 'pain')
    ('chopping: Full_pain/19.wav', 'pain')
    ('chopping: Full_hunger/48.wav', 'hungry')
    ('chopping: Full_pain/2.wav', 'pain')
    ('chopping: Full_hunger/34.wav', 'hungry')
    ('chopping: Full_hunger/33.wav', 'hungry')
    ('chopping: Full_asphyxia/66.wav', 'asphyxia')
    ('chopping: Full_pain/3.wav', 'pain')
    ('chopping: Full_hunger/77.wav', 'hungry')
    ('chopping: Full_pain/9b.wav', 'pain')
    ('chopping: Full_pain/15.wav', 'pain')
    ('chopping: Full_pain/74.wav', 'pain')
    ('chopping: Full_pain/1.wav', 'pain')
    ('chopping: Full_pain/49.wav', 'pain')
    ('chopping: Full_hunger/41.wav', 'hungry')
    ('chopping: Full_hunger/16.wav', 'hungry')
    ('chopping: Full_hunger/11.wav', 'hungry')
    ('chopping: Full_hunger/46.wav', 'hungry')
    ('chopping: Full_hunger/38.wav', 'hungry')
    ('chopping: Full_hunger/79.wav', 'hungry')
    ('chopping: Full_asphyxia/68.wav', 'asphyxia')
    ('chopping: Full_hunger/21.wav', 'hungry')
    ('chopping: Full_hunger/12.wav', 'hungry')
    ('chopping: Full_pain/45.wav', 'pain')
    ('chopping: Full_hunger/76.wav', 'hungry')
    ('chopping: Full_hunger/81.wav', 'hungry')
    ('chopping: Full_hunger/40.wav', 'hungry')
    ('chopping: Full_pain/32.wav', 'pain')
    ('chopping: Full_pain/8.wav', 'pain')
    ('chopping: Full_pain/9a.wav', 'pain')
    ('chopping: Full_asphyxia/65.wav', 'asphyxia')


## Step 3:  Transform .wav to frequency spectrum
Some files had some sample rate issues and caused errors in this step. If this happens, print out the file name at each pass and if one of the files is causing an error, go ahead and delete it for now


```python
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
```

## Step 4:  Make a Test-Train-Split of the data


```python
from sklearn.cross_validation import train_test_split


y = X[44]
del X[44]
X = X.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y)

```

## Step 5:  Fit the training data to a model & Check the models performance against the test data¶


```python
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
        return model.score(X_test, y_test), \
               precision_score(y_test, y_predict), \
               recall_score(y_test, y_predict)

print "    Model, Accuracy, Precision, Recall"
print "    Random Forest:", get_scores(RandomForestClassifier, X_train, X_test, y_train, y_test, n_estimators=25, max_features=5)
print "    Logistic Regression:", get_scores(LogisticRegression, X_train, X_test, y_train, y_test)
print "    Decision Tree:", get_scores(DecisionTreeClassifier, X_train, X_test, y_train, y_test)
print "    SVM:", get_scores(SVC, X_train, X_test, y_train, y_test)
#print "    Naive Bayes:", get_scores(MultinomialNB, X_train, X_test, y_train, y_test)
```

         Model, Accuracy, Precision, Recall
        Random Forest: (0.967741935483871, 0.97235023041474655, 0.967741935483871)
        Logistic Regression: (0.83870967741935487, 0.83136200716845876, 0.83870967741935487)
        Decision Tree: (1.0, 1.0, 1.0)
        SVM: (0.5161290322580645, 0.26638917793964617, 0.5161290322580645)


    //anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1203: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
      sample_weight=sample_weight)
    //anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1304: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
      sample_weight=sample_weight)
    //anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1203: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
      sample_weight=sample_weight)
    //anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1304: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
      sample_weight=sample_weight)
    //anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1203: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
      sample_weight=sample_weight)
    //anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1304: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
      sample_weight=sample_weight)
    //anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1203: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
      sample_weight=sample_weight)
    //anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1304: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
      sample_weight=sample_weight)


## Results Model: (Accuracy, Precision, Recall)
    Random Forest: (0.967741935483871, 0.97235023041474655, 0.967741935483871)
    Logistic Regression: (0.83870967741935487, 0.83136200716845876, 0.83870967741935487)
    Decision Tree: (1.0, 1.0, 1.0)
    SVM: (0.5161290322580645, 0.26638917793964617, 0.5161290322580645)

## After you are satisfied with the results of your model, you can save the model into a .pkl file that you can quickly use to make predictions of new data


```python
def pickle_model(model, modelname):
    with open('../models/' + str(modelname) + '.pkl', 'wb') as f:
        return cPickle.dump(model, f)
    

```

## After you pkl a model you can open it up later on as so.


```python
def getModel(pickle_path):
    with open(pickle_path, 'rb') as f:
        return cPickle.load(f)
```

# Making Actual Predictions on new sounds

You can decide how you want to set up your system to receive and process the audio files it receives. I chose to use concurrent queues in order to make multiple predictions and take the most common result of multiple 1-second slices of audio input.  You may choose to do something else, but as of right now, your algorithm makes a prediction based upon just one second of sound.  Here is some code that I used to make predictions after the audio was already chopped, MFCC transformed into an audio "fingerprint" , and placed in a queue.


```python
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

```

I hope this is useful!  Let me know if you have any other questions or concerns and I will be more than happy to help!
