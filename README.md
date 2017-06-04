# Baby Cry Analysis & Testing
An instructional writeup for a group of students in Gaza creating an application that classifies different types of baby cries.
## Part 1: Let's train a machine learning algorithm and test it's performance

We will begin by collecting all of the sample audio files we have, chopping them into smaller audio snippets and training a collection of machine learning algorithms with part of this data.  With another part of the data we will test and see how well the algorithms predict data they have never seen before and then choose the best algorithm for our project

### Step 1:  Grab the audio file and its label (we have 3 labels: hungry, pain, and asphyxia)


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


## Step 2:  Chop the audio file into 1 sec. snippets and save them in corresponding folders


```python
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
```

## Step 3:  Transform .wav files to frequency spectrum "fingerprints" using MFCC algorithm


```python
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
```


```python
X.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>34</th>
      <th>35</th>
      <th>36</th>
      <th>37</th>
      <th>38</th>
      <th>39</th>
      <th>40</th>
      <th>41</th>
      <th>42</th>
      <th>43</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>61.164639</td>
      <td>47.077801</td>
      <td>-31.072571</td>
      <td>-96.637245</td>
      <td>-130.160461</td>
      <td>-123.079628</td>
      <td>-122.755928</td>
      <td>-106.404770</td>
      <td>-89.965424</td>
      <td>-90.127457</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>88.442520</td>
      <td>76.313026</td>
      <td>-2.145989</td>
      <td>-30.406242</td>
      <td>-61.038811</td>
      <td>-67.060280</td>
      <td>-54.365772</td>
      <td>-55.774155</td>
      <td>-47.985542</td>
      <td>-15.517380</td>
      <td>...</td>
      <td>-86.546463</td>
      <td>-30.136751</td>
      <td>-7.350064</td>
      <td>-10.286840</td>
      <td>-12.592611</td>
      <td>-13.583686</td>
      <td>-21.501377</td>
      <td>-11.565689</td>
      <td>62.324951</td>
      <td>69.309616</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-29.781023</td>
      <td>-60.232071</td>
      <td>-96.828278</td>
      <td>-90.998222</td>
      <td>-94.779671</td>
      <td>-105.075417</td>
      <td>-97.519058</td>
      <td>-99.080528</td>
      <td>-105.163040</td>
      <td>-106.028229</td>
      <td>...</td>
      <td>-90.644783</td>
      <td>-83.937691</td>
      <td>-80.022545</td>
      <td>-91.994202</td>
      <td>-102.968040</td>
      <td>-90.376099</td>
      <td>-89.551079</td>
      <td>-84.076927</td>
      <td>-11.199362</td>
      <td>3.811244</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-48.579533</td>
      <td>-68.671791</td>
      <td>-113.557213</td>
      <td>-111.534363</td>
      <td>-108.681320</td>
      <td>-106.165474</td>
      <td>-101.677551</td>
      <td>-95.523392</td>
      <td>-102.661011</td>
      <td>-105.140739</td>
      <td>...</td>
      <td>-80.398064</td>
      <td>-92.367661</td>
      <td>-95.758377</td>
      <td>-104.873161</td>
      <td>-106.258102</td>
      <td>-109.926643</td>
      <td>-111.612045</td>
      <td>-98.675140</td>
      <td>-91.144066</td>
      <td>-95.634537</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10.824058</td>
      <td>13.150772</td>
      <td>-6.948591</td>
      <td>-17.075489</td>
      <td>-12.422709</td>
      <td>-0.860229</td>
      <td>3.998694</td>
      <td>3.655787</td>
      <td>-2.229299</td>
      <td>-17.267118</td>
      <td>...</td>
      <td>-9.970703</td>
      <td>-41.767918</td>
      <td>-99.628151</td>
      <td>-110.191643</td>
      <td>-78.499588</td>
      <td>-62.011497</td>
      <td>-63.363140</td>
      <td>-67.923515</td>
      <td>-67.200996</td>
      <td>-55.079521</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 44 columns</p>
</div>



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
```




<script>
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
To toggle on/off output_stderr, click <a href="javascript:code_toggle_err()">here</a>.




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
        Random Forest: 

    //anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1203: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
      sample_weight=sample_weight)
    //anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1304: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
      sample_weight=sample_weight)
    //anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1203: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
      sample_weight=sample_weight)


    (0.80693069306930698, 0.80875431729219438, 0.80693069306930698)
        Logistic Regression: 

    //anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1304: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
      sample_weight=sample_weight)
    //anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1203: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
      sample_weight=sample_weight)
    //anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1304: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
      sample_weight=sample_weight)
    //anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1203: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
      sample_weight=sample_weight)


    (0.70792079207920788, 0.72826645954770641, 0.70792079207920788)
        Decision Tree: (0.7722772277227723, 0.77233616218764733, 0.7722772277227723)
        SVM: (0.41584158415841582, 0.17292422311538083, 0.41584158415841582)


    //anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1074: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)
    //anaconda/lib/python2.7/site-packages/sklearn/metrics/classification.py:1304: DeprecationWarning: The default `weighted` averaging is deprecated, and from version 0.18, use of precision, recall or F-score with multiclass or multilabel data or pos_label=None will result in an exception. Please set an explicit value for `average`, one of (None, 'micro', 'macro', 'weighted', 'samples'). In cross validation use, for instance, scoring="f1_weighted" instead of scoring="f1".
      sample_weight=sample_weight)


## Results Model: (Accuracy, Precision, Recall)
We tested four popular machine learning algorithms to see which ones had the most accuracte predictions with our test data, here are the results:
#### Random Forest: ~80% Accuracy
#### Logistic Regression: ~70% Accuracy
#### Decision Tree: ~77% Accuracy
#### Support Vector Machines: ~47% Accuracy

I am glossing over a lot of details here.  Different algorithms have different performance speeds and settings that we can tweak to improve their accuracy, precision, and recall.  Random Forests usually perform best with little tweaking although they aren't the fastest in most cases. For this experiment however, I think random forests are fine for building the basic version of our application

## After you are satisfied with the results of your model, you can save the model into a .pkl file that you can quickly use to make predictions of new data.  I will fit a new random forest model that uses all of the data I have and save it as 'myRandomForest.pkl'


```python
import cPickle

def pickle_model(model, modelname):
    with open('models/' + str(modelname) + '.pkl', 'wb') as f:
        return cPickle.dump(model, f)

model = RandomForestClassifier()
model.fit(X,y)
pickle_model(model, "myRandomForest")
```

## After you pkl a model you can open it up later on as so.


```python
def getModel(pickle_path):
    with open(pickle_path, 'rb') as f:
        return cPickle.load(f)
```

# Part 2
## Let's see if it works! Making Actual Predictions on new sounds
I downloaded from YouTube an audio file of a hungry baby crying. (https://www.youtube.com/watch?v=n87mdkR4kIY) I know our dataset probably isn't big enough to make a strong prediction but let's see if we can get an algorithm working that makes a prediction.


## Step 1: Load the model from disk into Python


```python
model = getModel("models/myRandomForest.pkl")
```

## Step 2:  Chop the wav file and store it in a folder 
I should have done a better job making the old chop_songs method more reusable, oh well.


```python
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
```

## Step 3:  Transform the chopped snippets into MFCC fingerprints and make a prediction


```python
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
```

## Step 4: Take the mode of the predictions to come up with a final predition


```python
from collections import Counter
data = Counter(predictions)
print data.most_common()   # Returns all unique items and their counts
print data.most_common(1) 
```

    [('hungry', 29)]
    [('hungry', 29)]


So our algorithm worked! (But maybe we just got lucky)  I think when you build a bigger dataset and test with more sound files you will be able to see how well the algorithm really performs.  There are also many ways to adjust the algorithm parameters and test how that affects the accuracy of predictions.  I hope this is useful for getting started!  Let me know if you have any other questions or concerns and I will be more than happy to help!
