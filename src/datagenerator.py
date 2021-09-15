import os
from librosa import feature
import sklearn
from sklearn.metrics import classification_report
from sklearn import neighbors
import librosa
import numpy as np
from librosa.core.spectrum import  util
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

class DataGenerator:
    def __init__(self, hop, win,  sr=None, hop_mfcc=512, win_mfcc=2048, n_mels=15, center=False, n_mfcc=15) -> None:
        self.sr = sr
        self.hop = hop
        self.win = win
        self.hop_mfcc = hop_mfcc
        self.win_mfcc = win_mfcc
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.center = center

    def _get_start_end(self, path):

        file = open(path)
        raw_mark = file.read(-1)

        start = np.zeros(len(raw_mark.split('\t\n'))-1)
        end = np.zeros(len(raw_mark.split('\t\n'))-1)

        for i,mark in enumerate(raw_mark.split('\t\n')):
            if len(mark)==0:
                continue
            try :
                start[i], end[i] = mark.split('\t')
            except:
                try:
                    start[i], end[i] = mark.split('\t')[1].split("\n")[1],mark.split('\t')[2]
                except:
                    try:
                        start[i], end[i] = mark.split('\t')[2].split("\n")[1],mark.split('\t')[3]
                    except:
                        pass
        start = np.array(start*self.sr, dtype="int")
        end = np.array(end*self.sr, dtype="int")
        return start, end

    def _get_target(self, path, sound_end):
        start, end = self._get_start_end(path)
        steps = range(0, sound_end-self.win_mfcc-(self.win-1)*self.hop_mfcc+1, self.hop*self.hop_mfcc)
        target = np.zeros(len(steps))
        k=0
        for i,step in enumerate(steps):
            if step>end[k]:
                k+=1
                if k>=end.shape[0]:
                    break
            if (step<start[k]) and (end[k]<(step+self.win_mfcc+(self.win-1)*self.hop_mfcc)):
                target[i]=1
        return target

    def mfcc_features(self, data, transp=False, flat=False):
        mfccs = librosa.feature.mfcc(data, sr =self.sr, n_mfcc=self.n_mfcc, hop_length=self.hop_mfcc, win_length=self.win_mfcc, n_mels=self.n_mels, center=self.center)
        features = self.features_preporation(mfccs, True, True)
        return features

    def features_preporation(self, mfccs, transp=False, flat=False):
        mfccs = sklearn.preprocessing.scale(mfccs, axis=0)
        axis = 0
        if transp:
            mfccs = mfccs.T
        mfccs_frame = util.frame(mfccs, frame_length=self.win, hop_length=self.hop, axis =axis)
        if flat :
            mfccs_frame = mfccs_frame.reshape([mfccs_frame.shape[0], -1])
        return mfccs_frame

    def _get_data(self, sound_path, marks_path, transp = False, flat=False):
        sound_pcm = librosa.load(sound_path, sr=self.sr)[0]
        features = self.mfcc_features(sound_pcm, transp=transp, flat=flat)
        target = self._get_target(marks_path, (sound_pcm.shape[0]//self.hop_mfcc)*self.hop_mfcc)
        return features, target

def _from_denis(path):
    f = open(path)
    data = f.read()
    data = np.array(data.split('\n'))
    mfcc = np.empty([0,15])
    for d in data[:-1]:
        mfcc= np.append(mfcc,np.array(d.split(' ')[:15], dtype=float).reshape([1,-1]),0)
    return mfcc

def model_test(est, den , data, target):
    _, dentest, trainx , testx, trainy, testy = train_test_split(den, data, target, test_size = 0.2, random_state=15)
    est.fit(trainx, trainy)
    print(classification_report(testy, est.predict(testx)))
    print(classification_report(testy, est.predict(dentest)))

if __name__=="__main__":
    dg = DataGenerator(1,24,16000)
    feat, target = dg._get_data('data\wav\clear.wav', 'data\marks\clear__mark.txt', True, True)
    den = dg.features_preporation(_from_denis('RES (9).txt'),flat=True)
    model = KNeighborsClassifier(n_neighbors=3)
    model_test(model, den, feat, target)


    #print(feat.shape, target.shape)
    #knn = KNeighborsClassifier(n_neighbors=5)
    #print(dg.features_preporation(_from_denis('RES (9).txt'),flat=True).shape)
    #dentrain, dentest, trainx , testx, trainy, testy = train_test_split(den, feat, target, test_size = 0.2, random_state=15)
    #knn.fit(trainx, trainy)
    #print(classification_report(testy, knn.predict(testx)))
    #dentrain, dentest, ytr, ytst =  train_test_split(den, target , test_size = 0.2, random_state=15)
    #print(classification_report(testy, knn.predict(dentest)))
    #print(util.frame(librosa.load('data\wav\clear.wav', sr=16000)[0],2048+23*512,512*12).shape)
