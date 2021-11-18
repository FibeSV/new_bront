import os

file_names = os.listdir(path = "data\wav")

for file in file_names:
    audio = librosa.load(file, sr=...)