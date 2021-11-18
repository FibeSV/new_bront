import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import datagenerator
from progress.bar import IncrementalBar

headers = {'Authorization': 'Token fc6d26398a97b5e0290bc1c396a79425780877a3'}
n_mfcc = 15
cepst = np.empty([0,n_mfcc])
volt = np.zeros([0])

bar = IncrementalBar('Countdown', max = 3600-1500)

for i in range(169, 175):
    url = 'http://andrey797.pythonanywhere.com/api-data/Before?page='+str(i)
    cep = np.zeros([n_mfcc])
    r = requests.get(url, headers=headers)
    for req in r.json()['results']:
    #req = r.json()['results'][0]
        for i in range(n_mfcc):
           cep[i] = req['Cepstral_float_'+str(i+1)]
        cepst = np.append(cepst, [cep], axis= 0)
        volt = np.append(volt, req['BatteryStation'])
        bar.next()
bar.finish()
plt.plot(volt)
plt.show()
#print(req)
davlov = datagenerator._from_denis('data_denis/noviy.TXT')[:50]
davlov_old = datagenerator._from_denis('RES (9).txt')[:50]
data = scale(davlov.T, axis=1)
data_old = scale(davlov_old.T, axis=1)
mfccs = scale(cepst.T, axis=1)
plt.figure(figsize=[20,10])
plt.subplot(3, 1, 1)
plt.title("server")
plt.pcolormesh(mfccs[:,8:58])
plt.subplot(3, 1, 2)
plt.title("file_new")
plt.pcolormesh(data)
plt.subplot(3, 1, 3)
plt.title("old")
plt.pcolormesh(data_old)
plt.show()