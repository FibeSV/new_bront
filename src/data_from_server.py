import requests
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
headers = {'Authorization': 'Token fc6d26398a97b5e0290bc1c396a79425780877a3'}
n_mfcc = 15
cepst = np.empty([0,n_mfcc])

for i in range(2, 60):
    url = 'http://andrey797.pythonanywhere.com/api-data/Before?page='+str(i)
    cep = np.zeros([n_mfcc])
    r = requests.get(url, headers=headers)
    req = r.json()['results'][0]
    for i in range(n_mfcc):
        cep[i] = req['Cepstral_float_'+str(i+1)]
    cepst = np.append(cepst, [cep], axis= 0)

print(cepst.shape)
mfccs = scale(cepst.T, axis=1)
plt.pcolormesh(mfccs)
plt.show()