import requests
import datagenerator
# An example to test that the server works correctly.
# It takes one sample for each Iris type, requests prediction and compares it with the right target
if __name__ == '__main__':
    print("hello")
    dg = datagenerator.DataGenerator(1,24,16000)
    feat, target = dg._get_data('/Users/slowm/OneDrive/Desktop/new_bront/data/wav/clear.wav', '/Users/slowm/OneDrive/Desktop/new_bront/data/marks/clear__mark.txt', True, True)
    for i in [0, 50, 100]:
        x = feat[i]
        y = target[i]
        features = {
            'coefs': x.tolist()
        }
        resp = requests.post(
            "http://127.0.0.1:80/predict",
            json=features
        )
        print(f"Input features: {x}")
        print(f"Predicted: {resp.json()}")
        print(f"Expected: {y}")
        print("----")
