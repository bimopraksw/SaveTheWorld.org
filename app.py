from flask import Flask, render_template, url_for, request
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html", result_value={
        "color":"#497174",
        "cluster": "input data untuk melihat prediksi",
        "country": "-",
        "child_mort": "-",
        "exports": "-",
        "healt": "-",
        "import": "-",
        "income": "-",
        "inflation": "-",
        "life_expec": "-",
        "total_fer": "-",
        "gdpp": "-"})


@app.route('/', methods=['POST'])
def get_input_values():
    val = request.form['child_mort']


@app.route('/predict', methods=['POST'])
def predict():
    # kalau user mengecek secara langsung melalui URL
    if request.method == 'GET':
        return 'The URL /predict is accessed directly. Go to the main page firstly'

    # method post akan terjadi pada saat user menekan tombol submit
    if request.method == 'POST':
        input_val = request.form
        scaled_data = []
        # apabila input form tidak kosong
        if input_val != None:
            # collecting values
            vals = []
            for key, value in input_val.items():
                if key != 'country':
                    vals.append(float(value))
            mm_scaler = MinMaxScaler()
            # normalisasi data
            # menggunakan file hasil eksport
            scaler = pickle.load(open('scaler-new.pkl', 'rb'))
            x = scaler.transform([vals])
            # feature reduction
            # mengimport file dari file pca.pkl
            with open('pca.pkl', 'rb') as pickle_file:
                pca = pickle.load(pickle_file)
            # melakukan determinan terhadap data antara input data dengan model PCA yang sudah dibuat
            # Hasil data yang digunakan
            scaled_data = pca.transform(x).tolist()
            # testing
            print(scaled_data)

        # return scaled_data.tolist()

        # Calculate Euclidean distances to freezed centroids
        with open('freezed_centroids.pkl', 'rb') as file:
            freezed_centroids = pickle.load(file)

        assigned_clusters = []
        l = []  # list of distances
        print(scaled_data[0])
        print(freezed_centroids)
        for i, this_segment in enumerate(freezed_centroids):
            dist = distance.euclidean(scaled_data[0], this_segment[:3])
            l.append(dist)

        # mencari nilai terdekat dengan cluster
        res = 9999999999999
        cluster = -1
        for i, dis in enumerate(l):
            # memilih index sebagai cluster
            if dis < res:
                res = dis
                cluster = i
        hasil = ''
        color = ''
        # return {"cluster": str(cluster), "hasil reduksi": scaled_data, "titik pusat cluster": freezed_centroids.tolist(), "jarak tiap cluster" : l}
        if cluster == 0:
            hasil = "This Country Might Need Some Help"
            color = "#FFE15D"
        elif cluster == 1:
            hasil = "This Country Doesn't need Help"
            color = "#68B984"
        else:
            hasil = "This Country Realy Need Some Help"
            color = "#DC3535"

        return render_template(
            'index.html',
            result_value={
                "color":color,
                "cluster": hasil,
                "country": request.form['country'],
                "child_mort": request.form['child_mort'],
                "exports": request.form['exports'],
                "healt": request.form['healt'],
                "import": request.form['import'],
                "income": request.form['income'],
                "inflation": request.form['inflation'],
                "life_expec": request.form['life_expec'],
                "total_fer": request.form['total_fer'],
                "gdpp": request.form['gdpp']}
        )


if __name__ == "__main__":
    app.run(debug=True)
