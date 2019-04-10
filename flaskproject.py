from flask import Flask, jsonify, flash, request, redirect, render_template
import pandas as pd
from transformers import load_transformers
import pickle

import os

localhost = '0.0.0.0'

ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
# flask uploader folder
app.config['UPLOAD_FOLDER'] = 'download/'

# loading model and transformer
model = pickle.load(open('misc/rf_model.mdl', 'rb'))
transformer = load_transformers()


# print(transformer.named_steps['type_setter'].num_cols)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



@app.route('/')
def greeting():
    return redirect('/predict')


@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(full_path)
            values = str(predict(full_path))
            values = values.replace('\n', '')
            return  jsonify({'prediction': values})
    return render_template('upload.html')


def predict(filename):
    values = ''
    df = pd.read_csv(filename)

    df['x23'] = df['x23'].apply(lambda x: 0 if x == 'FALSE' else 1)
    print(df.shape, type(df), df.dtypes)
    inputs = transformer.transform(df)

    if model is not None:
        values = model.predict(inputs)
    return values



app.run(host=localhost, port='3000')

