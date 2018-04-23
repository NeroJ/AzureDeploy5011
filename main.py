from __future__ import print_function
from flask import Flask
import numpy as np
import pandas as pd
from flask import render_template
from flask import request
from flask import abort, redirect, url_for
from flask import send_from_directory
from CoverModel.project import *
from werkzeug.utils import secure_filename
import sys
import os
import copy

UPLOAD_FOLDER = os.getcwd()+'/uploads'
ALLOWED_EXTENSIONS = set(['csv'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads', methods=['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #print(filename,file=sys.stderr)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            Use_ModelSGBoost(filename = filename)
            return redirect(url_for('uploaded_file',
                                    filename=filename +'_predicted.csv'))
    return render_template('upfile.html')

@app.route('/downloads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/', methods=['POST', 'GET'])
def GPSData_Prediction():
    dict_para = {}
    dict_para['flag']=0
    dict_para['SoilType']=0
    dict_para['CoverType']=0
    dataDict = {}
    if request.method == 'POST':
        PCA_Method = Data_PCAReduction()
        data_dict = request.form
        for item in data_dict.items():
            if item[1] != '':
                dataDict[item[0]] = [float(item[1])]
            else:
                pass
        if len(dataDict) == 11:
        #print(len(dataDict),file=sys.stderr)
            dict_para['flag']=1
            df = pd.DataFrame(dataDict)
            df = PCA_Method.PCA_Reduction(df, ['Hillshade_9am','Hillshade_Noon','Hillshade_3pm'], 1, 'Hillshade')
            df = PCA_Method.PCA_Reduction(df, ['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology'], 1, 'Distance_To_Hydrology')
            columns = list(df.columns)
            input_Soil = list(np.array(df[columns])[0])
            temp = copy.deepcopy(input_Soil)
            UM = Use_Model()
            dict_para['SoilType'] = int(UM.Predict_Soil(input_Soil))
            input_Cover = temp+[dict_para['SoilType']]
            dict_para['CoverType'] = int(UM.Predict_Cover(input_Cover))
        elif len(dataDict) == 12:
            dict_para['flag']=2
            df = pd.DataFrame(dataDict)
            df = PCA_Method.PCA_Reduction(df, ['Hillshade_9am','Hillshade_Noon','Hillshade_3pm'], 1, 'Hillshade')
            df = PCA_Method.PCA_Reduction(df, ['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology'], 1, 'Distance_To_Hydrology')
            columns = list(df.columns)
            input_Cover = list(np.array(df[columns])[0])
            UM = Use_Model()
            dict_para['CoverType'] = int(UM.Predict_Cover(input_Cover))
        else:
            pass
    return render_template('hello.html', dict=dict_para)

if __name__ == '__main__':
  app.run()
