from __future__ import print_function
from flask import Flask
import numpy as np
import pandas as pd
from flask import render_template
from flask import request
from flask import abort, redirect, url_for
from CoverModel.project import *
import sys
import copy
app = Flask(__name__)

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
            dict_para['SoilType'] = UM.Predict_Soil(input_Soil)
            input_Cover = temp+[dict_para['SoilType']]
            dict_para['CoverType'] = UM.Predict_Cover(input_Cover)
        elif len(dataDict) == 12:
            dict_para['flag']=2
            df = pd.DataFrame(dataDict)
            df = PCA_Method.PCA_Reduction(df, ['Hillshade_9am','Hillshade_Noon','Hillshade_3pm'], 1, 'Hillshade')
            df = PCA_Method.PCA_Reduction(df, ['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology'], 1, 'Distance_To_Hydrology')
            columns = list(df.columns)
            input_Cover = list(np.array(df[columns])[0])
            UM = Use_Model()
            dict_para['CoverType'] = UM.Predict_Cover(input_Cover)
        else:
            pass
    return render_template('hello.html', dict=dict_para)

if __name__ == '__main__':
  app.run()
