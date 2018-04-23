import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.cross_validation import KFold
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import copy

class Data_PCAReduction:
    def __init__(self):
        pass
    def PCA_Reduction(self, df, columns, components_left, New_name):
        PCA_Original = np.array(df[columns]).T
        pca = PCA(n_components = components_left)
        pca.fit(PCA_Original)
        for i in range(len(columns)):
            if i >= components_left:
                del df[columns[i]]
            else:
                PCA_New = pca.components_[i]
                df[columns[i]] = PCA_New
                index = list(df.columns).index(columns[i])
                df.rename(columns = {df.columns[index]: New_name+'_PCA_'+str(i)}, inplace=True)
        return df

class Data_Preprocessing:
    def __init__(self):
        self.df = pd.read_csv('covtype.csv')
        self.Original_df = self.df
        self.df = self.PCA_Reduction(self.df, ['Hillshade_9am','Hillshade_Noon','Hillshade_3pm'], 1, 'Hillshade')
        self.df = self.PCA_Reduction(self.df, ['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology'], 1, 'Distance_To_Hydrology')
        self.Result_From_OneHotExtraction = self.One_Hot_Extraction(self.df)
        #----------One Hot Data--------#
        self.Soil_Type_OneHot = self.Result_From_OneHotExtraction[0][0]
        self.Area_Type_OneHot = self.Result_From_OneHotExtraction[0][1]
        self.OneHot_SoilArea = np.array([item.tolist()+elem.tolist() for item, elem in zip(self.Soil_Type_OneHot, self.Area_Type_OneHot)])
        #------------------------------#
        self.df = self.One_Hot_Transfer(self.df, self.Result_From_OneHotExtraction)
        self.columns = list(self.df.columns)

    def PCA_Reduction(self, df, columns, components_left, New_name):
        PCA_Original = np.array(df[columns]).T
        pca = PCA(n_components = components_left)
        pca.fit(PCA_Original)
        for i in range(len(columns)):
            if i >= components_left:
                del df[columns[i]]
            else:
                PCA_New = pca.components_[i]
                df[columns[i]] = PCA_New
                index = list(df.columns).index(columns[i])
                df.rename(columns = {df.columns[index]: New_name+'_PCA_'+str(i)}, inplace=True)
        return df

    def One_Hot_Extraction(self, df):
        Area_Num = 4
        Soil_Type = 40
        Soil_Type = ['Soil_Type'+str(i+1) for i in range(Soil_Type)]
        Wilderness_Area = ['Wilderness_Area'+str(i+1) for i in range(Area_Num)]
        result = [np.array(df[Soil_Type]), np.array(df[Wilderness_Area])]
        for i in range(len(Soil_Type)):
            if i != 0:
                del df[Soil_Type[i]]
            else:
                index = list(df.columns).index(Soil_Type[i])
                df.rename(columns = {df.columns[index]: 'Soil_Type'}, inplace = True)
        for i in range(len(Wilderness_Area)):
            if i != 0:
                del df[Wilderness_Area[i]]
            else:
                index = list(df.columns).index(Wilderness_Area[i])
                df.rename(columns = {df.columns[index]: 'Wilderness_Area'}, inplace = True)
        return [result,df]

    def One_Hot_Transfer(self, df, Result_From_OneHotExtraction):
        temp_np = Result_From_OneHotExtraction[0]
        df = Result_From_OneHotExtraction[1]
        Final_Transfered_Data = []
        for item in temp_np:
            New_data = np.array([item[i].tolist().index(1) for i in range(len(item))])
            Final_Transfered_Data.append(New_data)
        df['Soil_Type'] = Final_Transfered_Data[0]
        df['Wilderness_Area'] = Final_Transfered_Data[1]
        return df

class Training_Model:
    def __init__(self, estimators, Type = True):
        self.DP = Data_Preprocessing()
        self.estimators = estimators
        #---------Soil&Area_To_Coverage Training Input&Output-------------#
        self.X_Input_NonePre = np.array(self.DP.df[self.DP.columns[:-2]])
        self.X_Input = preprocessing.scale(np.array(self.DP.df[self.DP.columns[:-2]]))
        self.Y_SoilModel_Output = np.array(self.DP.df['Soil_Type'])
        self.X_Covreage = preprocessing.scale(np.array(self.DP.df[self.DP.columns[:-1]]))
        self.Y_Coverage = np.array(self.DP.df['Cover_Type'])
        #----------------Training-------------------#
        if Type == True:
            self.Saving_MoldelPickle()
        elif Type == False:
            self.Training_SoilArea_To_Coverage(5) # show n-folds validation result
    # validation accuracy: 90%
    def Soil_Type_Training(self, X_Train, Y_Train, X_Test, Y_Test):
        #rf = RandomForestClassifier(n_estimators=self.estimators, class_weight='balanced')
        #rf = GradientBoostingClassifier(n_estimators=self.estimators, learning_rate=0.1, max_depth=10)
        rf = SVC(class_weight='balanced')
        #rf = LogisticRegression(penalty='l2', solver='newton-cg', multi_class='multinomial', class_weight='balanced', max_iter=200)
        Model = rf.fit(X_Train, Y_Train)
        Y_Pre = rf.predict(X_Test)
        Model_Accuracy = accuracy_score(Y_Test, Y_Pre)
        return Y_Pre, Model_Accuracy, Model

    # Validation Accuracy is 96.5%
    def Training_FinalCover(self, X_Train, Y_Train, X_Test, Y_Test):
        #rf = RandomForestClassifier(n_estimators=self.estimators, class_weight='balanced')
        #rf = GradientBoostingClassifier(n_estimators=self.estimators, learning_rate=0.1, max_depth=10)
        rf = SVC(class_weight='balanced')
        #rf = LogisticRegression(penalty='l2', solver='newton-cg', multi_class='multinomial', class_weight='balanced', max_iter=200)
        Model = rf.fit(X_Train, Y_Train)
        Y_Pre = rf.predict(X_Test)
        Model_Accuracy = accuracy_score(Y_Test, Y_Pre)
        return Y_Pre, Model_Accuracy, Model

    # Validation Accuracy: 95%
    def Soil_To_Coverage(self, X_input_tst, Soil_Pre, Y_Coverage_tst, Model):
        X = preprocessing.scale(np.array([item.tolist()+[j] for item, j in zip(X_input_tst, Soil_Pre)]))
        Y = Y_Coverage_tst
        Y_Pre = Model.predict(X)
        Acc = accuracy_score(Y, Y_Pre)
        return Y_Pre, Acc

    def Training_SoilArea_To_Coverage(self, n):
        print ('-----------------Soil_To_Coverage Training Process---------------------')
        kf = KFold(n=len(self.X_Input), n_folds=n, shuffle=True)
        counter = 0
        soil_counter = 0
        cover_counter = 0
        combined_counter = 0
        for tr, tst in kf:
            counter = counter + 1
            print ('The '+str(counter)+'st Fold validation: ')
            Soil_Pre, Soil_Acc, Soil_Model = self.Soil_Type_Training(self.X_Input[tr], self.Y_SoilModel_Output[tr], self.X_Input[tst], self.Y_SoilModel_Output[tst])
            Cover_Pre, Cover_Acc, Cover_Model = self.Training_FinalCover(self.X_Covreage[tr], self.Y_Coverage[tr], self.X_Covreage[tst], self.Y_Coverage[tst])
            Combined_Pre, Combined_Acc = self.Soil_To_Coverage(self.X_Input_NonePre[tst], Soil_Pre, self.Y_Coverage[tst], Cover_Model)
            soil_counter += Soil_Acc
            cover_counter += Cover_Acc
            combined_counter += Combined_Acc
            print ('Soil Accuracy: ' + str(Soil_Acc))
            print ('Coverage Accuracy: ' + str(Cover_Acc))
            print ('CombinedModel Accuracy: ' + str(Combined_Acc))
            break
        print ('-----------------Final Average Accuracy-----------------')
        print ('Average Accuracy of Soil Model: ' + str(soil_counter/counter))
        print ('Average Accuracy of Coverage Model: ' + str(cover_counter/counter))
        print ('Average Accuracy of Combined Model: ' + str(combined_counter/counter))

    def Saving_MoldelPickle(self):
        X_SoilType = preprocessing.scale(np.array(self.DP.df[self.DP.columns[:-2]]))
        Y_SoilType = np.array(self.DP.df['Soil_Type'])
        X_Cover = preprocessing.scale(np.array(self.DP.df[self.DP.columns[:-1]]))
        Y_Cover = np.array(self.DP.df['Cover_Type'])
        rf = RandomForestClassifier(n_estimators=self.estimators)
        Model_SoilType = rf.fit(X_SoilType, Y_SoilType)
        joblib.dump(Model_SoilType, 'Soil_Model.pkl')
        rf = RandomForestClassifier(n_estimators=self.estimators)
        Model_Cover = rf.fit(X_Cover, Y_Cover)
        joblib.dump(Model_Cover, 'Cover_Model.pkl')

    def five_fold(self, X, Y, model, n):
        kf = KFold(n=len(Y), n_folds=n, shuffle=True)
        cv = 0
        average = []
        for tr, tst in kf:
            tr_features = X[tr]
            tr_target = Y[tr]
            tst_features = X[tst]
            tst_target = Y[tst]
            model.fit(tr_features, tr_target)
            tr_accuracy = np.mean(model.predict(tr_features) == tr_target)
            tst_accuracy = np.mean(model.predict(tst_features) == tst_target)
            print ("%d Fold Train Accuracy:%f, Test Accuracy:%f" % (cv+1, tr_accuracy, tst_accuracy))
            average.append(tst_accuracy)
            cv += 1
        ave = sum(np.array(average))/cv
        print ('average accuracy of the model: '+ str(ave))


class Training_ModelXGBoost:
    def __init__(self, Type = True):
        self.DP = Data_Preprocessing()
        #---------Soil&Area_To_Coverage Training Input&Output-------------#
        self.X_Input_NonePre = np.array(self.DP.df[self.DP.columns[:-2]])
        self.X_Input = preprocessing.scale(np.array(self.DP.df[self.DP.columns[:-2]]))
        self.Y_SoilModel_Output = np.array(self.DP.df['Soil_Type'])
        self.X_Covreage = preprocessing.scale(np.array(self.DP.df[self.DP.columns[:-1]]))
        self.Y_Coverage = np.array(self.DP.df['Cover_Type'])
        #----------------Training-------------------#
        if Type == True:
            self.Saving_SoilMoldelPickle()
            self.Saving_CoverMoldelPickle()
        elif Type == False:
            self.Training_SoilArea_To_Coverage() # show n-folds validation result
    # validation accuracy: 90%
    def Soil_Type_Training(self, X_Train, Y_Train, X_Test, Y_Test):
        param = {'max_depth': 20,
                'tree_method':'auto',
                'eta': 0.3,
                'silent': 1,
                'objective': 'multi:softmax',
                'num_class':40,
                'eval_metric':'mlogloss'}
        dtrain = xgb.DMatrix(X_Train, label=Y_Train)
        dtest = xgb.DMatrix(X_Test, label=Y_Test)
        evallist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 100
        Model = xgb.train(param, dtrain, num_round, evallist)
        Y_Pre = Model.predict(dtest)
        Model_Accuracy = accuracy_score(Y_Test, Y_Pre)
        print ('Soil Accuracy: ' + str(Model_Accuracy))
        return Y_Pre, Model_Accuracy, Model

    # Validation Accuracy is 96.5%
    def Training_FinalCover(self, X_Train, Y_Train, X_Test, Y_Test):
        param = {'max_depth': 15,
                'tree_method':'auto',
                'eta': 1,
                'silent': 1,
                'objective': 'multi:softmax',
                'num_class':8,
                'subsample':1,
                'colsample_bytree':1,
                'colsample_bylevel':1,
                'eval_metric':'mlogloss'}
        dtrain = xgb.DMatrix(X_Train, label=Y_Train)
        dtest = xgb.DMatrix(X_Test, label=Y_Test)
        evallist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 100
        Model = xgb.train(param, dtrain, num_round, evallist)
        Y_Pre = Model.predict(dtest)
        Model_Accuracy = accuracy_score(Y_Test, Y_Pre)
        print ('Coverage Accuracy: ' + str(Model_Accuracy))
        return Y_Pre, Model_Accuracy, Model

    # Validation Accuracy: 95%
    def Soil_To_Coverage(self, X_input_tst, Soil_Pre, Y_Coverage_tst, Model):
        X = preprocessing.scale(np.array([item.tolist()+[j] for item, j in zip(X_input_tst, Soil_Pre)]))
        Y = Y_Coverage_tst
        dtest = xgb.DMatrix(X)
        Y_Pre = Model.predict(dtest)
        Acc = accuracy_score(Y, Y_Pre)
        return Y_Pre, Acc

    def Training_SoilArea_To_Coverage(self):
        kf = KFold(n=len(self.X_Input), n_folds=5, shuffle=True)
        for tr, tst in kf:
            Soil_Pre, Soil_Acc, Soil_Model = self.Soil_Type_Training(self.X_Input[tr], self.Y_SoilModel_Output[tr], self.X_Input[tst], self.Y_SoilModel_Output[tst])
            Cover_Pre, Cover_Acc, Cover_Model = self.Training_FinalCover(self.X_Covreage[tr], self.Y_Coverage[tr], self.X_Covreage[tst], self.Y_Coverage[tst])
            Combined_Pre, Combined_Acc = self.Soil_To_Coverage(self.X_Input_NonePre[tst], Soil_Pre, self.Y_Coverage[tst], Cover_Model)
            print ('Soil Accuracy: ' + str(Soil_Acc))
            print ('Coverage Accuracy: ' + str(Cover_Acc))
            print ('CombinedModel Accuracy: ' + str(Combined_Acc))
            break

    def Saving_SoilMoldelPickle(self):
        param = {'max_depth': 20,
                'tree_method':'auto',
                'eta': 0.3,
                'silent': 1,
                'objective': 'multi:softmax',
                'num_class':40,
                'eval_metric':'mlogloss'}
        dtrain = xgb.DMatrix(self.X_Input, label=self.Y_SoilModel_Output)
        dtest = xgb.DMatrix(self.X_Input, label=self.Y_SoilModel_Output)
        evallist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 10
        Model = xgb.train(param, dtrain, num_round, evallist)
        joblib.dump(Model, 'Soil_Model.pkl')

    def Saving_CoverMoldelPickle(self):
        param = {'max_depth': 15,
                'tree_method':'auto',
                'eta': 1,
                'silent': 1,
                'objective': 'multi:softmax',
                'num_class':8,
                'subsample':1,
                'colsample_bytree':1,
                'colsample_bylevel':1,
                'eval_metric':'mlogloss'}
        dtrain = xgb.DMatrix(self.X_Covreage, label=self.Y_Coverage)
        dtest = xgb.DMatrix(self.X_Covreage, label=self.Y_Coverage)
        evallist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = 10
        Model = xgb.train(param, dtrain, num_round, evallist)
        joblib.dump(Model, 'Cover_Model.pkl')


class Use_Model:
    def __init__(self):
        self.Soil_Model = joblib.load('Soil_Model.pkl')
        self.Cover_Model = joblib.load('Cover_Model.pkl')

    def Predict_Soil(self, inputFeatures):
        inputFeatures = preprocessing.scale(np.array([inputFeatures]))
        dtest = xgb.DMatrix(inputFeatures)
        Pre_Y = self.Soil_Model.predict(dtest)[0]
        return Pre_Y

    def Predict_Cover(self, inputFeatures):
        inputFeatures = preprocessing.scale(np.array([inputFeatures]))
        dtest = xgb.DMatrix(inputFeatures)
        Pre_Y = self.Cover_Model.predict(dtest)[0]
        return Pre_Y


class Use_ModelSGBoost:
    def __init__(self, filename = 'test.csv'):
        self.filename = filename
        self.downloadName = filename + '_predicted.csv'
        self.df = pd.read_csv('uploads/'+self.filename)
        self.dfOriginal = copy.deepcopy(self.df)
        self.Soil_Model = joblib.load('Soil_Model.pkl')
        self.Cover_Model = joblib.load('Cover_Model.pkl')
        self.df = self.PCA_Reduction(self.df, ['Hillshade_9am','Hillshade_Noon','Hillshade_3pm'], 1, 'Hillshade')
        self.df = self.PCA_Reduction(self.df, ['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology'], 1, 'Distance_To_Hydrology')
        self.X_Input = preprocessing.scale(np.array(self.df[self.df.columns]))
        if len(self.X_Input[0]) == 9:
            self.Predict_Cover()
        if len(self.X_Input[0]) == 8:
            PreSoil = self.Predict_Soil()
            self.X_Input = preprocessing.scale(np.array(self.df[self.df.columns]))
            dtest = xgb.DMatrix(self.X_Input)
            Pre_Y = self.Cover_Model.predict(dtest)
            self.dfOriginal.insert(len(list(self.dfOriginal.columns)), 'Soil_Pre', PreSoil)
            self.dfOriginal.insert(len(list(self.dfOriginal.columns)), 'Cover_Pre', Pre_Y)
            self.dfOriginal.to_csv('uploads/'+self.downloadName)

    def PCA_Reduction(self, df, columns, components_left, New_name):
        PCA_Original = np.array(df[columns]).T
        pca = PCA(n_components = components_left)
        pca.fit(PCA_Original)
        for i in range(len(columns)):
            if i >= components_left:
                del df[columns[i]]
            else:
                PCA_New = pca.components_[i]
                df[columns[i]] = PCA_New
                index = list(df.columns).index(columns[i])
                df.rename(columns = {df.columns[index]: New_name+'_PCA_'+str(i)}, inplace=True)
        return df

    def Predict_Cover(self):
        dtest = xgb.DMatrix(self.X_Input)
        Pre_Y = self.Cover_Model.predict(dtest)
        self.dfOriginal.insert(len(list(self.dfOriginal.columns)), 'Cover_Pre', Pre_Y)
        self.dfOriginal.to_csv('uploads/'+self.downloadName)

    def Predict_Soil(self):
        dtest = xgb.DMatrix(self.X_Input)
        Pre_Y = self.Soil_Model.predict(dtest)
        self.df.insert(len(list(self.df.columns)), 'Soil_Pre', Pre_Y)
        return Pre_Y
