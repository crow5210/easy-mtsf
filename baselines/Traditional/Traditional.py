from darts.models import *
import pickle
import numpy as np

class Traditional():
    def __init__(self,name,args):
        MODEL_DICT = {
            "NAIVE":NaiveSeasonal,
            "ES":ExponentialSmoothing,
            "Theta":Theta,
            "ARIMA":ARIMA,
            "Kalman":KalmanForecaster,
            "LGBM":LightGBMModel,
            "Random_Forest":RandomForest
        }
        self.model = MODEL_DICT[name](**args)

    def fit(self,train):
        self.train = train
    

    def predict_univariate(self,test,pred_len):
        preds = []
        for i in range(len(self.train)):
            tr = self.train[i]
            te = test[i]
            pred = []
            for j in range(0,len(te),pred_len):
                if pred_len+j>len(te):
                    break

                if j>0:
                    self.model.fit(tr.append(te[:j]))
                else:
                    self.model.fit(tr)
                
                print
                pred.append(self.model.predict(pred_len)._xa.values[:,0,0][None,:])
            preds.append(np.concatenate(pred)[:,:,None])
        return np.concatenate(preds,axis=-1)
    
    def predict_multivariate(self,test,pred_len):
        pred = []
        for j in range(0,len(test),pred_len):
            if pred_len+j>len(test):
                break

            if j>0:
                self.model.fit(self.train.append(test[:j]))
            else:
                self.model.fit(self.train)
            
            pred.append(self.model.predict(pred_len)._xa.values[:,:,0][None,:,:])
        return np.concatenate(pred,axis=0)
    
    def predict(self,test,pred_len):
        if isinstance(test,list):
            return self.predict_univariate(test,pred_len)
        else:
            return self.predict_multivariate(test,pred_len)
    
    def save(self,save_path):
        with open(save_path, "wb") as handle:
            pickle.dump(obj=self, file=handle)

    
    def load(self,save_path):
        with open(save_path, "rb") as handle:
            model = pickle.load(file=handle)

        return model