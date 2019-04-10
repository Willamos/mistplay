import pandas as pd
import joblib

class ColDataTypeWorker(object) :
    
    def __init__(self, num_cols, cat_cols):
        self.num_cols = num_cols
        self.cat_cols = cat_cols

    def transform(self, x) :

        dat = x.copy()
        for col in self.num_cols:
            dat[col] = pd.to_numeric(dat[col])
        for col in self.cat_cols:
            dat[col] = dat[col].astype('category')
        return dat

    def fit(self, dat, y=None) :
        return self 


class ColRemoverTransformer(object):
    
    def __init__(self, colname):
        self.colname = colname

    def transform(self, x):
        dat = x.copy()
        dat = dat.drop(columns=self.colname)
        return dat

    def fit(self, dat, y=None) :
        return self



def load_transformers():
    return joblib.load(open('misc/pipeline.pl', 'rb'))

def save_transformers(pipeline):
    joblib.dump(pipeline, open('misc/pipeline.pl', 'wb'))

