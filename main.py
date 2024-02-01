import pandas as pd
from pandas.api.types import is_object_dtype
import streamlit as st
import os
from pandas_profiling import profile_report
from streamlit_pandas_profiling import st_profile_report
import pickle as pkl

import numpy as np
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


with st.sidebar:
    st.title('ML AutoRegression App')
    choice = st.radio('Navigation: ', ['Upload','EDA','Preprocessing','ML model','Download'])

if os.path.exists('./dataset.csv'):
    df = pd.read_csv('dataset.csv', index_col=None)

def cat_num(df, output):
    cat_columns = []
    num_columns = []
    for i in df.columns:
        if is_object_dtype(df[i]):
            cat_columns.append(i)
        else:
            num_columns.append(i)
    cat_columns.remove(output)
    return cat_columns, num_columns


if choice == 'Upload':  
    st.title('Upload Here')
    file = st.file_uploader('Upload your regression dataset')
    if file:
        data = pd.read_csv(file, index_col=None)
        st.dataframe(data.sample(8))
        data.to_csv('dataset.csv', index=None)
    try: 
        if os.path.exists('./dataset.csv'): 
            if st.button('Destroy the current dataset'):
                os.remove('./dataset.csv')
    except NameError:
        pass


if choice == 'EDA':
    try:
        st.title('Exploratory Data Analysis')
        profile = df.profile_report()
        st_profile_report(profile)
    except NameError:
        st.error('Dataset is not uploaded')


if choice == 'Preprocessing':
    st.warning('Still in production phase, sorry for making you wait😊')

if choice == "ML model":
    try: 
        target = st.selectbox('Check the target columns', df.columns)
        if st.button('Run modeling'):
            st.dataframe(df.sample(5))

            df.dropna(axis='index', how='any')

            cat_columns, num_columns = cat_num(df, target)

            X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=[target], axis=1), df[target], test_size=0.2, random_state=3)
            st.write(X_train.shape, 'this is the shape of X_train')
            st.write(X_test.shape, 'this is the shape of X_test')
            st.write(y_train.shape, 'this is the shape of y_train')
            st.write(y_test.shape, 'this is the shape of y_test')

            # making encoder 
            ohe = OneHotEncoder(sparse=False, drop='first')
            le = LabelEncoder()

            # encoding the values
            X_train_cat = ohe.fit_transform(X_train[cat_columns])
            X_test_cat = ohe.transform(X_test[cat_columns])

            X_train_trf = np.hstack((X_train_cat, X_train[num_columns].values))
            X_test_trf = np.hstack((X_test_cat, X_test[num_columns].values))

            y_train = le.fit_transform(y_train) 
            y_test = le.transform(y_test) 

            # shape after encoding 
            st.write(X_train_trf.shape, 'this is the shape of X_train after encoding ')
            st.write(X_test_trf.shape, 'this is the shape of X_test after encoding ')
            st.write(y_train.shape, 'this is the shape of y_train after encoding ')
            st.write(y_test.shape, 'this is the shape of y_test after encoding ')

            # model = SGDRegressor(penalty='l2', eta0=0.01, learning_rate='constant', alpha=0.1)
            model = LinearRegression()

            model.fit(X_train_trf,y_train)
            pred = model.predict(X_test_trf)
            st.write('Model is build...😎')
            st.write('R2 score of: ', r2_score(y_test, pred))

            if ~(os.path.exists('model/pipe.pkl')):
                pkl.dump(model,open('model/pipe.pkl','wb'))
        if os.path.exists('model/pipe.pkl'):
            if st.button('Destroy the model'):
                os.remove('./model/pipe.pkl')
    except NameError:
        st.error('Dataset is not uploaded😯')


if choice == 'Download':
    try:
        with open('model\pipe.pkl', 'rb') as f:
            st.download_button('Download Model', f, file_name='Model.pkl')

    except FileNotFoundError:
        st.error('Model is not built yet😕😥')

