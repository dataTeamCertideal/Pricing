
# this script takes as input the data of BM as csv file and propse an 
#    optimized_bspline inverse as clean version of the DATA as csv file


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os
import re
from datetime import datetime, timedelta
import time
import seaborn as sns
import plotly.express as px
import sklearn
from sklearn.preprocessing import SplineTransformer
from datetime import datetime, date
import matplotlib
import numpy as np
import scipy.interpolate as interpolate
from scipy.interpolate import LSQUnivariateSpline,UnivariateSpline

from geomdl import BSpline
from geomdl import utilities
from geomdl import exchange
from geomdl.visualization import VisMPL
from geomdl.knotvector import generate
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.pipeline import make_pipeline
import plotly.graph_objects as go
import matplotlib.pyplot as pltt

from yellowbrick.cluster import KElbowVisualizer
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
import random
from sklearn.metrics import mean_squared_error
print('The scikit-learn version is {}.'.format(sklearn.__version__))

# Function to generate a sorted list 
# of random numbers in a given
# range with unique elements

def createRandomSortedList(num, start = 4, end = 90):
    arr = []
    tmp = random.randint(start, end)
     
    for x in range(num):
         
        while tmp in arr:
            tmp = random.randint(start, end)
             
        arr.append(tmp)
         
    arr.sort()
     
    return arr

# Function to make a pre process of data

def pre_process_for_b_splines(df, shop_=1):
    df_ = df.copy() 
    
    #Select the market and median price per model since we have multiple scrapping
    df_ = df_[df_.id_shop==shop_].drop(['id_shop'], axis=1)
    df_ = df_.groupby(
      [
        'date',
        'name',
        'capacite',
        'etat'
        ]
        )['price'].median().reset_index()
    
    #take into account the duration of the sells start date
    df_['date'] = pd.to_datetime(df_['date'])
    df_['dateRef'] = df_.groupby(["name","capacite"])['date'].transform('min')
    df_['day_number'] = (df_['date']-df_['dateRef']).dt.days
    #df_.drop("dateRef", axis=1, inplace= True)
    df_['product'] = df_['name'] + " " + df_['capacite'] + " " + df_['etat']
    return df_[['dateRef', 'day_number', 'product', 'price']]

# Function that generate bsplines based on specific knots

def y_hat_b_spline(df, knots):
    df_ = df.copy()
    df_pivoted = df_.pivot(
        index='product',
        columns='day_number',
        values = 'price'
    )
    dates = df_pivoted.columns
    
    df_pivoted_filled = df_pivoted.T.apply(
        lambda y: interpolate.splrep(
            dates, y, s=0, k=3, t=knots
        )).T
    df_pivoted['prices'] = df_pivoted.values.tolist()
    df_pivoted_filled.columns=['knots', 'bSplines_coef', 'deg_pol']
    tmp = df_pivoted_filled.T.apply(
        lambda x: interpolate.BSpline(*x.values)(dates)
    ).interpolate(method='linear', limit_direction='forward', axis=0
                 ).T.interpolate(
        method='linear', limit_direction='forward', axis=1
    )
    tmp = tmp[tmp.notna().any(axis=1)]
    tmp['predictedP'] = tmp.values.tolist()
    
    df_pivoted_filled = pd.merge(
        df_pivoted_filled, 
        tmp.iloc[:,-1:], 
        left_index=True, 
        right_index=True
    )
    
    df_pivoted_filled = pd.merge(
        df_pivoted_filled, 
        df_pivoted.iloc[:,-1:], 
        left_index=True, 
        right_index=True
    )
    
    df_pivoted_filled["rmse"] = (
        df_pivoted_filled.apply(
            lambda x: mean_squared_error(x['prices'], x['predictedP'], squared=False), 
            axis=1
        )
    )
    
    return df_pivoted_filled

# The main process of optimized_bspline
def bSplinesOpt(df, linear = False, iterations=50, num_val=18, list_df=[], list_knots=[]):
    df_ = df.copy()
    if linear:
        knots=np.linspace(10, 90, num_val, dtype=int, endpoint=False)
        b_splines_df = y_hat_b_spline(df_, knots)
        list_df.append(
            knots,
            b_splines_df["rmse"].min(),
            b_splines_df["rmse"].median(), 
            b_splines_df["rmse"].mean(),
            float(b_splines_df["rmse"].quantile([0.75]).values),
            float(b_splines_df["rmse"].quantile([0.95]).values),            
            b_splines_df["rmse"].max()
        )
    else :
        for iterate in range(iterations):
            knots = createRandomSortedList(num_val)
            print('knots:', knots)
            b_splines_df = y_hat_b_spline(df_, knots)
            print("min : %f, med : %f, mean : %f, Q3 : %f, 95Q : %f, max : %f" %(
                np.round(b_splines_df["rmse"].min(), 2),
                np.round(b_splines_df["rmse"].median(), 2), 
                np.round(b_splines_df["rmse"].mean(), 2), 
                np.round(float(b_splines_df["rmse"].quantile([0.75]).values), 2),
                np.round(float(b_splines_df["rmse"].quantile([0.95]).values), 2),
                np.round(b_splines_df["rmse"].max(), 2))
            )
            list_df.append([
                knots,
                b_splines_df["rmse"].min(),
                b_splines_df["rmse"].median(), 
                b_splines_df["rmse"].mean(),
                float(b_splines_df["rmse"].quantile([0.75]).values),
                float(b_splines_df["rmse"].quantile([0.95]).values),
                b_splines_df["rmse"].max(),
            ])
    return pd.DataFrame(
        list_df, 
        columns=[
            "knots", 
            "min", 
            "med", 
            "mean", 
            "Q3",
            "95Quantile",
            "max"
            ]
            )


# The main function and the generation of the inverse of bsplines

def main():

    df_new = pd.read_csv('Data_brute.csv',sep=',')

    tmp = pre_process_for_b_splines(df_new, shop_=1)

    result = bSplinesOpt(tmp, linear = False, iterations=50, num_val=18)
    
    optimized_bspline = y_hat_b_spline(tmp, result[result['mean'] == result['mean'].min()].iloc[0,0])

    b_sp_clust = optimized_bspline[['bSplines_coef']]
    split = pd.DataFrame(b_sp_clust['bSplines_coef'].to_list())
    split = split.add_prefix('index_')
    b_sp_clust.reset_index()
    b_sp_clust = pd.concat([b_sp_clust.reset_index(),split], axis=1).set_index('product')
    b_sp_4clustering = b_sp_clust.iloc[:,1:-4]

    # distortions = []
    # K = range(1,10)
    # for k in K:
    #     kmeanModel = KMeans(n_clusters=k)
    #     kmeanModel.fit(b_sp_4clustering)
    #     distortions.append(kmeanModel.inertia_)
    # plt.figure(figsize=(16,8))
    # plt.plot(K, distortions, 'bx-')
    # plt.xlabel('k')
    # plt.ylabel('Distortion')
    # plt.title('The Elbow Method showing the optimal k')
    # plt.show()

    kmeanModel = KMeans(n_clusters=3)
    kmeanModel.fit(b_sp_4clustering)

    b_sp_4clustering['Clusters']=kmeanModel.predict(b_sp_4clustering)

    centroids = kmeanModel.cluster_centers_
    cluster_labels = kmeanModel.labels_

    centroids_df = pd.DataFrame(
         centroids, 
         index=['centroid1','centroid2','centroid3'], 
         columns = b_sp_4clustering.iloc[:,:-1].columns
    ) 
    centroids_df['Clusters']=[0,1,2]

    clusters_N_centroids = pd.concat([b_sp_4clustering,centroids_df], axis=0)


    df_pivoted = tmp.pivot(
        index='product',
        columns='day_number',
        values = 'price'
    )
    
    dates = df_pivoted.columns

    optimized_bspline = y_hat_b_spline(
        tmp, 
       result2[result2['mean'] == result2['mean'].min()].iloc[0,0]
    )

    for y in optimized_bspline.index :
         print('\n Product : ', y)
         y_fit2 = optimized_bspline.loc[y].predictedP
         plt.plot(dates, optimized_bspline.loc[y].prices, 'g', label="linear interpolation")
         plt.plot(dates, y_fit2, '-k', label="B-spline")
         plt.title("B-Splines Price Evolution")
         plt.legend(loc='best', fancybox=True, shadow=True)
         plt.grid()
         plt.show() 

    optimized_bspline.explode('predictedP','prices')

    optimized_bspline = optimized_bspline.assign(
         day_number=[np.arange(0,95) for i in optimized_bspline.index]
    )

    data_with_bs = tmp.merge(
         optimized_bspline[
          ['day_number','predictedP','prices']
         ].apply(pd.Series.explode).reset_index().rename(columns={"prices": "price"}),
         how='left',
         on=['day_number','product','price']
    )

    data_with_bs['day'] = data_with_bs['dateRef'] + pd.to_timedelta(
        data_with_bs['day_number'], 
        unit='D'
    )

    data_with_bs.to_csv('Data_clean.csv')


# call main
main()
