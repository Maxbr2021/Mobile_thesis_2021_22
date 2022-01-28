from sklearn import linear_model
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

#################### HELPER #########################################
#####################################################################

### scale datafram with given scikit-learn scaler ###
def scale_df(df,scaler):
    df_copy = df.copy()
    df_copy[list(df.columns)] = scaler.fit_transform(df[list(df.columns)])
    return df_copy

### build model for list of input features ###
def make_model(x,y,df):
    dependent = ' + '.join(x)
    ols_string = f"{y} ~ " + dependent
    model = smf.ols(ols_string, data=df).fit()
    print(model.summary())
    return model

### leav one out cross validation to measure model performance ###
def leav_one_out_cv(df_features,y,columns=None):
    if columns == None:
        columns= list(df_features.columns)
        X = df_features[columns].copy()
    else:
        X = df_features[columns].copy()
    model = LinearRegression()
    loo = LeaveOneOut()
    loo.get_n_splits(X)
    predictions = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        predictions.append([y_pred[0],y_test.iloc[0]])
    arr = np.array(predictions)
    score = r2_score(arr[:,1],arr[:,0])
    return arr, score
#################### read data ######################################
#####################################################################

PATH = "Final_dataset.xlsx"

### datafram that includes all mobility indicators and and SF12 scores among participants with assessed SF12 scores N=23
df = pd.read_excel(PATH).dropna()
df = df.set_index('user')
df.drop(df.columns[21:33], axis=1, inplace=True)
df = df.rename(columns={'CHull_skm':'CHull','Max_dist':'MaxDist','revisited_places':'NumRevPl',
'unique_places':'NumUniqPl','RevisitedLS':'AvgRevisitedLS','green_ratio':'GreenR','Surr_shops':'NumRes_shop',
'Surr_health':'NumRes_health','Surr_stations':'NumRes_stations','inter_dens_km':'InterDens','health_visits':'NumHealth','shop_visits':'NumShop','morning_p':'TPDistMax_morning',
'noon_p':'TPDistMax_noon','evening_p':'TPDistMax_evening','night_p':'TPDistMax_night','OH_loc':'OHLoc','Dur_atm':'DurATM','Dur_ptm':'DurPTM','Grav_compact':'GravCompact'})

### datafram that includes all mobility inidcators and sf12 questions for N=25 participants
df2 = pd.read_excel(PATH)
df2 = df2.set_index('user')
df2 = df2.rename(columns={'CHull_skm':'CHull','Max_dist':'MaxDist','revisited_places':'NumRevPl',
'unique_places':'NumUniqPl','RevisitedLS':'AvgRevisitedLS','green_ratio':'GreenR','Surr_shops':'NumRes_shop',
'Surr_health':'NumRes_health','Surr_stations':'NumRes_stations','inter_dens_km':'InterDens','health_visits':'NumHealth','shop_visits':'NumShop','morning_p':'TPDistMax_morning',
'noon_p':'TPDistMax_noon','evening_p':'TPDistMax_evening','night_p':'TPDistMax_night','OH_loc':'OHLoc','Dur_atm':'DurATM','Dur_ptm':'DurPTM','Grav_compact':'GravCompact'})

### calculate basic stats for observed mobility indicators
basic_stats = df.describe()
col_names = list(df2.columns)

### split data into x and y
df_features = df[col_names[:21]].copy()
df_ksk12 = df[col_names[-2:]][['KSK12']].copy()
df_psk12 = df[col_names[-2:]][['PSK12']].copy()

############## Plot correlation heatmaps ###########################
####################################################################

cols = col_names[:21]
cols.extend(col_names[-2:])
correlation = round(df[cols].corr(),2)
mask = np.triu(np.ones_like(correlation, dtype=bool))
plt.figure(figsize=(16, 14))
heatmap = sns.heatmap(correlation, annot=True, linewidths=0, vmin=-0.75, vmax=0.75, cmap="RdBu_r",cbar_kws={'label':'Pearson correlation coefficients'})
heatmap.set(xlabel='Mobility indicators')
heatmap.set(ylabel='Mobility indicators')
plt.savefig('corr_heathap_neu.png')

cols2 = col_names[0:21]
cols2.extend(col_names[21:-9])
cols2.extend(col_names[-7:-2])
sf12_q = cols2[-10:]
correlation2 = round(df2[cols2].corr(),2)
plt.figure(figsize=(16, 14))
heatmap = sns.heatmap(correlation2[sf12_q][:21], annot=True, linewidths=0, vmin=-0.75, vmax=0.75, cmap="RdBu_r",cbar_kws={'label':'Pearson correlation coefficients'})
heatmap.set(xlabel='Mobility indicators')
heatmap.set(ylabel='Mobility indicators')
plt.savefig('corr_sf12_heathap_neu.png')


############## build the multiple liner regression model ###########
####################################################################

physical_score = make_model(['NumHealth', 'DurATM', 'TPDistMax_noon', 'NumRevPl','CHull'],'KSK12',df)
physical_score_std = make_model(['NumHealth', 'DurATM', 'TPDistMax_noon', 'NumRevPl','CHull'],'KSK12',scale_df(df,StandardScaler()))
pred_array_p, r2_cv_p = leav_one_out_cv(df_features,df_ksk12,['NumHealth', 'DurATM', 'TPDistMax_noon', 'NumRevPl','CHull'])
print(f"Cross validation R² is {r2_cv_p}")
[print(f"Prediction: {round(a[0],2)} --> True value: {b[0]}") for a,b in pred_array_p ][0]


mental_score = make_model(['OHLoc','NumRes_stations','GreenR','GravCompact','AvgRevisitedLS'],'PSK12',df)
mental_score_std = make_model(['OHLoc','NumRes_stations','GreenR','GravCompact','AvgRevisitedLS'],'PSK12',scale_df(df,StandardScaler()))
pred_array_m, r2_cv_m = leav_one_out_cv(df_features,df_psk12,['OHLoc','NumRes_stations','GreenR','GravCompact','AvgRevisitedLS'])
print(f"Cross validation R² is {r2_cv_m}")
[print(f"Prediction: {round(a[0],2)} --> True value: {b[0]}") for a,b in pred_array_m ][0]
#
