#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 14:34:06 2020

@author: matt
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
import xgboost as xgb

train_values = pd.read_csv('train_values.csv')
train_labels = pd.read_csv('train_labels.csv')

train_values = train_values.applymap(lambda col: np.nan if col == 995 else col)

train_values.isnull().sum()

from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import category_encoders as ce

#%%
drop_vars = ['building_id',
             'has_superstructure_bamboo',                
             'has_superstructure_mud_mortar_brick',       
             'has_superstructure_adobe_mud',              
             'legal_ownership_status',                    
             'has_secondary_use',                         
             'has_superstructure_stone_flag',             
             'has_superstructure_rc_non_engineered',      
             'has_superstructure_rc_engineered',          
             'has_secondary_use_agriculture',             
             'has_superstructure_cement_mortar_stone',    
             'has_secondary_use_hotel',                   
             'has_superstructure_other',                
             'has_secondary_use_rental',                  
             'has_secondary_use_other',                   
             'has_secondary_use_industry',               
             'has_secondary_use_institution',             
             'has_secondary_use_school',                  
             'has_secondary_use_health_post',            
             'has_secondary_use_gov_office',          
             'has_secondary_use_use_police' ]




#%%
final_train = train_values.drop(drop_vars, axis = 1)

#%%
targ_enc = ce.TargetEncoder()
targ_enc.fit(final_train, train_labels['damage_grade'])
final_train = targ_enc.transform(final_train)

#%%
imp = IterativeImputer(max_iter=20, min_value = 0)
imp.fit(final_train)

final_train = pd.DataFrame(imp.transform(final_train), columns = final_train.columns)


#%%
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(init = RandomForestClassifier(max_depth = 20),
                                 subsample = .85,
                                 n_iter_no_change = 10,
                                 n_estimators=(1000),
                                 learning_rate=(.25),
                                 random_state=(23456))

cvs = cross_val_score(gbc, final_train, train_labels.damage_grade, 
                      cv = 5, n_jobs = (-1), scoring = make_scorer(f1_score, average = 'micro'))
print(cvs)
print("Mean F1 Micro Score: {}".format(np.mean(cvs)))

# [0.72235759 0.71780507 0.72419417 0.72275518 0.72177667] # 100 estimators
# [0.73049251 0.72722563 0.73503454 0.73227168 0.73227168] # 500 estimators, .01 learning rate
# [0.73584544 0.7289716  0.7390637  0.73762471 0.7386416 ] # 1000 estimators, .15 learning rate
# [0.73809021 0.7348043  0.73846892 0.73898695 0.73873753] # 1000 estimators, .225 learning rate
# [0.73927975 0.73263622 0.7414812  0.73889102 0.74000384] # current best sub, 1k, .25
# Mean F1 Micro Score: 0.7384584064476098

if np.mean(cvs) > 0.7384584064476098:
   gbc.fit(final_train, train_labels.damage_grade)
   print('Model fitted')

#%% experiment with dataset balancing
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline

#smote = SMOTETomek(sampling_strategy = 'minority', random_state = 23456)
#smote_train, smote_labels = smote.fit_resample(final_train, train_labels.damage_grade)

gbc = GradientBoostingClassifier(init = RandomForestClassifier(max_depth = 20),
                                 subsample = .85,
                                 n_iter_no_change = 10,
                                 n_estimators=(1000),
                                 learning_rate=(.25),
                                 random_state=(23456))

pipe = Pipeline([('balance', SMOTETomek(sampling_strategy = {1 : 30000}, random_state = 23456)),
                 ('model', gbc)
                 ])


cvs = cross_val_score(pipe, final_train, train_labels.damage_grade, 
                      cv = 5, n_jobs = (-1), scoring = make_scorer(f1_score, average = 'micro'))
print(cvs)
print("Mean F1 Micro Score: {}".format(np.mean(cvs)))

# [0.74029662 0.73653108 0.74155794 0.73915963 0.74044513]
# Mean F1 Micro Score: 0.7395980802125192

#%%

pipe.fit(final_train, train_labels.damage_grade)

#%%
test = pd.read_csv('test_values.csv')
mtest = test.drop(drop_vars, axis = 1)
mtest = targ_enc.transform(mtest)
mtest = pd.DataFrame(imp.transform(mtest), columns = mtest.columns)

test["damage_grade"] = pipe.predict(mtest)
print(test.value_counts('damage_grade')) # check that preds look ok

#%%
test[['building_id', 'damage_grade']].to_csv("best_model_smote.csv", index = False)
