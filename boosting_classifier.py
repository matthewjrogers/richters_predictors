#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 14:34:06 2020

@author: matt
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
#import xgboost as xgb


#from sklearn.model_selection import StratifiedKFold
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import category_encoders as ce

from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline

#%%

train_values = pd.read_csv('train_values.csv')
train_labels = pd.read_csv('train_labels.csv')

train_values = train_values.applymap(lambda col: np.nan if col == 995 else col)

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


#%% dataset balancing

gbc = GradientBoostingClassifier(init = RandomForestClassifier(max_depth = 20),
                                 subsample = .85,
                                 n_iter_no_change = 10,
                                 n_estimators=(1000),
                                 learning_rate=(.25),
                                 random_state=(23456))

pipe = Pipeline([('encode', ce.TargetEncoder()),
                 ('impute', IterativeImputer(max_iter=20, min_value = 0)),
                 ('balance', SMOTETomek(sampling_strategy = {1 : 30000}, random_state = 23456)),
                 ('model', gbc)
                 ])


cvs = cross_val_score(pipe, final_train, train_labels.damage_grade, 
                      cv = 5, n_jobs = (-1), scoring = make_scorer(f1_score, average = 'micro'))
print(cvs)
print("Mean F1 Micro Score: {}".format(np.mean(cvs)))

# [0.74029662 0.73653108 0.74155794 0.73915963 0.74044513]
# Mean F1 Micro Score: 0.7395980802125192

if np.mean(cvs) > 0.7395980802125192:
   pipe.fit(final_train, train_labels.damage_grade)
   print('Model fitted')

#%%
bag = BaggingClassifier(n_estimators = 500,
                        max_samples = .85,
                        oob_score = (True),
                        random_state=(23456),
                        max_features=(.75)
                        )

pipe = Pipeline([('encode', ce.TargetEncoder()),
                 ('impute', IterativeImputer(max_iter=20, min_value = 0)),
                 ('balance', SMOTETomek(sampling_strategy = {1 : 30000}, random_state = 23456)),
                 ('model', bag)
                 ])


cvs = cross_val_score(pipe, final_train, train_labels.damage_grade, 
                      cv = 5, scoring = make_scorer(f1_score, average = 'micro'))
print(cvs)
print("Mean F1 Micro Score: {}".format(np.mean(cvs)))

# [0.74348151 0.73775902 0.74282425 0.74353415 0.74169225]
# Mean F1 Micro Score: 0.7418582368322558

#%%
pipe.fit(final_train, train_labels.damage_grade)
print('Model fitted')


#%%
test = pd.read_csv('test_values.csv')
mtest = test.drop(drop_vars, axis = 1)
mtest = targ_enc.transform(mtest)
mtest = pd.DataFrame(imp.transform(mtest), columns = mtest.columns)

test["damage_grade"] = pipe.predict(mtest)
print(test.value_counts('damage_grade')) # check that preds look ok

#%%
test[['building_id', 'damage_grade']].to_csv("bag_dif_max_features_test.csv", index = False)
