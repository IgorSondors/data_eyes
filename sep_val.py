import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from scipy import special
import pandas as pd
import numpy as np
import cv2



df_test = pd.read_csv('/home/sondors/Documents/data_eyes/additional_tags_podium_const.csv', sep=';')
df_test_new = df_test.copy()

val_1 = df_test_new.loc[df_test_new['const_left'].isin([-1,float('inf'),1,0.97,0.42,0.4])]
val_2 = df_test_new.loc[~df_test_new['const_left'].isin([-1,float('inf'),1,0.97,0.42,0.4])]
val_1 = pd.concat([val_1, val_2.loc[val_2['const_right'].isin([-1,float('inf'),1,0.97,0.42,0.4])]], ignore_index=True, sort=False)
val_2 = val_2.loc[~val_2['const_right'].isin([-1,float('inf'),1,0.97,0.42,0.4])]


print(val_1)

print(val_2)
#val_1.to_csv('/home/sondors/Documents/data_eyes/val_bad.csv', sep=';',index=False)
#val_2.to_csv('/home/sondors/Documents/data_eyes/val_norm.csv', sep=';',index=False)

def select_data(df_test_new):
    """
    light_exposure:
    <0.15 - очень темные
    >0.9 - очень светлые
    примерно [0.4, 0.6] - норма
    light_background
    >0.95 - яркий фон
    примерно [0.4, 0.6] - норма
    light_lighting
    >0.95 - сильно неравномерно освещено лицо
    примерно [0, 0.5] - норма
    face_brightness
    <0.3 - темные лица
    sharpness
    =0 - сильно размыто
    примерно >0.625 нормально
    left_right и up_down по модулю меньше 0.3
    object<0.9
    """
    age, two_faces, podium, normalize = True, True, True, True
    df_bad = df_test_new.copy()

    if podium:
        #df_test_new = df_test_new[df_test_new.sharpness>0.625]
        
        df_test_new = df_test_new[abs(df_test_new.left_right)<0.4]
        df_test_new = df_test_new[abs(df_test_new.up_down)<0.6]
        df_test_new = df_test_new[abs(df_test_new.sharpness)>0.1]

        df_test_new = df_test_new[df_test_new.eyes_left!=-6]
        df_test_new = df_test_new[df_test_new.eyes_right!=-6]
        df_test_new = df_test_new[df_test_new.eyes_left!=-4]
        df_test_new = df_test_new[df_test_new.eyes_right!=-4]

        df_bad = df_bad[(~df_bad.filename.isin(df_test_new.filename))]

    
    return df_test_new, df_bad

val_2_clear,df_bad = select_data(val_2)

df_bad = pd.concat([val_1, df_bad], ignore_index=True, sort=False)


df_bad.to_csv('/home/sondors/Documents/data_eyes/val_bad_additional.csv', sep=';',index=False)
val_2_clear.to_csv('/home/sondors/Documents/data_eyes/val_norm_additional.csv', sep=';',index=False)

print('check')
print(val_2)
print(val_2_clear)

print(df_bad)


print(val_2_clear)

print('val')
print(val_2['const_left'].value_counts())

print( min(val_2['const_left']))
print( max(val_2['const_left']))

print(val_2['const_right'].value_counts())
print( min(val_2['const_right']))
print( max(val_2['const_right']))

print('val_clear')
print(val_2_clear['const_left'].value_counts())

print( min(val_2_clear['const_left']))
print( max(val_2_clear['const_left']))

print(val_2_clear['const_right'].value_counts())
print( min(val_2_clear['const_right']))
print( max(val_2_clear['const_right']))
