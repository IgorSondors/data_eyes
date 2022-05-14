import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from scipy import special
import pandas as pd
import numpy as np
import cv2
#import os
#os.environ["DISPLAY"]=":99"

app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
app.prepare(ctx_id=0, det_size=(128, 128))
def lmk_coord(image_file):
    img = cv2.imread(image_file)
    faces = app.get(img)
    if len(faces)==1:
        for face in faces:
            lmk = face.landmark_2d_106
            lmk = np.round(lmk).astype(np.int)
    else:
        lmk = '-1'
    return lmk
    
def draw_landm(basename, left_eye, right_eye, centre_nose):    
    color = (255, 0, 0)
    color2 = (0, 0, 255)
    color3 = (0, 255, 0)

    img = cv2.imread(basename)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(left_eye.shape[0]):
        p = tuple(left_eye[i])
        cv2.circle(img, p, 1, color, 1, cv2.LINE_AA)
        #cv2.putText(img, str(i), p, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color2, 1)

    for i in range(right_eye.shape[0]):
        p = tuple(right_eye[i])
        cv2.circle(img, p, 1, color2, 1, cv2.LINE_AA)
        #cv2.putText(img, str(i), p, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    p = tuple(centre_nose)
    cv2.circle(img, p, 1, color3, 1, cv2.LINE_AA)

    return img


df_test = pd.read_csv('/home/sondors/Documents/data_eyes/val_norm_ok.csv', sep=';')
df_test_new = df_test.copy()

"""print( min(df_test_new['const_left']))
print( max(df_test_new['const_left']))
print( min(df_test_new['const_right']))
print( max(df_test_new['const_right']))

print(df_test_new['const_right'].value_counts())
print(df_test_new['const_left'].value_counts())"""

if False:
    df_test_new['const_left']=df_test_new['const_left']-min(df_test_new.const_left)
    df_test_new['const_right']=df_test_new['const_right']-min(df_test_new.const_right)

    df_test_new['const_left']=(df_test_new['const_left']/max(df_test_new.const_left)*9).astype('int')
    df_test_new['const_right']=(df_test_new['const_right']/max(df_test_new.const_right)*9).astype('int')

print(df_test_new['lndm_ok'].value_counts())
for index, row in df_test_new.iterrows():
    if  row['lndm_ok'] != 2:
        continue
    file = row['filename']
    eyes_left, const_left, eyes_right, const_right, confidence = row['eyes_left'], row['const_left'], row['eyes_right'], row['const_right'], row['confidence']
    lmk = lmk_coord(file)
    print(index)
    print(row['lndm_ok'])

    left_eye = lmk[33:52]
    right_eye = lmk[87:]
    centre_nose = lmk[86]

    left_brow = left_eye[-9:]
    right_brow = right_eye[-9:]    
    left_eye = left_eye[:-9]
    right_eye = right_eye[:-9]

    #img = draw_landm(file, left_eye, right_eye, centre_nose)
    img = cv2.imread(file)
    #print('cv2')

    #title = 'eyes_left: {}, const_left: {}, eyes_right: {}, const_right: {}, confidence: {}'.format(eyes_left, const_left, eyes_right, const_right, confidence)
    title = 'const_left: {}, const_right: {}, \n eyes_left: {}, eyes_right: {}, confidence: {}'.format(const_left, const_right, eyes_left, eyes_right, round(confidence, 2))
    while(1):
        cv2.namedWindow(title, cv2.WND_PROP_FULLSCREEN)
        #cv2.setWindowProperty(title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow(title, 1200,1200)
        cv2.imshow(title,img)
        
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break
        if k == ord('1'):
            df_test_new.loc[index, 'lndm_ok'] = 1 #ok
            print('chosen 1')
            cv2.destroyAllWindows()
            break
        if k == ord('2'):
            df_test_new.loc[index, 'lndm_ok'] = 2 #стоит перепроверить
            print('chosen 2')
            cv2.destroyAllWindows()
            break
        if k == ord('0'):
            df_test_new.loc[index, 'lndm_ok'] = 0 #bad, ландмарки съехали
            print('chosen 0')
            cv2.destroyAllWindows()
            break
    df_test_new.to_csv('/home/sondors/Documents/data_eyes/val_norm_ok.csv', sep=';',index=False)
    print('save')
df_test_new.to_csv('/home/sondors/Documents/data_eyes/val_norm_ok.csv', sep=';',index=False)