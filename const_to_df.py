import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

from scipy import special
import pandas as pd
import numpy as np
import time
import cv2
import os

def eye_hw(lmk):
    def Euclidian_distant(xy1, xy2):
        return ((xy1[0]-xy2[0])**2+(xy1[1]-xy2[1])**2)**0.5
    
    left_eye = lmk[33:52]
    right_eye = lmk[87:]
    centre_nose = lmk[86]

    left_brow = left_eye[-9:]
    right_brow = right_eye[-9:]    
    left_eye = left_eye[:-9]
    right_eye = right_eye[:-9]

    left_eye_max_h = Euclidian_distant(left_eye[0], left_eye[7])
    left_eye_h = 1/3*(
                 Euclidian_distant(left_eye[3], left_eye[8])+
                 Euclidian_distant(left_eye[0], left_eye[7])+
                 Euclidian_distant(left_eye[4], left_eye[9]))
    left_eye_w = Euclidian_distant(left_eye[2], left_eye[6])

    right_eye_max_h = Euclidian_distant(right_eye[0], right_eye[7])    
    right_eye_h = 1/3*(
                 Euclidian_distant(right_eye[3], right_eye[8])+
                 Euclidian_distant(right_eye[0], right_eye[7])+
                 Euclidian_distant(right_eye[4], right_eye[9]))
    right_eye_w = Euclidian_distant(right_eye[2], right_eye[6])

    hw_left = [left_eye_max_h, left_eye_h, left_eye_w]                                  
    hw_right = [right_eye_max_h, right_eye_h, right_eye_w]  
    return hw_left, hw_right

def const_faces(app,df,start_ind):
    for index, row in df.iterrows():
        start_time = time.time()
        if index < start_ind:
            continue
        file = row['file']
        confidence = row['confidence']
        
        if confidence >= 0:
            img = cv2.imread(file)
            faces = app.get(img)
            if len(faces)==1:
                for face in faces:
                    lmk = face.landmark_2d_106
                    lmk = np.round(lmk).astype(np.int)
                    hw_left, hw_right = eye_hw(lmk)

                    const_l = round(hw_left[1]/hw_left[2], 2)
                    const_r = round(hw_right[1]/hw_right[2], 2)

            else:
                const_l = -1
                const_r = -1
        else:
            const_l = -1
            const_r = -1
        df.loc[index, 'const_left'] = const_l
        df.loc[index, 'const_right'] = const_r
        if index == start_ind + 250000:
            df.to_csv('/home/yandex/igor/data_eyes/altyn_frames_last_1mln.csv', sep=';',index=False)
            break
            
        #const_l_list.append(const_l)
        #const_r_list.append(const_r)
        print( '{}: {} ms'.format(index, int( (time.time() - start_time)*1000  ) ) )
    #df['const_left'] = const_l_list
    #df['const_right'] = const_r_list
    return df

app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
app.prepare(ctx_id=0, det_size=(192, 192))
for i in range(5):
    start_ind = i*250000
    frames_df = pd.read_csv('/home/yandex/igor/data_eyes/altyn_frames_last_1mln.csv', sep=';')
    print(frames_df)
    frames_df = const_faces(app, frames_df, start_ind)
    frames_df.to_csv('/home/yandex/igor/data_eyes/altyn_frames_last_1mln.csv', sep=';',index=False)
    print('iteration: {} complete'.format(i))
