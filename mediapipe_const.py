import pandas as pd
import os
import numpy as np

def eye_hw(data):
    def Euclidian_distant(xy1, xy2):
        return ((xy1[0]-xy2[0])**2+(xy1[1]-xy2[1])**2)**0.5

    rightEyeUpper0 = [246, 161, 160, 159, 158, 157, 173]
    rightEyeLower0 = [33, 7, 163, 144, 145, 153, 154, 155, 133]

    leftEyeUpper0 = [466, 388, 387, 386, 385, 384, 398]
    leftEyeLower0 = [263, 249, 390, 373, 374, 380, 381, 382, 362]

    left_eye = []
    right_eye = []
    left_eye_down = []
    right_eye_down = []
    centre_nose = data[1]

    for i in leftEyeUpper0:
        left_eye.append(data[i][:2])
    for i in rightEyeUpper0:
        right_eye.append(data[i][:2])
    for i in leftEyeLower0:
        left_eye_down.append(data[i][:2])
    for i in rightEyeLower0:
        right_eye_down.append(data[i][:2])

    left_eye_max_h = Euclidian_distant(left_eye[3], left_eye_down[4])
    left_eye_h = 1/7*(
                 Euclidian_distant(left_eye[0], left_eye_down[1])+
                 Euclidian_distant(left_eye[1], left_eye_down[2])+
                 Euclidian_distant(left_eye[2], left_eye_down[3])+
                 Euclidian_distant(left_eye[3], left_eye_down[4])+
                 Euclidian_distant(left_eye[4], left_eye_down[5])+
                 Euclidian_distant(left_eye[5], left_eye_down[6])+
                 Euclidian_distant(left_eye[6], left_eye_down[7]))
    left_eye_w = Euclidian_distant(left_eye_down[0], left_eye_down[8])

    right_eye_max_h = Euclidian_distant(right_eye[3], right_eye_down[4])    
    right_eye_h = 1/7*(
                 Euclidian_distant(right_eye[0], right_eye_down[1])+
                 Euclidian_distant(right_eye[1], right_eye_down[2])+
                 Euclidian_distant(right_eye[2], right_eye_down[3])+
                 Euclidian_distant(right_eye[3], right_eye_down[4])+
                 Euclidian_distant(right_eye[4], right_eye_down[5])+
                 Euclidian_distant(right_eye[5], right_eye_down[6])+
                 Euclidian_distant(right_eye[6], right_eye_down[7]))
    right_eye_w = Euclidian_distant(right_eye_down[0], right_eye_down[8])

    hw_left = [left_eye_max_h, left_eye_h, left_eye_w]                                  
    hw_right = [right_eye_max_h, right_eye_h, right_eye_w]  
    return hw_left, hw_right

def const_faces_mediapipe(df):
    shift = len('/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112')
    src = '/mnt/data/frames/altyn_original_fps_igor_mediapipe_lndm_npy'
    for index, row in df.iterrows():
        file = row['file']
        npy_file=os.path.join(src, file.split('/')[-2] + '_' + file.split('/')[-1] + ".npy")
        print(index)
        try:
            data = np.load(npy_file)
                    
            hw_left, hw_right = eye_hw(data)

            const_l = round(hw_left[1]/hw_left[2], 3)
            const_r = round(hw_right[1]/hw_right[2], 3)

            df.loc[index, 'const_left3'] = const_l
            df.loc[index, 'const_right3'] = const_r
        except:
            print('npy not found')
    return df

print('load csv')
df_test = pd.read_csv('/home/yandex/igor/data_eyes/csv/altyn_frames_podium_const_sep_norm.csv', sep=';')
print('csv loaded')
df_mediapipe = const_faces_mediapipe(df_test)

df_mediapipe.to_csv('/home/yandex/igor/data_eyes/altyn_frames_mediapipe_const.csv',  sep=';',index=False)