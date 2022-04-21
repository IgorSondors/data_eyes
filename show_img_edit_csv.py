import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from scipy import special
import pandas as pd
import numpy as np
import cv2

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
        lmk = -1
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


df_test = pd.read_csv('/home/yandex/igor/data_eyes/tags_57_71_const_genome_podium.csv', sep=';')
df_test_new = df_test.copy()
df_test_new['lndm_ok'] = [1 for i in range(len(df_test_new))]
for index, row in df_test.iterrows():
    file = row['filename']
    eyes_left, const_left, eyes_right, const_right, confidence = row['eyes_left'], row['const_left'], row['eyes_right'], row['const_right'], row['confidence']
    print(index)
    lmk = lmk_coord(file)
    if lmk != -1:
        left_eye = lmk[33:52]
        right_eye = lmk[87:]
        centre_nose = lmk[86]

        left_brow = left_eye[-9:]
        right_brow = right_eye[-9:]    
        left_eye = left_eye[:-9]
        right_eye = right_eye[:-9]

        img = draw_landm(file, left_eye, right_eye, centre_nose)
        title = 'eyes_left: {}, const_left: {}, eyes_right: {}, const_right: {}, confidence: {}'.format(eyes_left, const_left, eyes_right, const_right, confidence)
        while(1):
            cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL)
            cv2.imshow(title,img)
            k = cv2.waitKey(0) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                break
            if k == ord('1'):
                df_test_new.loc[index, 'lndm_ok'] = 1
                print('chosen 1')
                cv2.destroyAllWindows()
                break
            if k == ord('0'):
                df_test_new.loc[index, 'lndm_ok'] = 0
                print('chosen 0')
                cv2.destroyAllWindows()
                break
    else:
        df_test_new.loc[index, 'lndm_ok'] = 0

df_test_new.to_csv('/home/yandex/igor/data_eyes/tags_57_71_const_genome_podium_lndm_ok.csv', sep=';',index=False)