import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

import pandas as pd
import numpy as np
import click
import time
import cv2
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

frames_df = pd.read_csv('/home/yandex/igor/data_eyes/csv/altyn_frames_podium_const_sep_norm.csv', sep=';')
paths = tuple(frames_df['file'])
#paths = ['/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0021.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0012.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0042.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0018.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0014.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0015.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0035.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0061.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0064.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0038.jpg']
@click.command()
@click.option('--source', default=paths, type=str)
@click.option('--destination', default='/mnt/data/frames/altyn_original_fps_igor_insightface_lndm_npy', type=str)
def fmain(source, destination):
    i = -1
    shift = len('/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112')
    app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    for filename in source:
        filename = '/mnt/data/frames/altyn_original_fps_igor' + filename[shift:]
        
        start_time = time.time()
        i += 1
        image = cv2.imread(filename)
        faces = app.get(image)
        for face in faces:
            lmk = face.landmark_2d_106
            lmk = np.round(lmk).astype(np.int)
        #print('facial_landmarks: ', facial_landmarks)
        np.save(file=os.path.join(destination, filename.split('/')[-2] + '_' + filename.split('/')[-1] + ".npy"), arr=lmk)
        #click.echo(f"--> Processing {filename}.. SUCCESS")

        print('{}: {} ms'.format(i, int((time.time() - start_time)*1000 )))

if __name__ == '__main__':
    fmain()