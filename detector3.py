import pandas as pd
import numpy as np
import os
import click
import time
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

#frames_df = pd.read_csv('/home/yandex/igor/data_eyes/csv/altyn_frames_podium_const_sep_norm.csv', sep=';')
#paths = tuple(frames_df['file'])
paths = ['/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0021.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0012.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0042.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0018.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0014.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0015.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0035.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0061.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0064.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0038.jpg']
@click.command()
@click.option('--source', default=paths, type=str)
@click.option('--destination', default='/mnt/data/frames/altyn_original_fps_igor_mediapipe_lndm_npy', type=str)
def fmain(source, destination):
    i = -1
    shift = len('/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112')
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5) as face_mesh:
        for filename in source:
            filename = '/mnt/data/frames/altyn_original_fps_igor' + filename[shift:]
            start_time = time.time()
            i += 1
            image = cv2.imread(filename)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                click.echo(f"--> {filename} was processed with error {e}")
                continue

            #print('facial_landmarks: ', facial_landmarks)
            np.save(file=os.path.join(destination, filename.split('/')[-2] + '_' + filename.split('/')[-1] + ".npy"), arr=[[i.x, i.y, i.z] for i in results.multi_face_landmarks[0].landmark])
            #click.echo(f"--> Processing {filename}.. SUCCESS")

            print('{}: {} ms'.format(i, int((time.time() - start_time)*1000 )))



if __name__ == '__main__':
    fmain()