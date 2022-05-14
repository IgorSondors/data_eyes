import tensorflow as tf
import pandas as pd
import numpy as np
import os
import click
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

frames_df = pd.read_csv('/home/yandex/igor/data_eyes/csv/altyn_frames_podium_const_sep_norm.csv', sep=';')
paths = tuple(frames_df['file'])
#paths = ['/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0021.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0012.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0042.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0018.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0014.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0015.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0035.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0061.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0064.jpg', '/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112/445493eb-d0fa-4670-b785-bb3d2e10082d/0038.jpg']
@click.command()
@click.option('--source', default=paths, type=str)
@click.option('--destination', default='/mnt/data/frames/altyn_original_fps_igor_lndm_npy', type=str)
#@click.option('--destination', default='/home/yandex/igor/data_eyes/npy_landmarks', type=str)
@click.option('--export_dir', default='/home/yandex/igor/sensor_models/spider/0002', type=str)
def fmain(source, destination, export_dir):
    #click.echo(f"--> Loading detector: {export_dir}")
    detector = tf.saved_model.load(export_dir)
    i = -1
    shift = len('/mnt/data/frames/altyn_original_fps_igor_origami_extended_0003_112')
    for filename in source:
        filename = '/mnt/data/frames/altyn_original_fps_igor' + filename[shift:]
        start_time = time.time()
        i += 1
        signature = "serving_default"
        with open(filename, "rb") as f:
            images = tf.convert_to_tensor([f.read()])

        try:
            output = detector.signatures[signature](images)
        except Exception as e:
            click.echo(f"--> {filename} was processed with error {e}")
            continue

        num_of_all_faces = output["num_of_all_faces"].numpy()[0]

        if num_of_all_faces == 0:
            print(f"--> Processing {filename}.. with 0 faces")
            continue

        facial_landmarks = output["facial_landmarks"].numpy()[0]
        #print('facial_landmarks: ', facial_landmarks)
        np.save(file=os.path.join(destination, filename.split('/')[-2] + '_' + filename.split('/')[-1] + ".npy"), arr=facial_landmarks)
        #click.echo(f"--> Processing {filename}.. SUCCESS")

        print('{}: {} ms'.format(i, int((time.time() - start_time)*1000 )))



if __name__ == '__main__':
    fmain()