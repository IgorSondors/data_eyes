import tensorflow as tf
import pandas as pd
import threading
import click
import glob
import time
import os
import gc

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


class UnitThread(threading.Thread):
    def __init__(self, threadID, filenames, detector):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.filenames = filenames
        self.len_filenames = len(self.filenames)
        self.detector = detector


    def run(self):
        print(f"--> Starting thread #{self.threadID}..")
        df_test = pd.read_csv('/home/yandex/igor/data_eyes/tags_57_71_const_genome.csv', sep=';')
        blink_list = []
        sharpness_list = []
        up_down_list = []
        left_right_list = []
        confidence_list = []
        for index, row in df_test.iterrows():
            filename = row['filename']
            with open(filename, "rb") as f:
                image_bytes = f.read()
                output = self.detector.signatures["serving_default"](tf.convert_to_tensor([image_bytes], tf.string))
                blink = output['blink'].numpy()
                sharpness = output['sharpness'].numpy()
                up_down = output['up_down'].numpy()
                left_right = output['left_right'].numpy()
                confidence = output['confidence'].numpy()

                blink_list.append(blink)
                sharpness_list.append(sharpness)
                up_down_list.append(up_down)
                left_right_list.append(left_right)
                confidence_list.append(confidence)
                #print(output['blink'])
                #print(pd.Series(output))

        df_test['blink'] = blink_list
        df_test['sharpness'] = sharpness_list
        df_test['up_down'] = up_down_list
        df_test['left_right'] = left_right_list
        df_test['confidence'] = confidence_list
        df_test.to_csv('/home/yandex/igor/data_eyes/tags_57_71_const_genome_podium.csv', sep=';', index=False)            
        print(f"--> Finishing thread #{self.threadID}..")

@click.command()
@click.option('--source', default='/mnt/data/frames/new_test/frames_1_origami_extended_0003_112', type=str)
@click.option('--export_dir', default='/home/yandex/igor/sensordev/sensordev_runner/0003rules', type=str)
@click.option('--num_threads', default=1, type=int)
def fmain(source, export_dir, num_threads):
    click.echo("--> Init session config..")
    click.echo(f"--> Loading detector: {export_dir}")
    detector = tf.saved_model.load(export_dir)

    filenames = glob.glob(os.path.join(source, '**', "*.*g"), recursive=True)
    len_filenames = len(filenames)
    thread_len = int(len_filenames / num_threads)

    threads = []
    i_thread = 0
    for b in range(0, len_filenames, thread_len):
        thread_filenames = filenames[b:b+thread_len]
        threads.append(UnitThread(threadID=i_thread, filenames=thread_filenames, detector=detector))
        i_thread = i_thread + 1

    for thread in threads:
        thread.start()
    
if __name__ == '__main__':
    fmain()

    