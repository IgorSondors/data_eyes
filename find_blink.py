import matplotlib.pyplot as plt
import pandas as pd

def select_data1(df_test_new, normalize = False, blink_zero=False, two_faces=True, podium=True):
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

    if two_faces:
        df_test_new = df_test_new[df_test_new.const_left!=-1]
        df_test_new = df_test_new[df_test_new.const_right!=-1]

        df_test_new = df_test_new[df_test_new.const_left!=float('inf')]
        df_test_new = df_test_new[df_test_new.const_right!=float('inf')]


        #df_test_new = df_test_new[df_test_new.const_left<= 0.31]#0.4]
        #df_test_new = df_test_new[df_test_new.const_right<= 0.31]#0.4]

    if podium:
        df_test_new = df_test_new[df_test_new.light_exposure>0.15]
        df_test_new = df_test_new[df_test_new.light_exposure<0.9]
        df_test_new = df_test_new[df_test_new.light_background<0.95]
        df_test_new = df_test_new[df_test_new.light_lighting<0.95]
        df_test_new = df_test_new[df_test_new.object<0.9]
        df_test_new = df_test_new[df_test_new.face_brightness>0.3]
        df_test_new = df_test_new[df_test_new.sharpness>0.625]#0.625]
        
        df_test_new = df_test_new[abs(df_test_new.left_right)<0.3]
        df_test_new = df_test_new[abs(df_test_new.up_down)<0.3]

    if blink_zero:
        for index, row in df_test_new.iterrows():
            blink = row['blink']
            #const_left = row['const_left']
            #const_right = row['const_right']
            if blink > 0.95:
                df_test_new.loc[index, 'const_left'] = 0
                df_test_new.loc[index, 'const_right'] = 0

    if normalize:
        df_test_new['const_left']=df_test_new['const_left']-min(df_test_new.const_left)
        df_test_new['const_right']=df_test_new['const_right']-min(df_test_new.const_right)

        df_test_new['const_left']=(df_test_new['const_left']/max(df_test_new.const_left)*9).astype('int')
        df_test_new['const_right']=(df_test_new['const_right']/max(df_test_new.const_right)*9).astype('int')
    
    return df_test_new

def append_ext(fn):
    return fn.split('/')[-2]

df_test = pd.read_csv('/home/yandex/igor/data_eyes/altyn_video_frames_orig_fps_crops_podium_const_all.csv', sep=';')
df_test = select_data1(df_test)
df_test['person_id'] = df_test['file'].apply(append_ext)

print(df_test['person_id'].value_counts())

#Просто делаем нормализацию констант для каждой группы картинок person_id по отдельности
person_id_previous=''
index_list = []
const_left_list = []
const_right_list = []
df_test_new = df_test#.copy()
for index, row in df_test_new.iterrows():
    print(index)
    person_id = row['person_id']
    file = row['file']
    blink = row['blink']
    const_left = row['const_left']
    const_right = row['const_right']

    if person_id != person_id_previous:
        for i in range(len(const_left_list)):
            df_test_new.loc[index_list[i], 'const_left'] = const_left_list[i]/(max(const_left_list)+0.000000000001)
            df_test_new.loc[index_list[i], 'const_right'] = const_right_list[i]/(max(const_right_list)+0.000000000001)

        index_list = []
        const_left_list = []
        const_right_list = []
        person_id_previous = person_id
        
    const_left_list.append(const_left)
    const_right_list.append(const_right)
    index_list.append(index)

df_test_new.to_csv('/home/yandex/igor/data_eyes/altyn_frames_podium_const_sep_norm.csv',  sep=';',index=False)