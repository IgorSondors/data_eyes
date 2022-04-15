import pandas as pd
import cv2

i = 0
df_test = pd.read_csv('/home/sondors/Downloads/igor/qadev/smile_nn/eval_csv/train_2.csv', sep=';')
df_test_new = df_test.copy()
print(df_test_new['gt_class'].value_counts())

for index, row in df_test.iterrows():
    cls = row['cls']
    gt_class = row['gt_class']
    predict = row['predict']
    basename = row['basename']
    img = cv2.imread(basename, 3)
    title = 'gt: {}, cls: {}'.format(gt_class, cls)
    print(index)
    while(1):
        cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL)
        cv2.imshow(title,img)
        k = cv2.waitKey(0) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            break
        if k == ord('1'):
            df_test_new.loc[index, 'gt_class'] = 1
            print('chosen 1')
            cv2.destroyAllWindows()
            break
        if k == ord('2'):
            df_test_new.loc[index, 'gt_class'] = 2
            print(' chosen 2')
            cv2.destroyAllWindows()
            break
        if k == ord('0'):
            df_test_new.loc[index, 'gt_class'] = 0
            print('chosen 0')
            cv2.destroyAllWindows()
            break
#print(i)

#df_test_new.to_csv('/home/sondors/Downloads/igor/qadev/smile_nn/eval_csv/corrected/df_train_wrong2_corrected.csv', sep=';',index=False)