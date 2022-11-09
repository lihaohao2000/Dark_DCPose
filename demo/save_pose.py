import os
import pickle

def save_pose(pickle_file, train_x, train_y, test_x, test_y):
    if not os.path.isfile(pickle_file):    #判断是否存在此文件，若无则存储
        print('Saving data to pickle file...')
        try:
            with open(pickle_file, 'wb') as pfile:
                pickle.dump(
                    {
                        'train_x': train_x,
                        'train_y': train_y,
                        'test_x': test_x,
                        'test_y': test_y,
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise
        print('Data cached in pickle file.')


def load_pose(pickle_file):
    with open(pickle_file, 'rb') as f:
        pickle_data = pickle.load(f)       # 反序列化，与pickle.dump相反
        train_x = pickle_data['train_x']
        train_y = pickle_data['train_y']
        test_x = pickle_data['test_x']
        test_y = pickle_data['test_y']
        del pickle_data  # 释放内存
        print('Data and modules loaded.')
        return train_x, train_y, test_x, test_y