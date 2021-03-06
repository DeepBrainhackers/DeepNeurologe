from glob import glob
import os
import os.path as osp
import numpy as np
import pandas as pd
import nibabel as nib
from deepNeurologe_keras import init_network



def get_network(n_classes=2):
    return init_network(n_classes=n_classes)


def load_labels(data_folder):
    return pd.read_csv(osp.join(data_folder, 'meta.csv')).DX.values


def atleast_5d(arr):
    if len(arr.shape) != 5:
        arr = arr[..., np.newaxis]
    return arr


def load_data(data_folder, postfix='6mm'):
    return atleast_5d(np.load(osp.join(data_folder, 'data_{}.npy'.format(postfix))))



def get_data(parent_folder, tag, file_name):
    return np.array(sorted(glob(osp.join(parent_folder, tag, file_name))))


def extract_subj_folder(data_path):
    """
    Assumes that the last element in the array data_path is the filename (and the folder is the element before that)
    :param data_path:
    :return:
    """

    folder_names = np.char.asarray(np.char.split(data_path, '/'))[:, -2]
    folder_names = np.char.asarray(np.char.split(folder_names, '_'))
    dx = folder_names[:, 0]
    gender = folder_names[:, 1]
    subj_ids = np.array(folder_names[:, 2], dtype=np.int)
    return dx, subj_ids, gender


def load_subj(subj_file):
    return np.array(nib.load(subj_file).get_data(), dtype=np.float32)


def run(parent_folder, tags, file_name, save_folder, postfix=''):
    X_all, dx_all, ids_all, gender_all = [], [], [], []

    if not osp.exists(save_folder):
        os.makedirs(save_folder)

    for tag in tags:
        print tag
        X, dx_tag, subj_ids_tag, gender_tag = prepare_data_tag(file_name, parent_folder, tag)
        X_all.append(X)
        dx_all.append(dx_tag)
        ids_all.append(subj_ids_tag)
        gender_all.append(gender_tag)
    X = np.concatenate(X_all, axis=0)
    dx = np.concatenate(dx_all, axis=0)
    ids_all = np.concatenate(ids_all, axis=0)
    gender_all = np.concatenate(gender_all, axis=0)

    df = pd.DataFrame(data={'subj_ids': ids_all, 'DX_str': dx, 'gender': gender_all})
    df['DX'] = pd.get_dummies(df.DX_str)['ad'].values
    np.save(osp.join(save_folder, 'data_{}.npy'.format(postfix)), X)
    df.to_csv(osp.join(save_folder, 'meta.csv'), index=False)


def prepare_data_tag(file_name, parent_folder, tag):
    data_tag = get_data(parent_folder, tag, file_name)
    X = np.zeros((len(data_tag),) + nib.load(data_tag[0]).shape)
    dx_tag, subj_ids_tag, gender_tag = extract_subj_folder(data_tag)
    for i_subj in xrange(len(data_tag)):
        print '{}/{}'.format(i_subj + 1, len(data_tag))
        X[i_subj] = load_subj(data_tag[i_subj])
    return X, dx_tag, subj_ids_tag, gender_tag


if __name__ == '__main__':
    parent_folder = '/home/rthomas/Neuro_VUmc/asl_dementie'
    tags = ['ad*', 'smc*']
    file_name = 'asl2std_6mm.nii.gz'
    save_folder = '/home/paulgpu/git/DeepNeurologe'
    postfix = '6mm'
    run(parent_folder, tags, file_name, save_folder, postfix=postfix)
