# -*- coding: utf_8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import cv2
import argparse
import pickle

import nsml
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from nsml import DATASET_PATH

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.vgg19 import VGG19
import tensorflow.keras.backend as K
from data_loader import train_data_loader
from pprint import pprint
from sklearn.metrics.pairwise import cosine_similarity

class TestDataset(data.Dataset):
    def __init__(self, img_arr):
        self.img_arr = img_arr

    def __getitem__(self, index):
        return self.img_arr[index]

    def __len__(self):
        return len(self.img_arr)
        
def infer_loader(img_arr, layer_fn):
    test_loader = torch.utils.data.DataLoader(dataset=TestDataset(img_arr), batch_size=128, num_workers=os.cpu_count())
    
    query_arr = None

    for test_data in test_loader:
        test_data = test_data.numpy()

        query_vecs = np_flatten(layer_fn([test_data])[0])
        
        if query_arr is None:
            query_arr = query_vecs
        else:
            query_arr = np.concatenate((query_arr, query_vecs))

    return query_arr

def np_flatten(x):
    return x.reshape(x.shape[0], -1)

def get_cos_sim_fn(qs, rs):

    sim_matrix = []

    for q in qs:
        sim_arr = []
        q = q.reshape(1, -1)

        for r in rs:
            r = r.reshape(1, -1)
            sim = cosine_similarity(q, r)
            sim_arr.append(sim)
        sim_matrix.append(sim_arr)

    return np_flatten(np.array(sim_matrix))


def l2_normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


# data preprocess
def preprocess(queries, db):
    query_img = []
    reference_img = []
    img_size = (224, 224)

    for img_path in queries:
        img = cv2.imread(img_path, 1)
        img = img[250:750, 250:750]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        query_img.append(img)

    for img_path in db:
        img = cv2.imread(img_path, 1)
        img = img[250:750, 250:750]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_size)
        reference_img.append(img)

    return queries, query_img, db, reference_img

def model_fn():
    model = Sequential()
    model_conv = VGG19(include_top=False, input_shape=(224, 224, 3))
    model.add(model_conv)

    return model


def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(file_path):
        pass

    def infer(queries, db):

        # Query 개수: 195
        # Reference(DB) 개수: 1,127
        # Total (query + reference): 1,322

        queries, query_img, references, reference_img = preprocess(queries, db)

        print('test data load queries {} query_img {} references {} reference_img {}'.
              format(len(queries), len(query_img), len(references), len(reference_img)))

        queries = np.asarray(queries)
        query_img = np.asarray(query_img)
        references = np.asarray(references)
        reference_img = np.asarray(reference_img)

        query_img = query_img.astype('float32')
        query_img /= 255
        reference_img = reference_img.astype('float32')
        reference_img /= 255

        get_feature_layer = K.function([model.input]+ [K.learning_phase()], [model.output])

        print('inference start')

        # inference
        query_vecs = infer_loader(query_img, get_feature_layer)

        # caching db output, db inference
        db_output = './db_infer.pkl'
        if os.path.exists(db_output):
            with open(db_output, 'rb') as f:
                reference_vecs = pickle.load(f)
        else:
            reference_vecs = infer_loader(reference_img, get_feature_layer)
            with open(db_output, 'wb') as f:
                pickle.dump(reference_vecs, f)

        # l2 normalization
        query_vecs = l2_normalize(query_vecs)
        reference_vecs = l2_normalize(reference_vecs)

        # Calculate cosine similarity
        sim_matrix = get_cos_sim_fn(query_vecs, reference_vecs)

        retrieval_results = {}

        for (i, query) in enumerate(queries):
            query = query.split('/')[-1].split('.')[0]
            sim_list = zip(references, sim_matrix[i].tolist())
            sorted_sim_list = sorted(sim_list, key=lambda x: x[1], reverse=True)

            ranked_list = [k.split('/')[-1].split('.')[0] for (k, v) in sorted_sim_list]  # ranked list

            retrieval_results[query] = ranked_list
        print('done')

        return list(zip(range(len(retrieval_results)), retrieval_results.items()))

    # DONOTCHANGE: They are reserved for nsml
    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epochs', type=int, default=100)
    args.add_argument('--batch_size', type=int, default=15)

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0', help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')
    config = args.parse_args()

    model = model_fn()
    bind_model(model)

    if config.pause:
        nsml.paused(scope=locals())

    if config.mode == 'train':
        nsml.save(0)
