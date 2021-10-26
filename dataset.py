#!/usr/bin/env python
# -*- coding:utf-8 -*- 


#from preprocess import tf_record_utils
import tensorflow as tf
import os
import numpy as np
from my_decorator import print_time
import time
import pickle

class Dataset(object):

    def __init__(self,record_path, epoches=100):
        self.data_path = record_path
        self.epoches = epoches
        self.train_file_record = "train_record_add_feature.record"
        # self.train_file_record = "all_train_record_add_feature_case.record"
        self.train_file_list = 'train_list_add_feature.pickle'
        self.test_file_list = 'test_list_add_feature.pickle'
        self.valid_file_list = 'valid_list_add_feature.pickle'


        with open(os.path.join(self.data_path,self.test_file_list), 'rb') as f:
            self.test_data = pickle.load(f)
        with open(os.path.join(self.data_path,self.valid_file_list), 'rb') as f:
            self.valid_data = pickle.load(f)




    def batches(self,batch_size):
        train_data_path = os.path.join(self.data_path, self.train_file_record)
        self.train_data = self.read_tfRecord(train_data_path, batch_size, self.epoches)
        return self.train_data.make_one_shot_iterator().get_next()


    def load_test_set(self):
        data_path = os.path.join(self.data_path, self.test_file_list)
        data = self.load_data_list(data_path)
        return data

    def load_valid_set(self):
        data_path = os.path.join(self.data_path, self.valid_file_list)
        data = self.load_data_list(data_path)
        return data

    def load_test_valid_set(self):
        test_data = self.load_test_set()
        valid_data = self.load_valid_set()
        test_valid_data = dict()
        for key in test_data.keys():
            if key != "batch_size":
                test_valid_data[key] = np.concatenate([test_data[key], valid_data[key]],axis=0)

        test_valid_data['batch_size'] = test_data["batch_size"] + valid_data['batch_size']
        #test_valid_data["url_mask_idx"] = np.reshape(test_valid_data["url_mask_idx"], [-1, 1])
        test_valid_data["text"] = np.reshape(test_valid_data["text"], [-1, 1, 512])

        #test_valid_data["text_mask_idx"] = np.reshape(test_valid_data["text_mask_idx"], [-1, 1])
        #test_valid_data["image_mask_idx"] = np.reshape(test_valid_data["image_mask_idx"], [-1, 1])
        test_valid_data["label"] = np.reshape(test_valid_data["label"], [-1, 1])
        return test_valid_data

    def load_train_set(self):
        data_path = os.path.join(self.data_path,self.train_file_list)
        data = self.load_data_list(data_path)
        data["label"] = data["label"].reshape([-1,1])
        return data

    def load_data_by_idx(self,data,idx):
        result = None
        # if "test" in data_type:
        #     data = self.test_data
        # if "valid" in data_type:
        #     data = self.valid_data
        if data is not None:
            result = dict()
            sample = data[idx]
            image_array = self.process_image(sample['image'])
            result['st'] = np.reshape(sample['st'],[1,-1])
            result['url'] = np.reshape(sample['url'],[1,-1])
            result['url_mask_idx'] = np.array([sample['url_len']])
            result['text'] = np.reshape(sample['text'],[1,1,-1])
            result['sentence_mask_idx'] = np.reshape(sample['sentence_len'],[-1,1])
            result['text_mask_idx'] = np.array([sample['text_len']])
            result['image'] = np.reshape(image_array,[1,10,-1])
            result['image_mask_idx'] = np.array([sample['image_len']])
            result['label'] = np.reshape(np.array([sample['label']]), [-1,1])
            result['batch_size'] = 1

        return result

    def load_data_by_year(self,path):
        data_path = os.path.join(self.data_path,path)
        data = self.load_data_list(data_path)
        return data


    def load_data_list(self,path, max_image_len=10):
        result = dict()
        st_list = []
        url_list = []
        url_len_list = []
        text_list = []
        text_len_list = []
        sentence_len_list = []
        image_list = []
        image_len_list = []
        label_list = []
        with open(path, 'rb') as f:
            data_list = pickle.load(f)
            if "test" in path:
                self.test_data = data_list
            if "valid" in path:
                self.valid_data = data_list
        data_size = len(data_list)
        for data in data_list:
            # padding image
            image_array = self.process_image(data['image'])
            st_list.append(data['st'])
            url_list.append(data['url'])
            url_len_list.append(data['url_len'])
            text_list.append(data['text'])
            text_len_list.append(data['text_len'])
            sentence_len_list.append(data['sentence_len'])
            image_list.append(image_array)
            image_len_list.append(data['image_len'])
            label_list.append(data['label'])

        result['st'] = np.stack(st_list, axis=0)

        # print(data['st'].shape)
        # print(data['st'])
        result['url'] = np.stack(url_list, axis=0)
        result['url_mask_idx'] = np.stack(np.array(url_len_list), axis=0)
        result['text'] = np.expand_dims(np.stack(text_list, axis=0),axis=1)

        result['sentence_mask_idx'] = np.stack(sentence_len_list, axis=0)
        result['text_mask_idx'] = np.stack(np.array(text_len_list), axis=0)
        result['image'] = np.stack(image_list, axis=0)
        result['image_mask_idx'] = np.stack(np.array(image_len_list), axis=0)
        result['label'] = np.expand_dims(np.stack(np.array(label_list), axis=0),-1)
        result['batch_size'] = data_size
        return result

    def process_image(self,image,max_image_len=10):
        image_array = image
        image_num = image_array.shape[0]
        image_num_to_padding = max_image_len - image_num
        if image_num_to_padding != 0:
            image_to_padding = np.zeros(shape=[image_num_to_padding, 2048])
            image_array = np.concatenate([image_array, image_to_padding], axis=0)
        return image_array


    def get_next(self,sess,train_data):
        data = {}
        try:
            batches = sess.run(train_data)            # TODO: 把这个转化成字典
            data['st'] = batches[0]

            # print(data['st'].shape)
            # print(data['st'])
            data['url'] = batches[1]
            data['url_mask_idx'] = batches[2]
            data['text'] = batches[3]
            data['sentence_mask_idx'] = batches[4]
            data['text_mask_idx'] = batches[5]
            data['image'] = batches[6]
            data['image_mask_idx'] = batches[7]
            data['label'] = np.reshape(batches[8],[-1,1])
            data['batch_size'] = data['label'].shape[0]
        except  Exception as e:
            data = None
        return data

    def parser(self,record):

        features = {
            'st': tf.FixedLenFeature((), default_value='', dtype=tf.string),
            'url': tf.FixedLenFeature((), default_value="", dtype=tf.string),
            'url_len': tf.FixedLenFeature((), default_value=0, dtype=tf.int64),
            'sentence_len': tf.FixedLenFeature((), default_value='', dtype=tf.string),
            'text_mask_len': tf.FixedLenFeature((), default_value=0, dtype=tf.int64),
            'text_max_len': tf.FixedLenFeature((), default_value=0, dtype=tf.int64),
            'text_len': tf.FixedLenFeature((), default_value=0, dtype=tf.int64),
            'text': tf.FixedLenFeature((), default_value='', dtype=tf.string),
            'image_mask_len': tf.FixedLenFeature((), default_value=0, dtype=tf.int64),
            'image_max_len': tf.FixedLenFeature((), default_value=0, dtype=tf.int64),
            'image_len': tf.FixedLenFeature((), default_value=0, dtype=tf.int64),
            'image': tf.FixedLenFeature((), default_value='', dtype=tf.string),
            'label': tf.FixedLenFeature((), default_value=0, dtype=tf.int64)
        }

        example = tf.io.parse_single_example(record, features)
        sentence_len = tf.reshape(tf.decode_raw(example['sentence_len'], tf.int64), [-1])
        print(sentence_len)
        text_mask_len = example['text_mask_len']
        text_max_len = example['text_max_len']
        text_len = example['text_len']
        image_len = example['image_len']
        image_mask_len = example['image_mask_len']
        image_max_len = example['image_max_len']

        st = tf.reshape(tf.decode_raw(example['st'], tf.float64), [16], name="pip_reshape_st")    # 对应numpy float64？
        url = tf.reshape(tf.decode_raw(example['url'], tf.int64), [200], name="pip_reshap_url")
        # url_mask = tf.reshape(tf.decode_raw(example['url_mask'],tf.int64),[200],name="pip_reshape_url_mask")
        url_mask_idx = example['url_len']

        text_ = tf.reshape(tf.decode_raw(example['text'], tf.int64), [text_len, 512], name="pip_reshape_text")
        # text_add = tf.constant(np.zeros(shape=[text_mask_len,512]),name='pip_text_add')
        text_add = tf.zeros(shape=[text_mask_len, 512], name='pip_text_add', dtype=tf.int64)
        text = tf.concat([text_, text_add], axis=0)

        sentence_mask_idx = tf.concat([sentence_len, tf.zeros([text_mask_len], dtype=tf.int64)], axis=0)  # 句子级别的实际长度
        text_mask_idx = text_len

        image_ = tf.reshape(tf.decode_raw(example['image'], tf.float32), [image_len, 2048], name="pip_reshape_image")
        image_add = tf.zeros(shape=[image_mask_len, 2048], dtype=tf.float32, name='pip_mask_image')
        image = tf.concat([image_, image_add], axis=0)

        image_mask_idx = image_len

        label = example['label']

        return st, url, url_mask_idx, text, sentence_mask_idx, text_mask_idx, image, image_mask_idx, label

    def read_tfRecord(self,record_path, batch_size, epoches):
        # dataset = tf.data.TFRecordDataset(record_path).interleave(tf.data.TFRecordDataset,cycle_length=10)
        dataset = tf.data.TFRecordDataset(record_path)
        # dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=lambda x:parser(x),num_parallel_calls=64, batch_size=batch_size))
        dataset = dataset.map(lambda x: self.parser(x), num_parallel_calls=64)
        dataset = dataset.shuffle(buffer_size=20000).batch(batch_size).repeat(epoches).prefetch(5000)
        # dataset = dataset.repeat(epoches).prefetch(1000)
        return dataset


def model():
    dataset = Dataset('data/train_data.record', 100)
    next_batches = dataset.batches(500)
    count = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        while True:
            try:
                # next = sess.run(next_batches)
                # print(type(next))
                start_time = time.time()
                next = dataset.get_next(sess,next_batches)
                end_time = time.time()
                print(list(next.keys()),end_time-start_time)
                time.sleep(3)
                count += 1
            except tf.errors.OutOfRangeError:
                print("训练结束")
                print("数据量：\t{}".format(count))
                break

        # for next_batch in next_batches:
        #     print(sess.run(next_batch))
        #     break


if __name__ == "__main__":
    model()
    # dataset = Dataset('data/train_data.record',100)
    # next_batch = dataset.batches(1)
    # count= 0
    #
    # with tf.Session() as sess:
    #      sess.run(tf.global_variables_initializer())
    #      while True:
    #          try:
    #             next = sess.run(next_batch)
    #             count+=1
    #          except tf.errors.OutOfRangeError:
    #              print("训练结束")
    #              print("数据量：\t{}".format(count))
    #              break





