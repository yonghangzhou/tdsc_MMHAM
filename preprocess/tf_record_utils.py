#!/usr/bin/env python
# -*- coding:utf-8 -*- 



import tensorflow as tf
import numpy as np
import pickle
import json
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential

pretrained_imagenet = ResNet50(weights="imagenet")
imagenet_part = tf.keras.models.Model(inputs=pretrained_imagenet.input,outputs=pretrained_imagenet.get_layer('avg_pool').output)

def generate_test_sample(num):

    """generate test sample

    :param num: num of test sample need for test
    :return sample list: a list contains sample dict

    [{"st": np.array([32])
     "url":np.array([len(vovabulary])
     "url_mask":
     "text":[np.array(vocabulary)]
     "text_mask":
     "image":[np.array(3*128*128)]
     "label":0 or 1
    },]
    """
    text_max_len = 0
    image_max_len = 0
    sample_list = []
    for i in range(num):
        st = np.random.randint(0,100,32)

        url_len = np.random.randint(20,100)
        url_ = np.random.randint(0,200,url_len)
        url = _padding(url_,100,0)


        text_len = np.random.randint(2,10)        #一个网页的文本数量

        if text_len > text_max_len:
            text_max_len  = text_len
        text_ = np.array([np.random.randint(0,3000,np.random.randint(10,512)) for j in range(text_len)])
        sentence_len = np.array([a.shape[0] for a in text_ ]  ) #每个text实际的长度
        text = np.array([_padding(t,512,0) for t in text_ ])


        image_len = np.random.randint(2,10)
        if image_len > image_max_len:
            image_max_len = image_len
        image = np.array([np.random.randint(0,255,[128,128,3]) for j in range(image_len)])

        sample = {
            "st":st,
            "url":url,
            "url_len":url_len,
            "text":text,
            "text_len":text_len,
            "sentence_len":sentence_len,
            "image":image,
            "image_len":image_len,
            "label":np.random.randint(0,2)
        }

        sample_list.append(sample)
    return sample_list,text_max_len,image_max_len


def url_feature(url, url_max_len, url_word_idx):
    tmp_ = url.split('?')[0]
    tmp = [url_word_idx[w] for w in tmp_]

    if len(tmp) > url_max_len:
        tmp = tmp[:url_max_len]
    url_len = len(tmp)
    result = _padding(tmp, url_max_len, 0)
    return url_len, result


def text_feature(text, text_max_len, word_idx):
    text_len = 1
    tmp = text
    if len(tmp) > text_max_len:
        tmp = text[:text_max_len]
    sentence_len = len(tmp)
    text_idxs = [word_idx[word] if word_idx.get(word) else word_idx["UNK"] for word in tmp]
    result = _padding(text_idxs, text_max_len, 0)
    return text_len, np.array([sentence_len]), result



def image_feature(image_file_list):
    result = []
    image_len = 0
    image_file_path = []

    for imge_file in image_file_list:
        try:
            img = Image.open(os.path.join("../test", imge_file)).convert("RGB")
            img_reshape = img.resize([224, 224])
            image_np = np.array(img_reshape) / 255.0
            result.append(image_np)
            image_len += 1
            image_file_path.append(os.path.join("../test", imge_file))

        except Exception as e:
            print("处理图片数据出错！", e)
    x = np.stack(result, axis=0)
    print(x.shape)
    image_resnet_output = imagenet_part.predict(np.stack(result, axis=0))
    return image_file_path, image_len, image_resnet_output


def sample_data(data, url, url_max_len, text_max_len, url_word_idx, word_idx, label):
    sample = None
    #     try:
    print(label, url)
    data_ = data[url]
    st = np.array(data_["url_feature_norm"])
    url_len, url_ = url_feature(url, url_max_len, url_word_idx)
    text_len, sentence_len, text = text_feature(data_["text"], text_max_len, word_idx)
    image_file_path,image_len, image = image_feature(data_["image_list"])
    sample = {
        "st": st,
        "url": url_,
        "url_len": url_len,
        "text": text,
        "text_len": text_len,
        "sentence_len": sentence_len,
        "image": image,
        "image_len": image_len,
        "image_files":image_file_path,
        "label": label
    }
    #     except Exception as e:
    #         print("生成数据点发生错误 url：\t{}".format(url),e)
    return sample


def generate_sample_list(data_split=[8,1,1]):
    text_max_len = 512
    text_size = 88
    url_max_len = 200
    url_size = 88
    st_feature_size = 6
    images_size = 224
    word_idx = pickle.load(open("../data/word_idx.pickle", 'rb'))
    url_word_idx = pickle.load(open("../data/url_word_idx.pickle", 'rb'))

    positive_data = pickle.load(open("../data/positive_data_add_feature.pickle", "rb"))
    negative_data = pickle.load(open("../data/negative_data_add_feature.pickle", 'rb'))
    train_list = []
    test_list = []
    valid_list = []

    positive_data_urls = list(positive_data.keys())
    negative_data_urls = list(negative_data.keys())
    total_data_num = 2 * len(positive_data_urls)
    np.random.shuffle(negative_data_urls)
    count = 0
    for i in range(len(positive_data_urls)):
        positive_data_ = sample_data(positive_data, positive_data_urls[i], url_max_len, text_max_len, url_word_idx,
                                     word_idx, 0)
        negative_data_ = sample_data(negative_data, negative_data_urls[i], url_max_len, text_max_len, url_word_idx,
                                     word_idx, 1)


        if positive_data_ is not None:
            count += 1
            if count < total_data_num * 0.8:
                train_list.append(positive_data_)
            elif count < total_data_num * 0.9:
                test_list.append(positive_data_)
            else:
                valid_list.append(positive_data_)
        if negative_data_ is not None:
            count += 1
            if count < total_data_num * 0.8:
                train_list.append(negative_data_)
            elif count < total_data_num * 0.9:
                test_list.append(negative_data_)
            else:
                valid_list.append(negative_data_)
    return train_list,test_list,valid_list

def generate_all_data_example():
    text_max_len = 512
    text_size = 88
    url_max_len = 200
    url_size = 88
    st_feature_size = 6
    images_size = 224
    word_idx = pickle.load(open("../data/word_idx.pickle", 'rb'))
    url_word_idx = pickle.load(open("../data/url_word_idx.pickle", 'rb'))

    positive_data = pickle.load(open("../data/positive_data_add_feature.pickle", "rb"))
    negative_data = pickle.load(open("../data/negative_data_add_feature.pickle", 'rb'))
    negative_data_2020 = pickle.load(open("../data/negative_data_200_add_feature.pickle",'rb'))
    negative_data_2020_case = pickle.load(open("../data/negative_data_200_case.pickle","rb"))
    train_list = []


    positive_data_urls = list(positive_data.keys())
    negative_data_urls = list(negative_data.keys())
    negative_data_2020_urls = list(negative_data_2020.keys())
    negative_data_2020_case_urls = list(negative_data_2020_case.keys())

    np.random.shuffle(negative_data_urls)
    count = 0
    for i in range(len(positive_data_urls)):
        positive_data_ = sample_data(positive_data, positive_data_urls[i], url_max_len, text_max_len, url_word_idx,
                                     word_idx, 0)
        negative_data_ = sample_data(negative_data, negative_data_urls[i], url_max_len, text_max_len, url_word_idx,
                                     word_idx, 1)


        if positive_data_ is not None:
            count += 1
            train_list.append(positive_data_)

        if negative_data_ is not None:
            count += 1
            train_list.append(negative_data_)

    for i in range(len(list(negative_data_2020.keys()))):
        negative_data_ = sample_data(negative_data_2020, negative_data_2020_urls[i], url_max_len, text_max_len, url_word_idx,
                                     word_idx, 1)
        if negative_data_ is not None:
            count += 1
            train_list.append(negative_data_)

    for i in range(len(list(negative_data_2020_case.keys()))):
        negative_data_ = sample_data(negative_data_2020_case, negative_data_2020_case_urls[i], url_max_len, text_max_len, url_word_idx,
                                     word_idx, 1)
        if negative_data_ is not None:
            count += 1
            train_list.append(negative_data_)


    return train_list



def load_sample_list(path):
    text_max_len = 512
    text_size = 88
    url_max_len = 200
    url_size = 88
    st_feature_size = 6
    images_size = 224
    word_idx = pickle.load(open("../data/word_idx.pickle", 'rb'))
    url_word_idx = pickle.load(open("../data/url_word_idx.pickle", 'rb'))

    data = pickle.load(open(path, "rb"))

    data_list = []
    # test_list = []
    # valid_list = []

    data_urls = list(data.keys())

    for i in range(len(data_urls)):
        data_ = sample_data(data, data_urls[i], url_max_len, text_max_len, url_word_idx, word_idx, 0)

        data_list.append(data_)
    return data_list



def _masking(data,length,mask):
    """
    padding array
    :param data:
    :param length:
    :param pad:
    :return:
    """
    result = np.ones(length)
    tmp = list(data)
    if len(tmp) < length:
        result[len(tmp):] = mask
    return result

def _padding(data,length,pad):

    result = list(data)
    if len(result) < length:
        tmp = [pad] * (length-len(result))
        result += tmp
    return np.array(result)





def int64_feature(value_):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value_]))

def Byte_feature(value_):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value_.tostring()]))

def generate_tf_example(data,text_max_len,image_max_len):

    st = data['st']
    url = data['url']
    url_len = data['url_len']

    text = data['text']
    sentence_len = data['sentence_len']
    text_len = data['text_len']
    image = data['image']
    image_len = data['image_len']
    label = data['label']

    image_len = image.shape[0]
    tf_example = tf.train.Example(
        features = tf.train.Features(
            feature = {

                'st': Byte_feature(st),
                'url': Byte_feature(url),
                'url_len': int64_feature(url_len),
                'sentence_len':Byte_feature(sentence_len),
                'text_max_len':int64_feature(text_max_len),
                'text_mask_len': int64_feature(text_max_len - text_len),
                'text_len':int64_feature(text_len),
                'text':Byte_feature(text),
                'image_max_len': int64_feature(image_max_len),
                'image_mask_len':int64_feature(image_max_len-image_len),
                'image_len':int64_feature(image_len),
                'image':Byte_feature(image),
                'label':int64_feature(label)

            }
        )
    )

    return tf_example

def generate_tf_record_from_dict(datas,record_path,text_max_len,image_max_len):

    with tf.io.TFRecordWriter(record_path) as writer:
        for data in datas:
            tf_example = generate_tf_example(data,text_max_len,image_max_len)
            writer.write(tf_example.SerializeToString())


def parser(record):

    features = {
        'st': tf.FixedLenFeature((),default_value='',dtype=tf.string),
        'url': tf.FixedLenFeature((),default_value="",dtype=tf.string),
        'url_len': tf.FixedLenFeature((),default_value=0,dtype=tf.int64),
        'sentence_len': tf.FixedLenFeature((),default_value='',dtype=tf.string),
        'text_mask_len':tf.FixedLenFeature((),default_value=0,dtype=tf.int64),
        'text_max_len': tf.FixedLenFeature((), default_value=0, dtype=tf.int64),
        'text_len':tf.FixedLenFeature((),default_value=0,dtype=tf.int64),
        'text': tf.FixedLenFeature((),default_value='',dtype=tf.string),
        'image_mask_len':tf.FixedLenFeature((),default_value=0,dtype=tf.int64),
        'image_max_len': tf.FixedLenFeature((), default_value=0, dtype=tf.int64),
        'image_len':tf.FixedLenFeature((),default_value=0,dtype=tf.int64),
        'image': tf.FixedLenFeature((),default_value='',dtype=tf.string),
        'label': tf.FixedLenFeature((),default_value=0,dtype=tf.int64)
    }

    example = tf.io.parse_single_example(record,features)
    sentence_len = tf.reshape(tf.decode_raw(example['sentence_len'],tf.int64),[-1])
    print(sentence_len)
    text_mask_len = example['text_mask_len']
    text_max_len = example['text_max_len']
    text_len = example['text_len']
    image_len = example['image_len']
    image_mask_len = example['image_mask_len']
    image_max_len = example['image_max_len']

    st = tf.reshape(tf.decode_raw(example['st'],tf.int64),[16],name="pip_reshape_st")
    url = tf.reshape(tf.decode_raw(example['url'],tf.int64),[200],name="pip_reshap_url")
    #url_mask = tf.reshape(tf.decode_raw(example['url_mask'],tf.int64),[200],name="pip_reshape_url_mask")
    url_mask_idx = example['url_len']




    text_ = tf.reshape(tf.decode_raw(example['text'],tf.int64),[text_len,512],name="pip_reshape_text")
    #text_add = tf.constant(np.zeros(shape=[text_mask_len,512]),name='pip_text_add')
    text_add = tf.zeros(shape=[text_mask_len, 512], name='pip_text_add',dtype=tf.int64)
    text = tf.concat([text_,text_add],axis=0)



    sentence_mask_idx = tf.concat([sentence_len,tf.zeros([text_mask_len],dtype=tf.int64)],axis=0)   #句子级别的实际长度
    text_mask_idx = text_len

    image_ = tf.reshape(tf.decode_raw(example['image'],tf.float32),[image_len,2048],name="pip_reshape_image")
    image_add = tf.zeros(shape=[image_mask_len,2048],dtype=tf.float32,name='pip_mask_image')
    image = tf.concat([image_,image_add],axis=0)

    image_mask_idx = image_len

    label = example['label']

    return st, url, url_mask_idx, text, sentence_mask_idx, text_mask_idx, image,image_mask_idx, label

def read_tfRecord(record_path,batch_size,epoches):
    #dataset = tf.data.TFRecordDataset(record_path).interleave(tf.data.TFRecordDataset,cycle_length=10)
    dataset = tf.data.TFRecordDataset(record_path)
    #dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=lambda x:parser(x),num_parallel_calls=64, batch_size=batch_size))
    dataset = dataset.map(lambda x:parser(x),num_parallel_calls=64)
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).repeat(epoches).prefetch(1000)
    #dataset = dataset.repeat(epoches).prefetch(1000)
    return dataset



if __name__ == "__main__":
    # sample_list,text_max_len,image_max_len = generate_test_sample(10000)
    # for i in range(5):
    #     generate_tf_record_from_dict(sample_list[i*2000:(i+1)*2000],'../data/test_sample{}.record'.format(i),text_max_len,image_max_len)
    """
    url 长度 200
    url 字符数 88
    text 长度 1
    sentence 长度 512
    
    text 词的长度 21173
    image 长度 最长10
    
    
    """
    # train_list,test_list,valid_list  =  generate_sample_list()
    # with open("../data/train_list_add_feature.pickle","wb") as f:
    #     pickle.dump(train_list,f,0)
    # with open("../data/test_list_add_feature.pickle","wb") as f:
    #     pickle.dump(test_list,f,0)
    # with open("../data/valid_list_add_feature.pickle","wb") as f:
    #     pickle.dump(valid_list,f,0)
    # generate_tf_record_from_dict(train_list,'../data/train_record_add_feature.record',1,10)
    #
    # data_2015 = load_sample_list("../data/data_2015.pickle")
    # data_2016 = load_sample_list("../data/data_2016.pickle")
    #
    # with open("../data/data_2015_processed.pickle", "wb") as f:
    #     pickle.dump(data_2015, f, 0)
    #
    # with open("../data/data_2016_processed.pickle", "wb") as f:
    #     pickle.dump(data_2016, f, 0)

    train_list_all = generate_all_data_example()
    # with open("../data/all_data_add_feature.pickle",'rb') as f:
    #     train_list_all = pickle.loads(f.read())
    generate_tf_record_from_dict(train_list_all,'../data/all_train_record_add_feature_case.record',1,10)
    with open("../data/all_data_add_feature_with_case.pickle",'wb') as f:
        pickle.dump(train_list_all, f, 0)

    # data_2020 = load_sample_list("../data/negative_data_200_case.pickle")
    # with open("../data/data_2020_processed_case.pickle", "wb") as f:
    #     pickle.dump(data_2020, f, 0)
    print("hello word")





