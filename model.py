#!/usr/bin/env python
# -*- coding:utf-8 -*- 

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from config import Config
import tensorflow as tf
from dataset import Dataset
import math

import visdomer


from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import pickle




import time
from tensorboard.plugins import pr_curve
from collections import defaultdict

class Model(object):

    def __init__(self, config, dataset):

        self.config = config
        self.dataset = dataset
        #self.batch_size = self.config.get_parameter('batch_size')
        self.graph = tf.Graph()
        self.gpu_options = tf.GPUOptions(allow_growth=True)
        #visdom
        #self.visdom = vis

    def _creat_placeholder(self):

        with tf.name_scope("learning_rate_config"):
            self.global_step = tf.constant(0, name='global_steps')
            self.lr_step = tf.placeholder(tf.int32,shape=[],name='lr_step')

            self.learning_rate = tf.train.natural_exp_decay(self.config.get_parameter('learning_rate'), self.lr_step,
                                                            self.config.get_parameter('decay_step'),
                                                            self.config.get_parameter('decay_rate'),
                                                            self.config.get_parameter('stair_case'),
                                                         name='learning_rate')
            tf.summary.scalar('learning_rate', self.learning_rate)

        with tf.name_scope("input_data"):
            self.is_training = tf.placeholder(tf.bool, shape=[], name='is_train')
            self.keep_prob = tf.placeholder(tf.float32,shape=[],name='dropout_rate')

            self.batch_size = tf.placeholder(tf.int32,
                                           shape=[],
                                           name="batch_size")
            self.st_input = tf.placeholder(tf.float64,
                                           shape=[None, self.config.get_parameter('st_dimension')],
                                           name="st_input")

            self.url_input = tf.placeholder(tf.int32,
                                            shape=[None, self.config.get_parameter('url_dimension')],
                                            name='url_input')
            self.url_mask_idx_input = tf.placeholder(tf.int32, shape=[None], name="url_mask_idx")  # (batch_size,)
            self.text_input = tf.placeholder(tf.int32,
                                             shape=[None, self.config.get_parameter('max_text_len'),
                                                    self.config.get_parameter('max_sentence_len')])

            self.sentence_mask_idx_input = tf.placeholder(tf.int32,
                                                          shape=[None, self.config.get_parameter('max_text_len')],
                                                          name='sentence_mask_idx')
            self.text_mask_idx_input = tf.placeholder(tf.int32,
                                                      shape=[None, ], name='text_mask_id')

            self.image_input = tf.placeholder(tf.float32,
                                              shape=[None, self.config.get_parameter('max_image_len'),
                                                     self.config.get_parameter('image_size')],
                                              name='image')

            self.image_mask_idx_input = tf.placeholder(tf.int32, shape=[None, ], name='image_mask_idx')

            self.label_input = tf.placeholder(tf.int32, shape=[None, 1], name='image')

    def _creat_weight(self, shape=[], scope=None, name='', normlization='', trainable=True):

        var = tf.Variable(tf.truncated_normal(shape=shape,
                                            mean=0.0,
                                            stddev=tf.sqrt(tf.div(2.0, sum(shape)))),
                         dtype=tf.float32,
                         name=name,
                         trainable=trainable
                        )

        if 'l2' == normlization:
            tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(self.config.get_parameter('regularization_rate'))(var))

        return var

    def _creat_bias(self, shape=[], scope=None, name='', normlization='', trainable=True):

        var = tf.Variable(tf.truncated_normal(shape=shape,
                                                mean=0.0,
                                                stddev=tf.sqrt(tf.div(2.0,sum(shape)))),
                            dtype=tf.float32,
                            name=name,
                            trainable=trainable)
        if 'l2' == normlization:
            tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(self.config.get_parameter('regularization_rate'))(var))
        return var


    def _creat_variables(self):

        normlization = self.config.get_parameter('normalization')
        # 处理结构化数据
        with tf.name_scope('st_variable'):


            # mapping
            self.st_weight_1 = self._creat_weight(
                shape=[self.config.get_parameter('st_dimension'), self.config.get_parameter('st_hidden_size')],
                name='st_weight_w',
                normlization=normlization)
            self.st_bias_1 = self._creat_bias(shape=[self.config.get_parameter('st_hidden_size')],
                                            name='st_weight_b',
                                            normlization=normlization)

            self.st_weight_2 = self._creat_weight(
                shape=[self.config.get_parameter('st_hidden_size'), self.config.get_parameter('st_hidden_size')],
                name='st_weight_w',
                normlization=normlization)
            self.st_bias_2 = self._creat_bias(shape=[self.config.get_parameter('st_hidden_size')],
                                              name='st_weight_b',
                                              normlization=normlization)

            self.st_weight_3 = self._creat_weight(
                shape=[self.config.get_parameter('st_hidden_size'), self.config.get_parameter('st_hidden_size')],
                name='st_weight_w',
                normlization=normlization)
            self.st_bias_3 = self._creat_bias(shape=[self.config.get_parameter('st_hidden_size')],
                                              name='st_weight_b',
                                              normlization=normlization)



            self.st_weight = self._creat_weight(shape=[self.config.get_parameter('st_hidden_size'), self.config.get_parameter('st_final_size')],
                                                name='st_weight_w',
                                                normlization=normlization)
            self.st_bias = self._creat_bias(shape=[self.config.get_parameter('st_final_size')],
                                            name='st_weight_b',
                                            normlization=normlization)



            ## attention
            self.st_attention_target = self._creat_weight(
                shape=[1,self.config.get_parameter('st_final_size')],
                name='st_attention_target',
                normlization=normlization)


            tf.summary.histogram("st_weight", self.st_weight)
            tf.summary.histogram("st_bias", self.st_bias)


        # 处理url数据 scope = 'url_process'
        with tf.name_scope('url_variable'):

            self.url_embedding = self._creat_weight(shape=[self.config.get_parameter('url_vocab_size'), self.config.get_parameter('url_embedding_size')],
                                                    name='url_embedding',
                                                    normlization=normlization)


            self.url_feature_weight = self._creat_weight(shape=[self.config.get_parameter('url_embedding_size'), self.config.get_parameter('url_final_size')],
                                                         name='url_feature_weight',
                                                         normlization=normlization)


            self.url_feature_bias = self._creat_bias(shape=[self.config.get_parameter('url_final_size')],
                                                     name='url_feature_bias',
                                                     normlization=normlization)

            self.url_attention_target = self._creat_weight(shape=[self.config.get_parameter('url_final_size'), 1],
                                                           name='url_embedding_target',
                                                           normlization=normlization)

        # 处理文本数据 scope = 'text_process'
        with tf.name_scope('text_variable'):

            self.text_embedding = self._creat_weight(shape=[self.config.get_parameter('text_vocab_size'),self.config.get_parameter(('text_embedding_size'))],
                                                     name='text_embedding')

            # 词级别
            self.text_word_feature_weight = self._creat_weight(shape=[self.config.get_parameter('text_embedding_size'), self.config.get_parameter('text_final_size')],
                                                               name='text_word_feature_weight',
                                                               normlization=normlization)
            self.text_word_feature_bias = self._creat_bias(shape=[self.config.get_parameter('text_final_size')],
                                                           name='text_word_feature_bias',
                                                           normlization=normlization)
            self.text_word_attention_target = self._creat_weight(shape=[self.config.get_parameter('text_final_size'), 1],
                                                            name='text_attention_target',
                                                            normlization=normlization)




        # 处理图片数据 scope = 'image_process'
        with tf.name_scope('image_variable'):
            self.image_weight = self._creat_weight(shape=[self.config.get_parameter('image_hidden_size'),self.config.get_parameter('image_hidden_size')],
                                                   name="image_weight",
                                                   normlization=normlization)
            self.image_bias = self._creat_bias(shape=[self.config.get_parameter('image_hidden_size')],
                                               name="image_bias",
                                               normlization=normlization)
            self.image_attention_target = self._creat_weight(shape=[self.config.get_parameter('image_hidden_size'),1],
                                                                     name='sentence_attention_target',
                                                                     normlization=normlization)
        # 集成url,text,image 特征

        with tf.name_scope("integration_variable"):
            self.integration_attention_target = self._creat_weight(shape=[self.config.get_parameter('final_attention_size'),1],
                                                                    name='intergration_attention_target',
                                                                    normlization=normlization)
            self.integration_attention_target_y = self._creat_weight(
                shape=[self.config.get_parameter('final_attention_size'), 1],
                name='projection_intergration_attention_target_y',
                normlization=normlization)
            self.projection_matrix_1 = self._creat_weight(shape=[self.config.get_parameter('url_embedding_size'), self.config.get_parameter('final_attention_size')], name='matrix_1')
            self.projection_matrix_2 = self._creat_weight(shape=[self.config.get_parameter('text_embedding_size'), self.config.get_parameter('final_attention_size')], name='matrix_2')
            self.projection_matrix_3 = self._creat_weight(shape=[self.config.get_parameter('image_hidden_size'), self.config.get_parameter('final_attention_size')], name='matrix_3')
            self.projection_D = self._creat_weight(shape=[self.config.get_parameter('final_attention_size'), self.config.get_parameter('final_attention_size')], name="projection_D")

        # 输出
        with tf.name_scope('inference_variable'):
            self.final_weight = self._creat_weight(shape=[self.config.get_parameter('final_attention_size'), 1], name="inference_weight")  # TODO:final_input_size
            self.final_bias = self._creat_bias(shape=[1], name='inference_bias')






    def st_feature(self):
        with tf.name_scope("st_feature"):

            # 第一层 [batch_size, hidden_size]
            st_MLP = tf.nn.relu(tf.add(tf.matmul(tf.cast(self.st_input,tf.float32),self.st_weight_1), self.st_bias_1))
            st_MLP_drop = tf.layers.dropout(st_MLP, rate=self.keep_prob, training=self.is_training)


            st_MLP_layer = self.config.get_parameter("st_MLP_layer")
            if st_MLP_layer == 2:
                st_MLP = tf.nn.relu(tf.add(tf.matmul(tf.cast(st_MLP_drop, tf.float32), self.st_weight_2), self.st_bias_2))
                st_MLP_drop = tf.layers.dropout(st_MLP, rate=self.keep_prob, training=self.is_training)

            if st_MLP_layer == 2:
                st_MLP = tf.nn.relu(tf.add(tf.matmul(tf.cast(st_MLP_drop, tf.float32), self.st_weight_3), self.st_bias_3))
                st_MLP_drop = tf.layers.dropout(st_MLP, rate=self.keep_prob, training=self.is_training)


            self.st = tf.nn.relu(tf.add(tf.matmul(tf.cast(st_MLP_drop, tf.float32),self.st_weight), self.st_bias))   # (batch_size,st_dimension) * (st_dimension,final_attention_input_size) = (batch_size,fais)

            self.st_attention = tf.nn.softmax(self.st * self.st_attention_target)   # (batch_size,st_final_size)
            st_feature_output = self.st * self.st_attention

        self.st_feature_ouput = tf.layers.dropout(st_feature_output, rate=self.keep_prob, training=self.is_training)


    def url_feature(self):

        with tf.name_scope("url_feature"):
            self.url_feature_embedding = tf.nn.embedding_lookup(self.url_embedding,self.url_input)  # [batch_size,max_url_len,url_embedding_size]
            lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.config.get_parameter('url_embedding_size')) for _ in range(self.config.get_parameter('url_lstm_layer'))])
            rnn_output, state = tf.nn.dynamic_rnn(lstm_cell, self.url_feature_embedding, dtype=tf.float32)


            fc = tf.add(tf.matmul(rnn_output, self.url_feature_weight), self.url_feature_bias)
            if self.config.get_parameter("batch_normliaztion"):
                fc = tf.layers.batch_normalization(fc, training=self.is_training)
            dense_layer = tf.nn.tanh(fc)

            attention = tf.reduce_sum(tf.matmul(dense_layer , self.url_attention_target), axis=-1)          # [batch_size,max_url_len]
            mask_sequence = tf.sequence_mask(self.url_mask_idx_input, maxlen=self.config.get_parameter('url_dimension'), dtype=tf.float32)
            print(mask_sequence.shape)
            mask_mat = -99999.0 * (1 - mask_sequence)

            self.url_attention_weight = tf.nn.softmax(attention + mask_mat)                              #[batch_size,max_url_len]
            url_feature_output = tf.reduce_sum(rnn_output * tf.expand_dims(self.url_attention_weight, axis=-1),axis=-2)
            self.url_feature_output = tf.layers.dropout(url_feature_output, rate=self.keep_prob, training=self.is_training)

            tf.summary.histogram("url_attention_weight", self.url_attention_weight)



    def text_feature(self):
        with tf.name_scope("text_feature"):

            # ------------------------------TODO:使用简单的全连接网络测试一下  输入[batch_size,max_text_len,sentence_len]
            self.text_feature_embedding = tf.nn.embedding_lookup(self.text_embedding, self.text_input)    # [batch_size,max_text_len,sentence_len,text_embedding_size]
            self.text_feature_embedding_ = tf.squeeze(self.text_feature_embedding, axis=1)
            with tf.variable_scope("text_feature"):
                word_lstm_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(self.config.get_parameter('text_embedding_size')) for _ in range(self.config.get_parameter('text_lstm_layer'))])
                word_rnn_output, state = tf.nn.dynamic_rnn(word_lstm_cell, self.text_feature_embedding_, dtype=tf.float32)

            word_rnn_output_ = tf.expand_dims(word_rnn_output, axis=1)

            # word_rnn_output_projection = tf.layers.dense(word_rnn_output_, units=self.config.get_parameter('text_final_size'), use_bias=False)
            # word_rnn_output_projection = tf.layers.dropout(word_rnn_output_projection, rate=self.keep_prob, training=self.is_training)

            fc_word = tf.add(tf.matmul(word_rnn_output_, self.text_word_feature_weight), self.text_word_feature_bias)

            word_dense_layer = tf.nn.tanh(fc_word)


            # 计算词级别的attention
            with tf.name_scope("text_word_attention"):
                word_mask_sequence = tf.sequence_mask(self.sentence_mask_idx_input, maxlen=self.config.get_parameter('max_sentence_len'), dtype=tf.float32) #[batch_size,max_text_len,sentence_len]
                word_attention = tf.reduce_sum(tf.matmul(word_dense_layer, self.text_word_attention_target),axis=-1)             # [batch_size,max_text_len,sentence_len]
                word_mask_mat = -99999.0 * (1 - word_mask_sequence)
                self.text_word_attention_weight = tf.nn.softmax(word_mask_mat + word_attention)                           # [batch_size,max_text_len,sentence_len]

            text_word_attention_out = tf.reduce_sum(word_rnn_output_ * tf.expand_dims(self.text_word_attention_weight, axis=-1), axis=-2)   # [batch_size,max_text_len,text_final_size]

            self.text_word_attention_out = tf.layers.dropout(text_word_attention_out, rate=self.keep_prob, training=self.is_training)
            self.text_feature_output = tf.squeeze(self.text_word_attention_out, axis=1)
            tf.summary.histogram("text_word_attention_weight", self.text_word_attention_weight)


    def image_feature(self):
        with tf.name_scope("image_feature"):

            # resnet 预训练模型
            image_feature_projection = tf.layers.dense(self.image_input, units=self.config.get_parameter('image_hidden_size'), use_bias=False)


            image_feature = tf.nn.tanh(tf.add(tf.matmul(image_feature_projection, self.image_weight), self.image_bias))


            image_feature_reshape = tf.reshape(image_feature,[self.batch_size, -1, self.config.get_parameter('image_hidden_size')])   #[batch_size,10,64] // TODO:没有影响吧

            image_attention_mask = tf.sequence_mask(self.image_mask_idx_input,maxlen=self.config.get_parameter('max_image_len'),dtype=tf.float32)
            attention_mask_mat = -999999.0 * (1-image_attention_mask)
            image_attention = tf.squeeze(tf.matmul(image_feature_reshape,self.image_attention_target),axis=-1)            # [batch_size,10]
            self.image_attention_weight = tf.nn.softmax(image_attention+attention_mask_mat)
            images_feature_output = tf.reduce_sum(image_feature_projection * tf.expand_dims(self.image_attention_weight,axis= -1), axis=-2)  #[b,10,16]->[b,16]
            self.images_feature_output = tf.layers.dropout(images_feature_output, rate=self.keep_prob, training=self.is_training)




            tf.summary.histogram("image_attention_weight", self.image_attention_weight)
            tf.summary.histogram("image_output", self.images_feature_output)


    def _integration(self):

        with tf.name_scope('feature_integration'):


            self.url_feature_output_projection = tf.matmul(self.url_feature_output, self.projection_matrix_1)
            self.text_feature_output_projection = tf.matmul(self.text_feature_output, self.projection_matrix_2)
            self.images_feature_output_projection = tf.matmul(self.images_feature_output, self.projection_matrix_3)


            # self.features = tf.concat([self.url_feature_output_projection,self.text_feature_output_projection,self.images_feature_output_projection], axis=1) # [batch_size,3*final_attention_size]

            self.url_feature_ = tf.expand_dims(self.url_feature_output_projection, axis=1)
            self.text_feature_ = tf.expand_dims(self.text_feature_output_projection, axis=1)
            self.images_feature_ = tf.expand_dims(self.images_feature_output_projection, axis=1)


            self.features = tf.concat([self.url_feature_, self.text_feature_,self.images_feature_],axis=1)  # [batch_size,3,final_attention_size]
            attention = tf.reduce_sum(tf.matmul(self.features, self.integration_attention_target), axis=-1)  # [batch_size,3]
            self.integration_attention_weight = tf.nn.softmax(attention)

            feature_integration = tf.reduce_sum(self.features * tf.expand_dims(self.integration_attention_weight, axis=-1),
                                                axis=-2)  # [batch_size,final_attention_size]
            self.feature_integration = tf.layers.dropout(feature_integration, rate=self.keep_prob, training=self.is_training)
            tf.summary.histogram('final_attention_weight', self.integration_attention_weight)


        self.feature_integration_output = self.feature_integration  # [batch_size,final_attention_size + 1]


    def _intergration_y(self):
        with tf.name_scope('feature_integration_y'):
            self.url_feature_output_y = tf.expand_dims(tf.layers.dense(self.url_feature_output, units=self.config.get_parameter("final_attention_size"),name="projection_1"), axis=1)
            self.text_feature_output_y = tf.expand_dims(tf.layers.dense(self.text_feature_output, units=self.config.get_parameter("final_attention_size"),name="projection_2"), axis=1)
            self.image_feature_output_y = tf.expand_dims(tf.layers.dense(self.images_feature_output, units=self.config.get_parameter("final_attention_size"),name="projection_3"), axis=1)

            self.features_y = tf.concat([self.url_feature_output_y, self.text_feature_output_y, self.image_feature_output_y], axis=1)
            self.features_y_mlp = tf.layers.dense(self.features_y, units=self.config.get_parameter("final_attention_size"), activation=tf.nn.tanh, name="projection_4")
            attention_y = tf.reduce_sum(tf.matmul(self.features_y_mlp, self.integration_attention_target_y), axis=-1)
            self.integration_attention_weight_y = tf.nn.softmax(attention_y)
            self.feature_integration_output_ = tf.layers.dropout(tf.reduce_sum(self.features_y * tf.expand_dims(self.integration_attention_weight_y, axis=-1), axis=-2),rate=self.keep_prob, training=self.is_training)

        self.feature_integration_output_y = self.feature_integration_output_

    def _create_inference_y(self):
        with tf.name_scope('inference'):
            self.output_y = tf.layers.dense(self.feature_integration_output_y, units=1,name="projection_5")
            self.prediction_y = tf.nn.sigmoid(self.output_y)

    def _create_loss_y(self, var_list):
        self.loss_y = tf.losses.sigmoid_cross_entropy(self.label_input, self.output_y)

        self.loss_projection = tf.reduce_sum(
            tf.pow(self.url_feature_output_projection - tf.matmul(self.url_feature_output_y, self.projection_D), 2) + \
            tf.pow(self.text_feature_output_projection - tf.matmul(self.text_feature_output_y, self.projection_D), 2) + \
            tf.pow(self.images_feature_output_projection - tf.matmul(self.image_feature_output_y, self.projection_D),
                   2))

        self.loss_ = self.loss_y + self.config.get_parameter('loss_r') * self.loss_projection
        #self.loss_ = self.loss_y
        self.optimizer_ = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss_, var_list=var_list)




    def _creat_inference(self):

        with tf.name_scope('inference'):
            # ------------------------ TODO: 使用全连接神经网络进行测试

            self.output = tf.add(tf.matmul(self.feature_integration_output, self.final_weight), self.final_bias)
            print("self.output shape",self.output.shape)
            self.prediction = tf.sigmoid(self.output)

            pr_curve.summary.op(name="pr", labels=tf.cast(self.label_input, tf.bool), predictions=self.prediction,
                                num_thresholds=10)  # pr 曲线？

            prediction_3 = tf.cast(self.prediction > 0.3, tf.int8)
            prediction_5 = tf.cast(self.prediction>0.5,tf.int8)
            prediction_7 = tf.cast(self.prediction>0.7,tf.int8)

            self.train_acc_3, self.train_acc_update_3 = tf.metrics.accuracy(self.label_input,  prediction_3)
            self.train_pr_3, self.train_pr_update_3 = tf.metrics.precision(self.label_input, prediction_3)
            self.train_re_3, self.train_re_update_3 = tf.metrics.recall(self.label_input, prediction_3)

            self.train_acc_5, self.train_acc_update_5 = tf.metrics.accuracy(self.label_input,  prediction_5)
            self.train_pr_5, self.train_pr_update_5 = tf.metrics.precision(self.label_input,  prediction_5)
            self.train_re_5, self.train_re_update_5 = tf.metrics.recall(self.label_input, prediction_5)

            self.train_acc_7, self.train_acc_update_7 = tf.metrics.accuracy(self.label_input, prediction_7)
            self.train_pr_7, self.train_pr_update_7 = tf.metrics.precision(self.label_input, prediction_7)
            self.train_re_7, self.train_re_update_7 = tf.metrics.recall(self.label_input, prediction_7)

            tf.summary.scalar('train_acc_3', self.train_acc_3)
            tf.summary.scalar('train_pr_3', self.train_pr_3)
            tf.summary.scalar('train_re_3', self.train_re_3)

            tf.summary.scalar('train_acc_5', self.train_acc_5)
            tf.summary.scalar('train_pr_5', self.train_pr_5)
            tf.summary.scalar('train_re_5', self.train_re_5)

            tf.summary.scalar('train_acc_7', self.train_acc_7)
            tf.summary.scalar('train_pr_7', self.train_pr_7)
            tf.summary.scalar('train_re_7', self.train_re_7)


    def _creat_loss(self):

        self.loss_x = tf.losses.sigmoid_cross_entropy(self.label_input, self.output)       #TODO：使用二分类交叉熵损失函数  + self.config.get_parameter("loss_r") * self.loss_projection



        self.loss = self.loss_x
        self.metrics_loss = tf.summary.scalar('train_loss', self.loss)





        # gradients_node = tf.gradients(self.loss,self.text_word_attention_weight)
        # tf.summary.histogram("gradient",gradients_node)


    def _creat_optimizer(self):

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        # 自己计算梯度
        # optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        #
        # params = tf.trainable_variables()
        # gradients_vars = optimizer.compute_gradients(self.loss)
        # clipped_gradients_vars = [(tf.clip_by_value(grad,-1.0,1.0),vars) for grad,vars in gradients_vars ]
        # self.optimizer = optimizer.apply_gradients(clipped_gradients_vars)


    def build_graph(self):
        with self.graph.as_default():
            self._creat_placeholder()
            self._creat_variables()
            self.st_feature()
            self.url_feature()
            self.text_feature()
            self.image_feature()
            self._integration()
            self._creat_inference()
            self._creat_loss()
            self._creat_optimizer()
            # self._intergration_y()
            # self._create_inference_y()
            #
            # trainable_vari = []
            # varis = self.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            # for vari in varis:
            #
            #
            #     if "projection" in str(vari.name):
            #         trainable_vari.append(vari)
            # self._create_loss_y(trainable_vari)
            #
            # for vari in trainable_vari:
            #     print(vari.name)

            #tf.summary.scalar('concat_loss_recall', self.concat_loss_recall)

            self.summary_merge = tf.summary.merge_all()

    def train(self):
        self.build_graph()
        with self.graph.as_default():

            saver = tf.train.Saver(max_to_keep=5)  # 最多保存5个模型
            interval = math.ceil(self.config.get_parameter('num_data') / self.config.get_parameter('batch_size'))
            batches = self.dataset.batches(self.config.get_parameter('batch_size'))

            test_set = self.dataset.load_test_set()
            valid_set = self.dataset.load_valid_set()

            test_interval = self.config.get_parameter('test_interval')
            count = 0  # 间隔多少保存一下结果
            global_step = 0
            learning_rate_step = 0

            total_loss = 0
            test_result = defaultdict(list)
            valid_result = defaultdict(list)



            with tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options),graph=self.graph) as sess:

                train_writer = tf.summary.FileWriter('tmp/logs/train/', sess.graph)
                test_writer = tf.summary.FileWriter('tmp/logs/test/')

                # 是否使用预训练
                if self.config.get_parameter('pre_train') == 1:
                    chkpt_fname = tf.train.latest_checkpoint(self.config.get_parameter('model_path'))
                    sess.run(tf.global_variables_initializer())
                    saver.restore(sess, chkpt_fname)
                    global_step = sess.run([self.global_step])
                else:
                    sess.run([tf.local_variables_initializer(), tf.global_variables_initializer()])


                total_time = 0
                precise_list = []
                recall_list = []
                accuracy_list = []
                while True:
                    time_to_test = False
                    time_to_save = False
                    time_1 = time.time()
                    try:
                        batch = self.dataset.get_next(sess, batches)
                    except:
                        batch = None
                    time_2 = time.time()
                    # print(type(batch))
                    if batch != None:
                        time_1 = time.time()
                        count += 1                # mini batch
                        feed_dict = {self.global_step: global_step,
                                     self.lr_step:learning_rate_step,
                                     self.batch_size: batch['batch_size'],
                                      self.st_input: batch['st'],
                                      self.url_input: batch['url'],
                                      self.url_mask_idx_input: batch['url_mask_idx'],
                                      self.text_input: batch['text'],
                                      self.sentence_mask_idx_input: batch['sentence_mask_idx'],
                                      self.text_mask_idx_input: batch['text_mask_idx'],
                                      self.image_input: batch['image'],
                                      self.image_mask_idx_input: batch['image_mask_idx'],
                                      self.label_input: batch['label'],
                                     self.keep_prob:self.config.get_parameter('dropout_keep_prob'),
                                     self.is_training:True}

                        time_2 = time.time()


                        summary_, train_acc_3, train_prec_3, train_recall_3,\
                        train_acc_5, train_prec_5, train_recall_5,\
                        train_acc_7, train_prec_7, train_recall_7, loss, _ = sess.run([self.summary_merge,
                                                                                           self.train_acc_update_3,self.train_pr_update_3,self.train_re_update_3,
                                                                                           self.train_acc_update_5,self.train_pr_update_5,self.train_re_update_5,
                                                                                           self.train_acc_update_7,self.train_pr_update_7,self.train_re_update_7,
                                                                                           self.loss, self.optimizer], feed_dict=feed_dict)
                        precise_list.append(train_prec_5)
                        recall_list.append(train_recall_5)
                        accuracy_list.append(train_acc_5)



                        total_loss+=loss
                        train_writer.add_summary(summary_,count)

                        train_writer.flush()
                        time_3 = time.time()
                        #print("batch {}: \t total_time:{}s,\t batch_time:{} \t gpu_time:{}".format(count,time_3-time_1,time_2-time_1,time_3-time_2))
                        total_time += (time_3-time_1)




                        if count % interval == 0:
                            global_step += 1
                            print("epoch:{},\tloss:{},\ttotal_time:{} \t train_acc:{} \t train_recall:{} \t train_precise:{}".format(global_step,total_loss/interval,total_time,np.mean(accuracy_list),np.mean(recall_list),np.mean(precise_list)))
                            total_loss = 0
                            total_time = 0
                            time_to_test = True
                            time_to_save = True
                            precise_list = []
                            recall_list = []
                            accuracy_list = []

                        # 设置什么时候学习率开始递减
                        if global_step >  self.config.get_parameter('learning_rate_decay_step'):
                            learning_rate_step+=1

                        if global_step % self.config.get_parameter('save_interval')  == 0 and time_to_save: # TODO：设置每隔多少次保存模型
                            print("保存模型")
                            saver.save(sess, self.config.get_parameter('model_path'))                   # 要保存结果
                            time_to_save = False

                        if global_step % test_interval  == 0 and time_to_test:
                            acc,pre,re,f1,roc = self.test_during_train(sess,test_set,global_step,self.prediction)
                            test_result["acc"].append(acc)
                            test_result["pre"].append(pre)
                            test_result["rec"].append(re)
                            test_result["f1"].append(f1)
                            test_result["roc"].append(roc)
                            #print(acc)
                            print("test_set acc:{}\t precision :{} \t recall: {} \t f1_score:{} \t roc:{}".format(acc,pre,re,f1,roc))

                        if global_step % self.config.get_parameter('valid_interval')  == 0 and time_to_test:
                            acc,pre,re,f1,roc = self.test_during_train(sess,valid_set,global_step,self.prediction)
                            valid_result["acc"].append(acc)
                            valid_result["pre"].append(pre)
                            valid_result["rec"].append(re)
                            valid_result["f1"].append(f1)
                            valid_result["roc"].append(roc)
                            print("valid_set acc:{}\t precision :{} \t recall: {} \t f1_score:{} \t roc:{}".format(acc,pre,re,f1,roc))



                    else:

                        print("训练结束")
                        break
                print(count)
            train_writer.close()
            test_writer.close()

            #test_writer.close()
            return test_result,valid_result


    def   test_during_train(self,sess,test_set,global_step,obj):

        feed_dict = {self.global_step: global_step,
                     self.batch_size: test_set['batch_size'],
                     self.st_input: test_set['st'],
                     self.url_input: test_set['url'],
                     self.url_mask_idx_input: test_set['url_mask_idx'],
                     self.text_input: test_set['text'],
                     self.sentence_mask_idx_input: test_set['sentence_mask_idx'],
                     self.text_mask_idx_input: test_set['text_mask_idx'],
                     self.image_input: test_set['image'],
                     self.image_mask_idx_input: test_set['image_mask_idx'],
                     self.label_input: test_set['label'],
                     self.keep_prob: 0.5,
                     self.is_training: False}

        prediction_out = sess.run([obj],feed_dict=feed_dict)

        prediction = (prediction_out[0] >= 0.5).astype(np.int32).reshape([-1])

        lable = test_set["label"].reshape([-1])

        acc = accuracy_score(lable, prediction)
        precision = precision_score(lable, prediction)
        recall = recall_score(lable, prediction)
        f1 = f1_score(lable, prediction)
        roc = roc_auc_score(lable, prediction_out[0].reshape([-1]))

        return acc, precision, recall, f1, roc





    def train_other(self):
        self.build_graph()
        with self.graph.as_default() as graph:

            with tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options), graph=graph) as sess:

                saver = tf.train.Saver(max_to_keep=5)
                sess.run([tf.local_variables_initializer(),tf.global_variables_initializer()])
                saver.restore(sess, self.config.get_parameter('model_path'))

                interval = math.ceil(self.config.get_parameter('num_data') / self.config.get_parameter('batch_size'))
                batches = self.dataset.batches(self.config.get_parameter('batch_size'))

                test_set = self.dataset.load_test_set()
                valid_set = self.dataset.load_valid_set()

                test_interval = self.config.get_parameter('test_interval')
                count = 0  # 间隔多少保存一下结果
                global_step = 0
                learning_rate_step = 0

                total_loss = 0
                test_result = defaultdict(list)
                valid_result = defaultdict(list)

                try:
                    train_writer = tf.summary.FileWriter('tmp/logs/train/', sess.graph)
                    test_writer = tf.summary.FileWriter('tmp/logs/test/')

                    total_time = 0
                    precise_list = []
                    recall_list = []
                    accuracy_list = []
                    while True:
                        time_to_test = False
                        time_to_save = False
                        time_1 = time.time()
                        try:
                            batch = self.dataset.get_next(sess, batches)
                        except:
                            batch = None
                        time_2 = time.time()
                        # print(type(batch))
                        if batch != None:
                            time_1 = time.time()
                            count += 1  # mini batch
                            feed_dict = {self.global_step: global_step,
                                         self.lr_step: learning_rate_step,
                                         self.batch_size: batch['batch_size'],
                                         self.st_input: batch['st'],
                                         self.url_input: batch['url'],
                                         self.url_mask_idx_input: batch['url_mask_idx'],
                                         self.text_input: batch['text'],
                                         self.sentence_mask_idx_input: batch['sentence_mask_idx'],
                                         self.text_mask_idx_input: batch['text_mask_idx'],
                                         self.image_input: batch['image'],
                                         self.image_mask_idx_input: batch['image_mask_idx'],
                                         self.label_input: batch['label'],
                                         self.keep_prob: self.config.get_parameter('dropout_keep_prob'),
                                         self.is_training: True}

                            time_2 = time.time()

                            summary_, train_acc_3, train_prec_3, train_recall_3, \
                            train_acc_5, train_prec_5, train_recall_5, \
                            train_acc_7, train_prec_7, train_recall_7, loss, _ = sess.run([self.summary_merge,
                                                                                           self.train_acc_update_3,
                                                                                           self.train_pr_update_3,
                                                                                           self.train_re_update_3,
                                                                                           self.train_acc_update_5,
                                                                                           self.train_pr_update_5,
                                                                                           self.train_re_update_5,
                                                                                           self.train_acc_update_7,
                                                                                           self.train_pr_update_7,
                                                                                           self.train_re_update_7,
                                                                                           self.loss_, self.optimizer_],
                                                                                          feed_dict=feed_dict)
                            precise_list.append(train_prec_5)
                            recall_list.append(train_recall_5)
                            accuracy_list.append(train_acc_5)

                            total_loss += loss
                            train_writer.add_summary(summary_, count)

                            train_writer.flush()
                            time_3 = time.time()
                            # print("batch {}: \t total_time:{}s,\t batch_time:{} \t gpu_time:{}".format(count,time_3-time_1,time_2-time_1,time_3-time_2))
                            total_time += (time_3 - time_1)

                            if count % interval == 0:
                                global_step += 1
                                print(
                                    "epoch:{},\tloss:{},\ttotal_time:{} \t train_acc:{} \t train_recall:{} \t train_precise:{}".format(
                                        global_step, total_loss / interval, total_time, np.mean(accuracy_list),
                                        np.mean(recall_list), np.mean(precise_list)))
                                total_loss = 0
                                total_time = 0
                                time_to_test = True
                                time_to_save = True
                                precise_list = []
                                recall_list = []
                                accuracy_list = []

                            # 设置什么时候学习率开始递减
                            if global_step > self.config.get_parameter('learning_rate_decay_step'):
                                learning_rate_step += 1

                            if global_step % self.config.get_parameter(
                                    'save_interval') == 0 and time_to_save:  # TODO：设置每隔多少次保存模型
                                print("保存模型")
                                saver.save(sess, self.config.get_parameter('model_path_y'))  # 要保存结果
                                time_to_save = False

                            if global_step % test_interval == 0 and time_to_test:
                                acc, pre, re, f1, roc = self.test_during_train(sess, test_set, global_step,self.prediction_y)
                                test_result["acc"].append(acc)
                                test_result["pre"].append(pre)
                                test_result["rec"].append(re)
                                test_result["f1"].append(f1)
                                test_result["roc"].append(roc)
                                # print(acc)
                                print("test_set acc:{}\t precision :{} \t recall: {} \t f1_score:{} \t roc:{}".format(acc,
                                                                                                                      pre,
                                                                                                                      re,
                                                                                                                      f1,
                                                                                                                      roc))

                            if global_step % self.config.get_parameter('valid_interval') == 0 and time_to_test:
                                acc, pre, re, f1, roc = self.test_during_train(sess, valid_set, global_step,self.prediction_y)
                                valid_result["acc"].append(acc)
                                valid_result["pre"].append(pre)
                                valid_result["rec"].append(re)
                                valid_result["f1"].append(f1)
                                valid_result["roc"].append(roc)
                                print("valid_set acc:{}\t precision :{} \t recall: {} \t f1_score:{} \t roc:{}".format(acc,
                                                                                                                       pre,
                                                                                                                       re,
                                                                                                                       f1,
                                                                                                                       roc))



                        else:

                            print("训练结束")
                            break
                    print(count)
                except Exception as e:
                    print("训练出错！",e)

                train_writer.close()
                test_writer.close()

                # test_writer.close()
                return test_result, valid_result

    def load_model(self):
        with self.graph.as_default():
            self.build_graph()
            self.sess = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=self.gpu_options),graph=self.graph)
            self.sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(self.sess, self.config.get_parameter('model_path_y'))

    def predict(self, x):
        result = dict()
        feed_dict = {self.global_step: 0,
                                     self.batch_size: x['batch_size'],
                                     self.st_input: x['st'],
                                     self.url_input: x['url'],
                                     self.url_mask_idx_input: x['url_mask_idx'],
                                     self.text_input: x['text'],
                                     self.sentence_mask_idx_input:x['sentence_mask_idx'],
                                     self.text_mask_idx_input: x['text_mask_idx'],
                                     self.image_input:x['image'],
                                     self.image_mask_idx_input: x['image_mask_idx'],
                                     self.label_input:x['label'],
                                     self.keep_prob: 0.5,
                                     self.is_training: False}

        prediction,st_weight,url_attention,word_attention,image_attention,final_attention = self.sess.run([self.prediction_y,
                                                                                                           self.st_attention,
                                                                                                           self.url_attention_weight,
                                                                                                           self.text_word_attention_weight,
                                                                                                           self.image_attention_weight,
                                                                                                           self.integration_attention_weight_y],
                                                                                                          feed_dict=feed_dict)
        result["prediction"] = prediction
        result["st_weight"] = st_weight
        result["url_attention"] = url_attention
        result["word_attention"] = word_attention
        result["image_attention"] = image_attention
        result["final_attention"] = final_attention
        return result

    def close_model(self):
        self.sess.close()

def run(para):
    config = Config()
    # config.parameters.final_attention_size = para


    data = Dataset('data/', epoches=100)
    model = Model(config, data)
    # model.build_graph()
    test_result, valid_result = model.train()

    return test_result, valid_result

def run_y(para):
    config = Config()
    config.parameters.final_attention_size = para
    data = Dataset("data/",epoches=100)
    model = Model(config,data)
    test_result,valid_result = model.train_other()
    return test_result,valid_result



def find_top_N(data, N):
    data_ = pd.Series(data)
    data_sorted = data_.sort_values(ascending=False)
    return data_sorted[:N].values,data_sorted.index.values

def get_values(data,index):
    data_ = pd.Series(data)
    return data_.loc[index]


def save_result(path,data):
    with open(path,'wb') as f:
        pickle.dump(data, f, 0)

if __name__ == "__main__":

    # test_result, valid_result = run(1)

    vis =  visdomer.Visdomer(env="with projection")
    opts = dict(
            xtickmin=0,
            xtickmax=10,
            xtickstep=1,
            ytickmin=0.9,
            ytickmax=1,
            ytickstep=0.01,
            markersymbol='dot',
            markersize=5,
            showlegend=True
            )

    parameter_list = [8]

    parameter_list_result = defaultdict(list)
    legend = []
    for para in parameter_list:


        # hidden_size
        legend.append("pretrain :{}".format(para))

        test_acc_list = []
        test_pre_list = []
        test_rec_list = []
        test_f1_list = []
        test_roc_list = []

        valid_acc_list = []
        valid_pre_list = []
        valid_rec_list = []
        valid_f1_list = []
        valid_roc_list = []

        #time.sleep(3)
        # run(para)


        for i in range(10):

            time_1 = time.time()

            test_result, valid_result = run(para)

            time_2 = time.time()
            print("该次训练用时：{}分钟" .format((time_2-time_1)/60))

            test_roc_, data_index = find_top_N(test_result["roc"],5)

            test_acc = np.mean(get_values(test_result["acc"], data_index))
            test_pre = np.mean(get_values(test_result["pre"], data_index))
            test_f1 = np.mean(get_values(test_result["f1"], data_index))
            test_rec = np.mean(get_values(test_result["rec"], data_index))
            test_roc = np.mean(get_values(test_result["roc"], data_index))

            valid_roc_,data_index  = find_top_N(valid_result["roc"], 5)

            valid_acc = np.mean(get_values(valid_result["acc"], data_index))
            valid_pre = np.mean(get_values(valid_result["pre"], data_index))
            valid_f1 = np.mean(get_values(valid_result["f1"], data_index))
            valid_rec = np.mean(get_values(valid_result["rec"], data_index))
            valid_roc = np.mean(get_values(valid_result["roc"], data_index))


            test_acc_list.append(test_acc)
            test_pre_list.append(test_pre)
            test_rec_list.append(test_rec)
            test_f1_list.append(test_f1)
            test_roc_list.append(test_roc)

            valid_acc_list.append(valid_acc)
            valid_pre_list.append(valid_pre)
            valid_rec_list.append(valid_rec)
            valid_f1_list.append(valid_f1)
            valid_roc_list.append(valid_roc)

        parameter_list_result["test_acc"].append(test_acc_list)
        parameter_list_result["test_pre"].append(test_pre_list)
        parameter_list_result["test_rec"].append(test_rec_list)
        parameter_list_result["test_f1"].append(test_f1_list)
        parameter_list_result["test_roc"].append(test_roc_list)

        parameter_list_result["valid_acc"].append(valid_acc_list)
        parameter_list_result["valid_pre"].append(valid_pre_list)
        parameter_list_result["valid_rec"].append(valid_rec_list)
        parameter_list_result["valid_f1"].append(valid_f1_list)
        parameter_list_result["valid_roc"].append(valid_roc_list)

        vis.update_graph("test_drop_acc", "test", {"X": np.transpose(np.array(parameter_list_result["test_acc"]),[1,0]), "legend": legend}, visdomer.BOXPLOT, visdomer.APPEND, opts=opts)
        vis.update_graph("test_drop_pre","test",  {"X": np.transpose(np.array(parameter_list_result["test_pre"]),[1,0]), "legend": legend}, visdomer.BOXPLOT,
                         visdomer.APPEND, opts=opts)
        vis.update_graph("test_drop_rec", "test", {"X": np.transpose(np.array(parameter_list_result["test_rec"]),[1,0]), "legend": legend}, visdomer.BOXPLOT,
                         visdomer.APPEND, opts=opts)
        vis.update_graph("test_drop_f1","test",  {"X": np.transpose(np.array(parameter_list_result["test_f1"]),[1,0]), "legend": legend}, visdomer.BOXPLOT,
                         visdomer.APPEND, opts=opts)
        vis.update_graph("test_drop_roc","test",  {"X": np.transpose(np.array(parameter_list_result["test_roc"]),[1,0]), "legend": legend}, visdomer.BOXPLOT,
                         visdomer.APPEND, opts=opts)

        vis.update_graph("valid_drop_acc", "test",
                         {"X": np.transpose(np.array(parameter_list_result["valid_acc"]), [1, 0]), "legend": legend},
                         visdomer.BOXPLOT, visdomer.APPEND, opts=opts)
        vis.update_graph("valid_drop_pre", "test",
                         {"X": np.transpose(np.array(parameter_list_result["valid_pre"]), [1, 0]), "legend": legend},
                         visdomer.BOXPLOT,
                         visdomer.APPEND, opts=opts)
        vis.update_graph("valid_drop_rec", "test",
                         {"X": np.transpose(np.array(parameter_list_result["valid_rec"]), [1, 0]), "legend": legend},
                         visdomer.BOXPLOT,
                         visdomer.APPEND, opts=opts)
        vis.update_graph("valid_drop_f1", "test",
                         {"X": np.transpose(np.array(parameter_list_result["valid_f1"]), [1, 0]), "legend": legend},
                         visdomer.BOXPLOT,
                         visdomer.APPEND, opts=opts)
        vis.update_graph("valid_drop_roc", "test",
                         {"X": np.transpose(np.array(parameter_list_result["valid_roc"]), [1, 0]), "legend": legend},
                         visdomer.BOXPLOT,
                         visdomer.APPEND, opts=opts)




    # hidden_size
    save_result("result/best_with_dense.pickle", parameter_list_result)







