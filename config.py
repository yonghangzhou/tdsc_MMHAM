#!/usr/bin/env python
# -*- coding:utf-8 -*- 


import argparse

class Config(object):

    def __init__(self,args=[]):
        parser = argparse.ArgumentParser(description="Run Model")
        parser.add_argument('--batch_size', nargs='?', type=int,default=1000, help='train batch_size')
        parser.add_argument('--attention',nargs="?",type=int,default=0, help="attention function,0 for ...")  # TODO:解释含义
        parser.add_argument('--activation',nargs='?',type=int,default=0,help='activation function,0 for ...')  # TODO：解释含义
        parser.add_argument('--max_text_len',nargs='?',type=int,default=1,help='num of texts per website')
        parser.add_argument('--max_image_len',nargs='?',type=int,default=10,help='num of images per website')
        # parser.add_argument('--max_url_len', nargs='?', type=int, default=100, help='num of images per website')

        parser.add_argument('--normalization',nargs='?',type=str,default='l2',help='way of normalization')
        parser.add_argument('--learning_rate', nargs='?', type=float, default=0.01, help='learning rateing')
        parser.add_argument('--dropout_keep_prob',nargs='?',type=float,default=0.8,help='dropout rate')
        parser.add_argument('--regularization_rate', nargs='?', type=float, default=0.01, help='l2 normlization rate')
        parser.add_argument('--data_path',nargs='?',type=str,default='data/train_data.record')
        parser.add_argument('--epoches',nargs='?',type=int,default=100)
        parser.add_argument('--save_interval',nargs='?',type=int,default=10)
        parser.add_argument('--test_interval',nargs='?',type=int,default=5)
        parser.add_argument('--valid_interval', nargs='?', type=int, default=5)

        parser.add_argument('--max_save',nargs='?',type=int,default=10)
        parser.add_argument('--model_path',nargs='?',type=str,default='model/my_model.ckpt') # TODO: tensorflow 模型怎么储存的？
        parser.add_argument('--model_path_y',nargs='?',type=str,default='model/my_model_y.ckpt')
        parser.add_argument('--num_data',nargs='?',type=int,default=8500)    # TODO : ,required=True
        parser.add_argument('--decay_rate',nargs='?',type=float,default=0.4,help='learning rate decay rate')
        parser.add_argument('--decay_step',nargs='?',type=int,default=8,help='learning rate decay step')
        parser.add_argument('--stair_case',nargs='?',type=bool,default=True,help='decay stair or smooth')

        parser.add_argument('--st_dimension',nargs='?',type=int,default=16,help='num of structure feature')

        parser.add_argument('--url_dimension', nargs='?', type=int, default=200, help='num of url characteres')
        parser.add_argument('--max_sentence_len', nargs='?', type=int, default=512,help="max num of words per sentence")
        parser.add_argument('--image_size',nargs='?',type=int,default=2048,help='image width or height')
        parser.add_argument('--url_vocab_size',nargs='?',type=int,default=88,help='url vocabulary size')
        parser.add_argument('--text_vocab_size',nargs='?',type=int,default=25034,help='text vocabulary size')


        parser.add_argument('--url_embedding_size',nargs='?',type=int,default=32,help='url embedding size')
        parser.add_argument('--text_embedding_size',nargs='?',type=int,default=32,help='text embedding size')
        parser.add_argument('--st_hidden_size', nargs='?', type=int, default=32, help='st final size')
        parser.add_argument('--image_hidden_size', nargs='?', type=int, default=32, help='image final size')


        parser.add_argument('--st_final_size', nargs='?', type=int, default=32, help='st final size')

        parser.add_argument('--url_final_size', nargs='?', type=int, default=32, help='url final size')
        parser.add_argument('--text_final_size', nargs='?', type=int, default=32, help='text final size')
        parser.add_argument('--image_final_size', nargs='?', type=int, default=32, help='image final size')

        parser.add_argument('--final_attention_size',nargs='?', type=int, default=16, help='final attention layer input vector dimension')




        parser.add_argument('--st_MLP_layer', nargs='?', type=int, default=1, help='url_lstm_layer')
        parser.add_argument('--url_lstm_layer', nargs='?', type=int, default=1, help='url_lstm_layer')
        parser.add_argument('--text_lstm_layer', nargs='?', type=int, default=2, help='text_lstm_layer')

        parser.add_argument('--learning_rate_decay_step', nargs='?', type=int, default=20, help='image_final_size')
        parser.add_argument('--pre_train',nargs='?',type=int,default=0, help="wheather train with trained model")
        parser.add_argument('--batch_normliaztion', nargs='?', type=bool, default=False, help="wheather train with trained model")
        parser.add_argument('--loss_r',nargs='?', type=float, default=0.001, help="weight of projection loss")
        self.parameters = parser.parse_args(args=args)


    def get_parameters(self):

        return self.parameters

    def get_parameter(self,key):
        return eval("self.parameters.{}".format(key))
