#!/usr/bin/env python
# -*- coding:utf-8 -*- 


import time
from visdom import Visdom
import requests
import os
import numpy as np

default_line_opt = dict(
                                xtickmin=0,
                                xtickmax=100,
                                xtickstep=10,
                                ytickmin=0,
                                ytickmax=1,
                                ytickstep=0.1,
                                markersymbol='dot',
                                markersize=5,
                            )
APPEND = 'append'
NEW = "new"

LINE = "line"
BOXPLOT = "boxplot"


class Visdomer(object):

    def __init__(self,server='http://0.0.0.0',port=8097,env="main"):
        self.viz = Visdom(server=server, port=port,env=env)
        assert self.viz.check_connection()
        self.viz.close()
        self.graphs = dict()
        self.type = [LINE,BOXPLOT]




    def update_graph(self,graph_name, data_name,data,type,update,opts=None):

        if self.graphs.get(graph_name) is not None:
            graph = self.graphs[graph_name]
            type = graph['type']
            if type == LINE:
                eval("self.viz.{}(X={},Y={},win=graph['graph'],update='{}',name='{}')".format(type, data["X"], data["Y"], update, data_name))
            elif type == BOXPLOT:
                X = data["X"]
                legend = data["legend"]
                eval("self.viz.{}(X=X,win=graph['graph'],opts=dict(legend=legend))".format(type))

        else:
            if type in self.type:

                if opts is None:
                    opt = eval("default_{}_opt".format(type))
                else:
                    opt = opts

                if type == LINE:
                    win = eval("self.viz.{}(X={},Y={},opts=opt,name='{}')".format(type, data["X"], data["Y"], data_name))
                elif type == BOXPLOT:
                    X = data["X"]
                    legend = data["legend"]
                    win = eval("self.viz.{}(X=X,opts=dict(legend={}))".format(type,legend))

                self.graphs[graph_name] = {"type": type, "graph": win}




