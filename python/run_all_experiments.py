# -*- coding: utf-8 -*-
"""
Created on Mon Feb 01 14:26:51 2016

@author: zck
"""

from sklearn import cross_validation
from sklearn import svm, tree
import sklearn.datasets
import pandas
import numpy as np
import time
from threading import Thread
from myUDT import UDT

def mapy(y_t):
    h = {}
    cnt = 0
    for t in set(y_t):
        h[t] = cnt
        cnt += 1
    N = len(y_t)
    y = []
    for i in xrange(N):
        y.append(h[y_t[i]])
    return np.asarray(y, dtype=int)
    
def load_Banknote():
    t = pandas.read_csv("../datasets/Banknote/data_banknote_authentication.txt",
                        header=None).as_matrix()
    x = t[:, 0:-1]
    y = t[:, -1].astype(int)
    return (x,y)

def load_Bench():
    t = pandas.read_csv("../datasets/connectionist-bench/vowel-context.data",
                        header=None,
                        delimiter=' ', skipinitialspace=True).as_matrix()
    x = t[:, 3:13]
    y = t[:, 13].astype(int)
    return (x,y)

def load_Glass():
    t = pandas.read_csv("../datasets/glass/glass.data",
                        header=None).as_matrix()
    x = t[:, 1:10]
    y = mapy( t[:, 10].astype(int) )
    return (x, y)

def load_Heart():
    t = pandas.read_csv("../datasets/Heart/Heart.dat",
                        header=None,
                        delimiter=' ', skipinitialspace=True).as_matrix()
    x = t[:, 0:13]
    y = t[:, 13].astype(int)
    return (x,y)

def load_Iris():
    iris = sklearn.datasets.load_iris()
    return  (iris.data, iris.target)

def load_Leaf():
    t = pandas.read_csv("../datasets/Leaf/leaf.csv",
                        header=None).as_matrix()
    x = t[:, 2:]
    y = mapy(t[:, 0].astype(int))        
    return (x,y)
    
def load_Seeds():
    t = pandas.read_csv("../datasets/Seeds/seeds.csv",
                        header=None).as_matrix()
    x = t[:, :-1]
    y = mapy(t[:, -1].astype(int))
    return (x,y)
    
def load_Wine():
    t = pandas.read_csv("../datasets/wine/wine.data",
                        header=None).as_matrix()
    x = t[:, 1:]
    y = mapy(t[:, 0].astype(int))
    return (x,y)

def load_Winequality():
    t1 = pandas.read_csv("../datasets/Wine-quality/winequality-red.csv",
                        sep=';').as_matrix()
    t2 = pandas.read_csv("../datasets/Wine-quality/winequality-white.csv",
                        sep=';').as_matrix()
    t = np.vstack((t1,t2))               
    x = t[:, :-1]
    y = mapy(t[:, -1].astype(int))
    return (x,y)
    
def load_Yeast():
    t = pandas.read_csv("../datasets/yeast/yeast.data",
                        header=None,
                        delimiter=' ', skipinitialspace=True).as_matrix()
    x = t[:,1:9]
    y = mapy(t[:,9])
    return (x,y)
    

def run_experiment(experiment_name, x, y, clf):
    print( "{0}: {1}, {2}, {3}".format(experiment_name,
              x.shape[0],x.shape[1],len(set(y))) )
#    print( x[0] )    
    starttime = time.clock()
    
    scores =cross_validation.cross_val_score(clf, x, y, cv=10)
    
    endtime = time.clock()
    print scores
    print ( 'mean is {0}, std is {1}'.format(scores.mean(),scores.std() ))
    print("used time {0} seconds".format( endtime-starttime ) )
    print("----------------------------------------------")
    return scores

class experiment(Thread):
    def __init__(self, experiment_name, x, y, clf):
        self.experiment_name = experiment_name
        self.x = x
        self.y = y
        self.clf = clf
        super(experiment, self).__init__()
    def run(self):
        experiment_name = self.experiment_name
        x = self.x
        y = self.y
        clf = self.clf
        print( "{0}: {1}, {2}, {3}".format(experiment_name,
                  x.shape[0],x.shape[1],len(set(y))) )
    #    print( x[0] )    
        starttime = time.clock()
        
        scores =cross_validation.cross_val_score(clf, x, y, cv=10)
        
        endtime = time.clock()
        print scores
        print ( 'mean is {0}, std is {1}'.format(scores.mean(),scores.std() ))
        print("used time {0} seconds".format( endtime-starttime ) )
        print("----------------------------------------------")
        with open("log/{0}.log".format(experiment_name), "w") as f:
            f.write( str(scores)+"\n" )
            f.write( 'mean is {0}, std is {1}\n'.format(scores.mean(),scores.std() ))
            f.write("used time {0} seconds\n".format( endtime-starttime ))
        
        self.result = np.concatenate(
            (scores,[scores.mean(),scores.std(),float(endtime-starttime)]))
        
if __name__ == '__main__':
    names = ['Banknote', 'Bench', 'Glass', 'Heart', 'Iris',
             'Leaf', 'Seeds', 'Wine', 'Winequality', 'Yeast']
    load_datas = {'Banknote':load_Banknote,
            'Bench':load_Bench,
            'Glass':load_Glass,
            'Heart':load_Heart,
            'Iris':load_Iris,
            'Leaf':load_Leaf,
            'Seeds':load_Seeds,
            'Wine':load_Wine, 
            'Winequality':load_Winequality,
            'Yeast':load_Yeast}
    results = []
    threads = []
    for name in names:
        if not load_datas.has_key(name):
            raise( name + ' loading function doesn\'t implement!' )
        (x,y) = load_datas[name]()
        #res = run_experiment(name, x, y, svm.SVC(kernel='linear',max_iter=5000) )
        #res = run_experiment(name, x, y, tree.DecisionTreeClassifier() )
        #scores.append(res)
        #t = experiment(name, x, y, tree.DecisionTreeClassifier() )
        t = experiment(name, x, y, UDT() )
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
        results.append(t.result)
    results = np.asarray(results)
    results = pandas.DataFrame(data=results, index=names,
                         columns=[1,2,3,4,5,6,7,8,9,10,'mean','std','time(second)'])
    print(results)
    results.to_csv('log/result.log')
    