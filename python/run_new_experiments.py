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
import time, sys
from threading import Thread
from myUDT import UDT
import scipy.io

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
    
def load_crx():
    #读取mat数据
    mat = scipy.io.loadmat("../datasets/crx.mat")
    #除了最后一行
    x = mat['standdata'][:, :-1]
    #取最后一行
    y = np.asarray(mat['standdata'][:, -1], dtype=np.dtype(int))
    return (x,y)
    
def load_german():
    #读取mat数据
    mat = scipy.io.loadmat("../datasets/german.mat")
    #除了最后一行
    x = mat['standdata'][:, :-1]
    #取最后一行
    y = np.asarray(mat['standdata'][:, -1], dtype=np.dtype(int))
    return (x,y)
    

def run_experiment(experiment_name, x, y, clf, fold_idx = None):

    print( "{0}: {1}, {2}, {3}".format(experiment_name,
              x.shape[0],x.shape[1],len(set(y))) )
#    print( x[0] )
    
    starttime = time.clock()
    
    #scores =cross_validation.cross_val_score(clf, x, y, cv=10)
    skf = cross_validation.StratifiedKFold(y, n_folds=10, random_state=1992)
    scores = []
    cnt = -1
    for train_index, test_index in skf:
        cnt = cnt+1
        if ( None!=fold_idx and cnt!=fold_idx ):
            continue
        xTr, xTe = x[train_index], x[test_index]
        yTr, yTe = y[train_index], y[test_index]
        clf.fit(xTr, yTr)
        scores.append( clf.score(xTe, yTe) )  
    scores = np.asarray(scores)
    
    endtime = time.clock()
    print scores
    print ( 'mean is {0}, std is {1}'.format(scores.mean(),scores.std() ))
    print("used time {0} seconds".format( endtime-starttime ) )
    print("----------------------------------------------")
    with open("log/{0}.log".format(experiment_name), "w") as f:
        f.write( str(scores)+"\n" )
        f.write( 'mean is {0}, std is {1}\n'.format(scores.mean(),scores.std() ))
        f.write("used time {0} seconds\n".format( endtime-starttime ))
    
    result = np.concatenate(
        (scores,[scores.mean(),scores.std(),float(endtime-starttime)]))
    return result

        
if __name__ == '__main__':
#    (x,y) = load_crx()
#    (x,y) = load_german()
#    sys.exit(0)
    names = ['crx', 'german']
    names = ['crx']
    load_datas = {
            'crx':load_crx,
            'german':load_german}

    results = []
    if len(sys.argv)>=2:
        names = [sys.argv[1]]
        
    realnames = []
    for name in names:
        if not load_datas.has_key(name):
            raise( name + ' loading function doesn\'t implement!' )
        for bb in [10,15]:
            for w in [0.01]:
                (x,y) = load_datas[name]()
                fold_idx = int(sys.argv[2]) if len(sys.argv)>=3 else None
                #res = run_experiment(name, x, y, svm.SVC(kernel='linear',max_iter=5000) )
                tname = "{0}--w-{1}--bb-{2}".format(name, w, bb)
                realnames.append(tname)
                res = run_experiment(tname, x, y, UDT(w=w, bb=bb), fold_idx )
                #res = run_experiment(name, x, y, tree.DecisionTreeClassifier(), fold_idx )
                results.append(res)
                #print res
                #t = experiment(name, x, y, tree.DecisionTreeClassifier() )
    results = np.asarray(results)
    if len(sys.argv)<=2:
        results = pandas.DataFrame(data=results, index=realnames,
                             columns=[1,2,3,4,5,6,7,8,9,10,'mean','std','time(second)'])
        print(results)
    else:
        results = pandas.DataFrame(data=results, index=realnames,
                             columns=['result','mean','std','time(second)'])
        print(results)
            
    if len(sys.argv)==2:
        results.to_csv('log/{0}.csv'.format(sys.argv[1]) )
    elif len(sys.argv)==1:
        results.to_csv('log/result.csv')
    elif len(sys.argv)==3:
        results.to_csv('log/{0}_{1}.csv'.format(sys.argv[1], sys.argv[2]))
    