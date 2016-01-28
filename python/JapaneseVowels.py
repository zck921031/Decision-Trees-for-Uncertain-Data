# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:09:47 2016

@author: zck
"""

#from sklearn import cross_validation
from sklearn import tree
import numpy as np
from myDT_AVG import DT_AVG
from myUDT import UDT

def load_JapaneseVowels(folder):
    folder = folder+'/'
    xTr = []
    yTr = []
    xTe = []
    yTe = []
    
    # Train labels
    with open(folder+'size_ae.train') as f:
        sizeT = [int(x) for x in f.readline().strip().split()]
        for i in xrange( len(sizeT) ):
            for x in xrange( sizeT[i] ):
                yTr.append(i)
    
    # Test labels 
    with open(folder+'size_ae.test') as f:
        sizeT = [int(x) for x in f.readline().strip().split()]
        for i in xrange( len(sizeT) ):
            for x in xrange( sizeT[i] ):
                yTe.append(i)
    
    # Train feature
    with open(folder+'ae.train') as f:
       tp = []
       for line in f.readlines():
           lst = [float(x) for x in line.strip().split()]
           if 0==len(lst):
               xTr.append( np.asarray(tp).T )
               tp = []
           else:
               tp.append(lst)
               
    # Test feature
    with open(folder+'ae.test') as f:
       tp = []
       for line in f.readlines():
           lst = [float(x) for x in line.strip().split()]
           if 0==len(lst):
               xTe.append( np.asarray(tp).T )
               tp = []
           else:
               tp.append(lst)           
               
    return (xTr, np.asarray(yTr), xTe, np.asarray(yTe) )
    
    
def change_to_avg(_x):
    x = []
    for t in _x:
        x.append( t.mean(axis=1) )
    return np.asarray(x)


def change_to_pdf(data):
    N = len(data)
    M = len(data[0])
    pdf = [ [{}for j in xrange(M)] for i in xrange(N) ]
    for i in xrange(N):
        for j in xrange(M):
            c = 1.0 / len(data[i][j])
            for k in xrange(len(data[i][j])):
                pdf[i][j][data[i][j][k]] = pdf[i][j].get(data[i][j][k],0) + c
    return pdf
    
    
def run_AVG(_xTr, yTr, _xTe, yTe):
    xTr = change_to_avg(_xTr)
    xTe = change_to_avg(_xTe)
    for rng in xrange(55,56):
        clf = tree.DecisionTreeClassifier(random_state=rng)
        clf = clf.fit(xTr, yTr)
        acc = clf.score(xTe, yTe)
        print( "JapaneseVowels acc is {0}, random_state is {1}".format(acc,rng) )

def run_AVG_DT_paper(_xTr, yTr, _xTe, yTe):
    xTr = change_to_avg(_xTr)
    xTe = change_to_avg(_xTe)
    clf = DT_AVG()
    clf.fit(xTr, yTr)
    print( 'JapaneseVowels AVG acc is {0}'.format( clf.score(xTe, yTe) ) )

if __name__ == '__main__':
    (xTr, yTr, xTe, yTe) = load_JapaneseVowels('../datasets/JapaneseVowels-mld/')
#    xTr = xTr[:27]
#    yTr = yTr[:27]
#    xTe = xTe[:37]
#    yTe = yTe[:37]
    run_AVG_DT_paper(xTr, yTr, xTe, yTe)
    
    pdfTr = change_to_pdf(xTr)
    pdfTe = change_to_pdf(xTe)
    clf = UDT(max_depth=10, debug=True)
    clf.fit(pdfTr, yTr)
    pred = clf.predict(pdfTe)
    print( 'JapaneseVowels UDT acc is {0}'.format( clf.score(pdfTe, yTe) ) )
    
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(xTr, yTr)
#acc = clf.score(xTe, yTe)
    
    