# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:09:47 2016

@author: zck
"""

from sklearn import cross_validation
from sklearn.datasets import load_iris
#from sklearn.tree import DecisionTreeClassifier
from myUDT import UDT
from myDT_AVG import DT_AVG

def change_to_pdf_single(data):
    (N,M) = data.shape
    return [ [{data[i][j]:1.0} for j in xrange(M)] for i in xrange(N) ]
if __name__ == '__main__':
    iris = load_iris()
    xTr, xTe, yTr, yTe = cross_validation.train_test_split(
        iris.data, iris.target, test_size=0.1, random_state=692)
    #clf = DecisionTreeClassifier()
    clf = DT_AVG(debug=True)
    clf.fit(xTr, yTr)
    acc = clf.score(xTe, yTe)
    print( "iris DT acc is {0}".format(acc) )

    pdfTr = change_to_pdf_single(xTr)
    pdfTe = change_to_pdf_single(xTe)
    clf = UDT(debug=True)
    clf.fit(pdfTr, yTr)
    acc = clf.score(pdfTe, yTe)
    print( "iris UDT acc is {0}".format(acc) )
#xTr,xTe = cross_validation.train_test_split(iris.data, test_size = 0.2, random_state=100)
#yTr,yTe = cross_validation.train_test_split(iris.target, test_size = 0.2, random_state=100)
    
    clf = DT_AVG()
    print( "iris DT 10-forld acc is {0}".format(
        (cross_validation.cross_val_score(clf, iris.data, iris.target, cv=10)).mean() ) )

    clf = UDT()
    print( "iris UDT 10-forld acc is {0}".format(
        (cross_validation.cross_val_score(clf, iris.data, iris.target, cv=10)).mean() ) )

#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(xTr, yTr)
#answer = clf.score(xTe, yTe)