# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:09:47 2016

@author: zck
"""

from sklearn import cross_validation
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import math
import numpy as np

class Node_C45:
    '''
    Represents a decision tree node.
    '''
    def __init__(self, parent = None, uid = -1):
        self.label = None # 结果类标签
        self.split_attr = None # 该结点的分裂属性ID
        self.split_value = None # 分裂值(左子树: <=value; 右子树: >value)
        self.chl = None # 左子树
        self.chr = None # 右子树
        self.parent = parent # 该结点的父亲结点        
        self.uid = uid
    def printMe(self):
        if None==self.label:
            print('this is not a leaf split_attr is {0} split_value is {1}'.format(
                self.split_attr, self.split_value))
        else:
            print('this is a leaf node, label is {0}'.format(self.label) )
                
class DT_AVG(DecisionTreeClassifier):
    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 class_weight=None,
                 debug=False):
        self.tree = None
        self.debug = debug
        super(DT_AVG, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state)
            
    def get_major_label(self, label):
        '''
        获取数据集中实例数最大的类别
        '''
        count = {}
        result = None
        for t in label:
            count[t] = count.get(t, 0) + 1
        max_count = 0
        for key, value in count.items():
            if (value > max_count):
                max_count = value
                result = key
        return result
        
    def entropy(self, y):
        """
        信息增益, y是类别集合        
        """
        ent=0
        ly = list(y)
        for k in set(y):
            p_i = float(ly.count(k))/len(y)
            ent -= p_i*math.log(p_i+1e-50, 2.0)
        return ent
        
    def info_gain(self, data, label, a, v):
        """
        Decision Trees for Uncertain Data 里面的方法
        """
        t_l = np.where(data[:,a]<=v)
        t_r = np.where(data[:,a]>v)
        p_l = len(t_l[0])*1.0/len(label)
        p_r = 1 - p_l
        #p_l += 1e-50
        #p_r += 1e-50        
        #info = self.entropy(label)
        infoA = p_l*self.entropy(label[t_l]) + p_r*self.entropy(label[t_r])        
        #SplitInfoA = - p_l*math.log(p_l,2) - p_r*math.log((p_r),2)
        #return infoA
        return -infoA
        #return info - infoA
        #print( (info - infoA)/SplitInfoA )
        #return (info - infoA)/SplitInfoA
    
    def build_tree(self, data, label, threshold = 0.0001, u = None):
        #print('! a new node achieve')
        if (None == u):
            u = Node_C45(None, self.uid)
            
        if ( 0==len(set(label)) ):
            raise('dataset can not be empty!')
            
        if ( 1==len(set(label)) ):
            u.label = label[0]
            return u
        
        max_gain = -1e50
        best_attr = None
        best_value = None
        
        for attr in xrange(data.shape[1]):
            for value in set(data[:,attr]):
                temp_gain = self.info_gain(data, label, attr, value)
                if temp_gain > max_gain:
                    max_gain = temp_gain
                    best_attr = attr
                    best_value = value
        
        if None==best_attr:
            u.label = self.get_major_label(label)
            return u
            
        t_l = np.where(data[:,best_attr] <= best_value)
        t_r = np.where(data[:,best_attr] >  best_value)
        data_l = data[t_l]
        data_r = data[t_r]
        label_l = label[t_l]
        label_r = label[t_r]
        if ( 0==len(label_l) or 0==len(label_r) ):            
            u.label = self.get_major_label(label)
            return u
        
        if self.debug:
            print('choose attr {0} to split value is {1} my size is {2}'.format(
                best_attr, best_value, len(data)) )
        u.split_attr = best_attr
        u.split_value = best_value
        self.uid += 1
        u.chl = self.build_tree(data_l, label_l, threshold, Node_C45(parent=u,uid=self.uid) )
        self.uid += 1
        u.chr = self.build_tree(data_r, label_r, threshold, Node_C45(parent=u,uid=self.uid) )
        
        #print(u.chl)
        #print(u.chr)
        #print(u.label)
        
        return u
        
    def fit(self, X, y, sample_weight=None, check_input=True):
        #print(X)
        self.uid = 0 # 记录树节点的编号
        self.tree = self.build_tree(X, y)
        #self.printMe( self.tree )
    def classify(self, x, tree):
        if None != tree.label:
            return tree.label
        else:
            if x[tree.split_attr] <= tree.split_value:
                return self.classify(x, tree.chl)
            else:
                return self.classify(x, tree.chr)
                
    def predict(self, X, check_input=True):
        all_proba = []
        for k in range( len(X) ):
            all_proba.append( self.classify(X[k,:], self.tree) )
        return all_proba
    
    def printMe(self, u=None):
        u.printMe()
        #print(u.label==None)
        if None == u.label:
            self.printMe(u.chl)
            self.printMe(u.chr)
        
        
if __name__ == '__main__':
    iris = load_iris()
    #clf = tree.DecisionTreeClassifier()
    clf = DT_AVG()
    scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=10)
    score = scores.mean()
    print( 'iris C4.5 10-fold acc is {0}'.format( score ) )
    #x = np.asarray([2,-2,2,-2,2,-2.0])
    #x = x.reshape([6,1])
    #y = np.asarray([0,0,0,1,1,1])
    #
    #clf = DT_C45()
    #clf.fit(x, y)
    #pred = clf.predict(x)
    
    #xTr,xTe = cross_validation.train_test_split(iris.data, test_size = 0.2, random_state=100)
    #yTr,yTe = cross_validation.train_test_split(iris.target, test_size = 0.2, random_state=100)
    #
    #clf = tree.DecisionTreeClassifier()
    #clf = clf.fit(xTr, yTr)
    #answer = clf.score(xTe, yTe)