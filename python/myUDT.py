# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:09:47 2016

@author: zck
"""

from sklearn.tree import DecisionTreeClassifier
import math, operator
#import numpy as np

class Node:
    '''
    Represents a decision tree node.
    '''
    def __init__(self, parent = None, uid = -1):
        self.label = None # 结果类标签, 每个类别记录一个概率
        self.split_attr = None # 该结点的分裂属性ID
        self.split_value = None # 分裂值(左子树: <=value; 右子树: >value)
        self.chl = None # 左子树
        self.chr = None # 右子树
        self.parent = parent # 该结点的父亲结点
        self.uid = uid
    def printMe(self):
        if None==self.label:
            print('node {0} is not a leaf split_attr is {1} split_value is {2} childs is {3} {4}'.format(
                self.uid, self.split_attr, self.split_value, self.chl.uid, self.chr.uid))
        else:
            print('node {0} a leaf node, label is {1}'.format(self.uid, self.label) )
                
class UDT(DecisionTreeClassifier):
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
        super(UDT, self).__init__(
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
            
            
    def dict_norm(self, d):
        c = max(float(sum( d.values() )), 1e-50)
        return {k:v/c for (k,v) in d.items()}

    def split_label_prob(self, pdf, y, label_prob, a, v):
        """
        属性a的分裂点(<=, >)的值为v
        获得分裂后训练集左右子树各个实例的 类标记概率
        yes
        """
        label_prob_left = list(label_prob)
        label_prob_right = list(label_prob)
        N = len(pdf)
        for i in xrange(N):
            left = 0.0
            right = 0.0
            for (k,p) in pdf[i][a].items():
                if k<=v:
                    left += p
                else:
                    right += p
            lr = max(1e-50, left+right)
#            if not label_prob_left[i]/lr * left <= label_prob_left[i]:
#                print (lr, left, right)
#                raise('bug')
            label_prob_left[i] = label_prob_left[i]/lr * left
            label_prob_right[i] = label_prob_right[i]/lr * right
        return (label_prob_left, label_prob_right)

    def sum_y_label_prob(self, y, label_prob):
        """
        将所有label的概率按类别累加，返回每个类的总概率
        输入list, 返回dict
        yes
        """
        s = {}
        for i in xrange( len(y) ):
            s[y[i]] = s.get(y[i],0) + label_prob[i]
        return s
        
    def entropy(self, y, label_prob):
        """
        信息增益, label_prob是每个实例的类别概率
        yes
        """
        prob = self.dict_norm( self.sum_y_label_prob(y, label_prob) )
        #print(prob)
        ent=0
        for p_i in prob.values():
            ent -= p_i*math.log(max(p_i,1e-50), 2.0)
        #print ent
        return ent
        
    def info_gain(self, pdf, y, label_prob, a, v):
        """
        Decision Trees for Uncertain Data 里面的UDT方法
        属性a的分裂点(<=, >)的值为v
        yes
        """
        (label_prob_left, label_prob_right) = self.split_label_prob(pdf, y, label_prob, a, v)
        p_l = sum( label_prob_left )
        p_r = sum( label_prob_right )
        p_lr = max(1e-50, p_l+p_r)
        p_l_norm = p_l / p_lr
        p_r_norm = p_r / p_lr
        #print(p_l, p_r)
        infoA = 0
#        print(p_l, p_r, p_lr, p_l_norm, p_r_norm)
#        print(label_prob_left)
#        print(label_prob_right)
#        print(p_l_norm)
#        print(p_r_norm)
        infoA += p_l_norm*self.entropy(y,label_prob_left)
        infoA += p_r_norm*self.entropy(y,label_prob_right)
        #SplitInfoA = - p_l*math.log(p_l,2) - p_r*math.log((p_r),2)
        return -infoA

    def get_divide_points(self, pdf, attr):
        divide_points = []
        for i in xrange(len(pdf)):
            for k in pdf[i][attr]:
                divide_points.append(k)
        divide_points = sorted(set(divide_points))
        return divide_points
    
    def stop_condition(self, y, label_prob):
        cnt = 0
        for (k,v) in self.sum_y_label_prob(y, label_prob).items():
            if v>0:
                cnt += 1
        return True if cnt<=1 else False
        
    def build_tree(self, pdf, y, label_prob, threshold=0.0001, u=None, depth=0):
        #print('! a new node achieve')    
        N = len(pdf)
        M = len(pdf[0])
        if (None == u):
            u = Node(parent=None, uid=self.uid)            
        if ( 0==len(set(y)) ):
            raise('dataset can not be empty!')        
        if ( self.stop_condition(y, label_prob) or depth>self.max_depth ):
            u.label = self.dict_norm(self.sum_y_label_prob(y,label_prob))
            return u
            
        max_gain = -1e50
        best_attr = None
        best_value = None
        
        
        for attr in xrange(M):
            #print attr
            for value in self.get_divide_points(pdf, attr):
                #print( (attr, value) )
                temp_gain = self.info_gain(pdf, y, label_prob, attr, value)
                #print(attr, value, temp_gain)
                if temp_gain > max_gain:
                    max_gain = temp_gain
                    best_attr = attr
                    best_value = value
        
        #print(best_attr, best_value) # yes
        #return 
        
        if None==best_attr:
            u.label = self.dict_norm(self.sum_y_label_prob(y,label_prob))
            return u
        
        (label_prob_left, label_prob_right) = self.split_label_prob(
                            pdf, y, label_prob, best_attr, best_value)
        
#        print(self.uid, self.dict_norm(self.sum_y_label_prob(y,label_prob)) )
#        print( label_prob )
#        print( label_prob_left )
        
        if ( sum(label_prob_left)==0 or sum(label_prob_right)==0 ):
            u.label = self.dict_norm(self.sum_y_label_prob(y,label_prob))
            return u
            
        u.split_attr = best_attr
        u.split_value = best_value   
        #print(self.dict_norm(self.sum_y_label_prob(y,label_prob_right)))
        
        if self.debug:
            print('choose attr {0} to split value is {1} my size is {2}'.format(
                best_attr, best_value, len(pdf)) )
        
        # Recursion left child, 
        pdf_ch = [ [pdf[i][j] for j in xrange(M)] for i in xrange(N) ]
        for i in xrange(N):
            pdf_ch[i][best_attr] = {}
            for k in pdf[i][best_attr]:
                if k<=best_value:
                    pdf_ch[i][best_attr][k] = pdf[i][best_attr][k]
        pdf_ch_down = []
        y_down = []
        label_prob_left_down = []
        for i in xrange(len(pdf_ch)):
            if label_prob_left[i]>0:
                pdf_ch_down.append(pdf_ch[i])
                y_down.append(y[i])
                label_prob_left_down.append(label_prob_left[i]);
        self.uid += 1
        u.chl = self.build_tree(pdf_ch_down, y_down, label_prob_left_down, threshold,
                                Node(parent=u,uid=self.uid), depth+1 )  
                                
        # Recursion right child, 
        pdf_ch = [ [pdf[i][j] for j in xrange(M)] for i in xrange(N) ]
        for i in xrange(N):
            pdf_ch[i][best_attr] = {}
            for k in pdf[i][best_attr]:
                if k>best_value:
                    pdf_ch[i][best_attr][k] = pdf[i][best_attr][k]
        pdf_ch_down = []
        y_down = []
        label_prob_right_down = []
        for i in xrange(len(pdf_ch)):
            if label_prob_right[i]>0:
                pdf_ch_down.append(pdf_ch[i])
                y_down.append(y[i])
                label_prob_right_down.append(label_prob_right[i]);
        self.uid += 1
        u.chr = self.build_tree(pdf_ch_down, y_down, label_prob_right_down, threshold,
                                Node(parent=u,uid=self.uid), depth+1 )  
        
        #print(u.chl)
        #print(u.chr)
        #print(label_prob)      
        return u
        
                        
    def fit(self, pdf, _y, sample_weight=None, check_input=True):    
        #print(pdf)
        if None == self.max_depth:
            self.max_depth = 10000000
        y = list(_y)
        label_prob = [ 1 for k in y ] # 实例剩余概率
        #print(label_prob) # yes
        self.uid = 0 # 记录树节点的编号
        self.tree = self.build_tree(pdf, y, label_prob)
        
    def classify(self, pdf, tree, attr=0, prob=1): 
        if None != tree.label:
            label = {}
            #print( type(tree.label) )
            for x in tree.label:
                label[x] = tree.label[x]*prob              
            return label
        else:
            M = len(pdf)
            best_attr = tree.split_attr
            best_value = tree.split_value
            left = 0
            right = 0
            for k in pdf[best_attr]:
                if k<=best_value:
                    left  += pdf[best_attr][k]
                else:
                    right += pdf[best_attr][k]
            lr = max(1e-50, left+right)
            prob_left = prob/lr*left
            prob_right = prob/lr*right
            
            # Recursion left child, 
            pdf_ch = [pdf[j] for j in xrange(M)]
            pdf_ch[best_attr] = {}                
            for k in pdf[best_attr]:
                if k<=best_value:
                    pdf_ch[best_attr][k] = pdf[best_attr][k]
            label_left = self.classify(pdf_ch, tree.chl, best_attr, prob_left)            
                                    
            # Recursion right child, 
            pdf_ch = [pdf[j] for j in xrange(M)]
            pdf_ch[best_attr] = {}                
            for k in pdf[best_attr]:
                if k>best_value:
                    pdf_ch[best_attr][k] = pdf[best_attr][k]
            label_right = self.classify(pdf_ch, tree.chr, best_attr, prob_right)
            
            # Merge two children
            label={}
            for (k,v) in label_left.items():
                label[k] = label.get(k,0) + v
            for (k,v) in label_right.items():
                label[k] = label.get(k,0) + v
            return label
                
    def predict(self, pdf, check_input=True):
        all_proba = []
        for k in range( len(pdf) ):
            label_prob = self.classify(pdf[k], self.tree)
            label_tuple = max(label_prob.items(), key=operator.itemgetter(1))            
            all_proba.append( label_tuple[0] )
            #print(pdf[k])
            #print(label_prob)
        return all_proba
    
    def printMe(self, u=None):
        u.printMe()
        #print(u.label==None)
        if None == u.label:
            self.printMe(u.chl)
            self.printMe(u.chr)
        
        
if __name__ == '__main__':
    
    #clf = tree.DecisionTreeClassifier()
    #(xTr, yTr, xTe, yTe) = load_JapaneseVowels('../datasets/JapaneseVowels-mld/')
    
    pdf = [ [{-1.0:8.0/11.0, 10.0:3.0/11.0}],
            [{-10.0:1.0/9.0, -1.0:8.0/9.0}],
            [{-1.0:5.0/8.0,   1.0:1.0/8.0, 10.0:2.0/8.0}],
            [{-10.0:5.0/19.0,-1.0:1.0/19.0, 1.0:13.0/19.0}],
            [{0.0:1.0/35.0,   1.0:30.0/35.0, 10.0:4.0/35.0}],
            [{-10.0:3.0/11.0, 1.0:8.0/11.0}]]
            
#    pdf = [ [{-1.0:8.0/11.0, 10.0:3.0/11.0},{0:1}],
#            [{-10.0:1.0/9.0, -1.0:8.0/9.0},{0:1}],
#            [{-1.0:5.0/8.0,   1.0:1.0/8.0, 10.0:2.0/8.0},{0:1}],
#            [{-10.0:5.0/19.0,-1.0:1.0/19.0, 1.0:13.0/19.0},{0:1}],
#            [{0.0:1.0/35.0,   1.0:30.0/35.0, 10.0:4.0/35.0},{0:1}],
#            [{-10.0:3.0/11.0, 1.0:8.0/11.0},{0:1}]]
            
    y = [0,0,0,1,1,1]
    clf = UDT(debug=True)
    clf.fit(pdf, y)
    pred = clf.predict(pdf)
    print( 'Sample UDT acc is {0}'.format( clf.score(pdf, y) ) )
    clf.printMe(clf.tree)
    #print( 'JapaneseVowels UDT acc is {0}'.format( clf.score(xTe, yTe) ) )
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