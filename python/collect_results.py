# -*- coding: utf-8 -*-
"""
Created on Thu Feb 04 09:32:11 2016

@author: zck
"""

import pandas
import numpy as np

if __name__=="__main__":    
    names = ['Banknote', 'Bench', 'Glass', 'Heart', 'Iris',
             'Leaf', 'Seeds', 'Wine', 'Winequality', 'Yeast']
    results = []
    for name in names:
        try:
            t = pandas.read_csv('log/{0}.csv'.format(name),sep=',').as_matrix()
            results.append( t[0,1:] )
        except:
            try:
                res = np.linspace( 0, 0, 13)
                for i in xrange(10):
                    t = pandas.read_csv('log/{0}_{1}.csv'.format(name,i),sep=',').as_matrix()
                    res[i] = t[0,1]
                    res[-1] += t[0,-1]
                res[-3] = res[:10].mean()
                res[-2] = res[:10].std()
                results.append(res)
            except:
                raise('can\'t find {0} result.'.format(name) ) 
    results = pandas.DataFrame(data=results, index=names,
                             columns=[1,2,3,4,5,6,7,8,9,10,'mean','std',
                             '10-fold total time(second)'])
    print(results)
    results.to_csv('log/result.csv')
#    t = pandas.read_csv("../datasets/Wine-quality/winequality-red.csv",
#                        sep=',').as_matrix()
