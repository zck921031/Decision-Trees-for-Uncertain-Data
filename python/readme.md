#Decision Trees for Uncertain Data
>实现论文[Decision Trees for Uncertain Data](http://www2.informatik.uni-freiburg.de/~danlee/publications/uclass-tkde.pdf)的算法，对不确定数据建立决策树。
>
>不确定数据的意思是，样本的特征用概率密度函数(<b>P</b>robability <b>D</b>ensity <b>F</b>unction)表示，论文采用频率分布直方图实现。
>
##Paradigm
>需要python运行环境及numpy sklearn等依赖包，我使用的是[anaconda python2.7](https://www.continuum.io/downloads)环境
>
>实现的算法提供相同的训练和测试方法。

1. 实例化一个分类器: clf = UDT()
2. 训练决策树: clf.fit(xTrain, yTrain)
3. 用决策树做预测: pred = clf.predict(xTest)
>甚至可以是

* 10-fold交叉验: scores = cross_validation.cross_val_score(clf, xTrain, yTrain, cv=10)
* 验证测试集精度: clf.score(xTest, yTest)

##Averaging
###数据格式
>给定N个M维样本数据，xTrain,xTest是NxM的2-D numpy-array作为特征，yTrain,yTest是N行的1-D numpy-array作为类别。
###训练
>
>clf.fit(xTrain, yTrain)
>
###预测
>pred = clf.predict(xTest)
>
###算法
>传统的C4.5决策树，规则按照论文的公式实现。
>
##Uncertain Decision Trees(UDT)
###数据格式
>给定N个M维样本数据，xTrain,xTest是NxM的2-D list，每个list用一个dict记录频率分布直方图(概率分布)，例如论文的样例写成如下形式:  
>>          pdf = [ [{-1.0:8.0/11.0, 10.0:3.0/11.0}],
            [{-10.0:1.0/9.0, -1.0:8.0/9.0}],
            [{-1.0:5.0/8.0,   1.0:1.0/8.0, 10.0:2.0/8.0}],
            [{-10.0:5.0/19.0,-1.0:1.0/19.0, 1.0:13.0/19.0}],
            [{0.0:1.0/35.0,   1.0:30.0/35.0, 10.0:4.0/35.0}],
            [{-10.0:3.0/11.0, 1.0:8.0/11.0}]]
>
>yTrain,yTest是N行的1-D list作为类别。
###训练
>
>clf.fit(xTrain, yTrain)
>
###预测
>pred = clf.predict(xTest)
>
###算法
>论文的算法，与C4.5类似。

###demo
>python iris.py
>
>python JapaneseVowels.py


##Experiment Result

| Data Set       | AVG      |  UDT    |
| --------       | -----:   | :----:  |
| JapaneseVowels | 0.8162   | 0.9324(max_depth=8)  0.9405(max_depth=10)  |
| iris           |   0.96   |   NA    |