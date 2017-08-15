'''
E-mail: anxinguoguo@126.com
=================================================================
程序说明：
1，基于python3.5.2实现逻辑回归算法【线性可分数据】；
2，运行程序的库：Pandas,Numpy,matplotlib,xlsxwriter；
=================================================================
'''
print(__doc__)

'''++++++++++++++++++++第一部分：导入需要的库++++++++++++++++'''
import pandas as pd                    #读取数据
import numpy as np                     #程序运算
import matplotlib.pyplot as plt        #画图
import xlsxwriter as xw                #输出至excel文件

'''++++++++++++++++++++第二部分：数据准备++++++++++++++++'''

Txt_Path=r'E:\4月14日工作数据\coding_NN_learning\逻辑回归分类\money.txt'

def Read_Txt(txtpath):
    typedata=[]#类别数据
    alldata=[]#变量数据
    readtxt=open(txtpath,'r')
    for lines in readtxt.readlines():
        sigdata=[]
        for isplit in lines.split(','):
            sigdata.append(isplit.replace('\n',''))
        typedata.append(sigdata[-1])#最后一行类别数据
        alldata.append(sigdata[0:-1])
        
    #变换形式
    alldata=np.array(alldata[1:],dtype=float)
    typedata=np.array(typedata[1:],dtype=int)
    readtxt.close()
    return alldata,typedata


Data_Vary=Read_Txt(Txt_Path)[0]
Data_Type=Read_Txt(Txt_Path)[1]

'''++++++++++++++++++++第二部分：叁数设置++++++++++++++++'''
Tra_Percent=0.7  #数据用途分类百分比
Learn_lv=0.2    #学习率
Init_WB=np.ones((len(Data_Vary[0])+1,1))  #初始化w和b的值
Iterate_Times=5000
Degree=2e-8


'''++++++++++++++++++++第三部分：数据处理++++++++++++++++'''

#定义剩余编号的函数
def Get_Node(alli,exli):
    return [inode for inode in list(range(alli)) if inode not in exli]
     
#定义随机选择数据的函数
def Random_Select(datava,dataty,tr=Tra_Percent):
    #变量与类别数据结合
    datajoin=np.hstack((datava,np.array([dataty]).reshape(len(dataty),1)))
    #数据长度
    countva=len(datajoin[datajoin[:,-1]==0])
    seleva=int(countva*tr)+1
    countty=len(datajoin[datajoin[:,-1]==1])
    selety=int(countty*tr)+1
    #随机选取数据的编号
    randlistva=list(np.random.choice(countva,seleva,replace=False))
    randlistty=list(np.random.choice(countty,selety,replace=False))
    #分离数据
    zerodata=datajoin[datajoin[:,-1]==0]
    onedata=datajoin[datajoin[:,-1]==1]
    #训练数据
    trainzero=zerodata[randlistva]
    trainone=onedata[randlistty]
    #测试数据
    testzero=zerodata[Get_Node(countva,randlistva)]
    testone=onedata[Get_Node(countty,randlistty)]
    #结合数据
    jointestva=np.vstack((trainzero,trainone))
    jointestty=np.vstack((testzero,testone))    
    return jointestva,jointestty


#处理后的数据
Train_Data=Random_Select(Data_Vary,Data_Type)[0]
       
Test_Data=Random_Select(Data_Vary,Data_Type)[1]

'''++++++++++++++++++++第四部分：逻辑回归实现++++++++++++++++'''
   
#定义实现逻辑回归的函数
def Logistic_Regression(datr,initwb=Init_WB,learnlv=Learn_lv,maxtimes=Iterate_Times,deac=Degree):
    #数据顺序打乱
    np.random.shuffle(datr)
    #变量数据与类别数据
    varyable=datr[:,0:-1]
    classlast=datr[:,-1]

    #变量数据添加1
    varylast=np.hstack((varyable,np.ones((len(varyable),1))))

    #迭代次数
    itertimes=0

    #精度标量
    optiac=deac+1

    #存储精度的list
    saveac=[]
    
    #开始循环
    while itertimes<maxtimes or optiac>deac:#精度和次数必须同时达到
        #迭代次数加1
        itertimes+=1

        #最大似然函数的值
        maxlikehood=((np.log(1/(1+np.exp(-np.dot(varylast,Init_WB))))*(classlast.reshape(len(classlast),1))\
                     +np.log((np.exp(-np.dot(varylast,Init_WB)))/(1+np.exp(-np.dot(varylast,Init_WB))))\
                     *(1-classlast.reshape(len(classlast),1))).sum())*(-1/(len(varylast)))

        #计算梯度
        wdirection=((((classlast.reshape(len(classlast),1))-(1/(1+np.exp(-np.dot(varylast,Init_WB)))))*varyable).sum(axis=0))*(-1/(len(varylast)))

        bdirection=(((classlast.reshape(len(classlast),1))-(1/(1+np.exp(-np.dot(varylast,Init_WB))))).sum())*(-1/(len(varylast)))

        sumdirection=np.array(list(wdirection)+[bdirection])

        #更改W和B值
        initwb-=(learnlv*sumdirection).reshape(len(sumdirection),1)

        #保存最值
        saveac.append(maxlikehood)

        #精度的判断
        if len(saveac)>=2:
            optiac=abs(saveac[-2]-saveac[-1])

        print(maxlikehood)
          
        
    return initwb
    #初始化参数
    
Weight_Num=Logistic_Regression(Train_Data)


'''++++++++++++++++++++第五部分：逻辑回归测试++++++++++++++++'''

#逻辑回归验证函数
def Test_LR(datate,weight):
    classresult=1/(1+np.exp(-np.dot(datate[:,0:-1],weight[0:-1])+weight[-1]))
    #转换后的类别
    classresult[classresult>0.5]=1
    classresult[classresult<=0.5]=0

    #计算正确率
    corretlv=len(classresult[classresult==(datate[:,-1].reshape(len(datate),1))])/len(datate)

    print('测试正确率为%.3f%%'%(corretlv*100))
    return 
    
    
print(Test_LR(Test_Data,Weight_Num))
















































