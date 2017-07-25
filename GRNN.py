'''
=================================================================
程序说明：
1，基于python3.5.2实现广义回归神经网络,适合在线增量学习；
2，运行程序的库：Pandas，Numpy，matplotlib，xlsxwriter；
3，如果输出的向量的维数大于1，则需要更改图标的展示；
=================================================================
'''
print(__doc__)

'''++++++++++++++++++++第一部分：导入需要的库++++++++++++++++'''
import pandas as pd                    #读取数据
import numpy as np                     #程序运算
import matplotlib.pyplot as plt        #画图
import xlsxwriter as xw                #输出至excel文件

'''++++++++++++++++++++++第二部分：数据准备++++++++++++++++++++'''

MyPath=r'E:\4月14日工作数据\coding_NN_learning\广义回归神经网络'#数据文件存储路径

#训练数据
GRNN_Read_Train_Data=pd.read_excel(r'%s\GRNN_train.xlsx'%MyPath)
GRNN_Train_Data=GRNN_Read_Train_Data.values
GRNN_Train_Feature=np.array(GRNN_Train_Data[...,:-2],dtype=float)#训练特征数据[[1 2 3][4 5 6]...]
GRNN_Train_Target=np.array(GRNN_Train_Data[...,-2:],dtype=float)#训练数据[1 -1...]

#测试数据
GRNN_Read_Test_Data=pd.read_excel(r'%s\GRNN_test.xlsx'%MyPath)
GRNN_Test_Data=GRNN_Read_Test_Data.values
GRNN_Test_Feature=np.array(GRNN_Test_Data[...,:-2],dtype=float)#测试特征数据[[1 2 3][4 5 6]...]
GRNN_Test_Target=np.array(GRNN_Test_Data[...,-2:],dtype=float)#测试数据[1 -1...]


#预测数据
GRNN_Read_Predict_Data=pd.read_excel(r'%s\GRNN_predict.xlsx'%MyPath)
GRNN_Predict_Data=GRNN_Read_Predict_Data.values
GRNN_Predict_Feature=np.array(GRNN_Predict_Data[...,:-1],dtype=float)#预测特征数据
GRNN_Predict_Target=np.array(GRNN_Predict_Data[...,-1],dtype=float)#预测数据

'''
#定义处理数据的函数
def Normal_Data(tr_indata,tr_outdata,te_indata,te_outdata):
    #输入最大值
    max_intr=tr_indata.max(axis=0)
    max_inte=te_indata.max(axis=0)
    max_in=np.array([max_intr,max_inte]).max(axis=0)
    #输入最小值
    min_intr=tr_indata.min(axis=0)
    min_inte=te_indata.min(axis=0)
    min_in=np.array([min_intr,min_inte]).min(axis=0)

    #输出最大值
    max_outtr=tr_outdata.max(axis=0)
    max_outte=te_outdata.max(axis=0)
    max_out=np.array([max_outtr,max_outte]).max(axis=0)

    #输出最小值
    min_outtr=tr_outdata.min(axis=0)
    min_outte=te_outdata.min(axis=0)
    min_out=np.array([min_outtr,min_outte]).min(axis=0)
    
    return max_in,min_in,max_out,min_out

stand_data=Normal_Data(GRNN_Train_Feature,GRNN_Train_Target,GRNN_Test_Feature,GRNN_Test_Target)

print(stand_data)
#处理数据
def Deal_Data(exdata,maxex,minex):
    return (exdata-minex)/(maxex-minex)

#标准化以后的数据
trin_data=Deal_Data(GRNN_Train_Feature,stand_data[0],stand_data[1])
trout_data=Deal_Data(GRNN_Train_Target,stand_data[2],stand_data[3])

tein_data=Deal_Data(GRNN_Test_Feature,stand_data[0],stand_data[1])
teout_data=Deal_Data(GRNN_Test_Target,stand_data[2],stand_data[3])
    
prin_data=Deal_Data(GRNN_Predict_Feature,stand_data[0],stand_data[1])

'''

'''++++++++++++++++++++++第四部分：网络结构++++++++++++++++++++'''
#定义GRNN神经网络结构
def Grnn_Network(tr_indata,tr_outdata,te_indata,te_outdata,optina_para):
    #数据结构
    TrInOne=np.ones(len(te_indata)*len(tr_indata[0])).reshape(len(te_indata),1,len(tr_indata[0]))

    TeInOne=np.ones(len(tr_indata)*len(tr_indata[0])).reshape(len(tr_indata),len(tr_indata[0]))

    #通过乘积变为相同的数据结构
    Same_Tr=TrInOne*tr_indata
    Same_Te=TeInOne*(te_indata.reshape(len(te_indata),1,len(tr_indata[0])))
    
    #模式层
    Pattern_data=np.exp(-((((Same_Tr-Same_Te)**2).sum(axis=2))/2*(optina_para)**2))
    
    #求和层
    Direct_sum=Pattern_data.sum(axis=1)#直接求和
    Weight_sum=np.dot(Pattern_data,GRNN_Train_Target)#加权求和

    #输出层
    Output_num=Weight_sum/(Direct_sum.reshape(len(Direct_sum),1))

    #转化输出值正常数据量级

    #Normal_output=Output_num*(stand_data[2]-stand_data[3])+stand_data[3]

    #计算误差
    try:
        if len(te_outdata[0])!=1:
            Error=((Output_num-te_outdata)**2).sum()/len(te_outdata)*len(te_outdata[0])
    except TypeError:
        Error=(((Output_num.T)[0]-te_outdata)**2).sum()/len(te_outdata)

    return Error,Output_num


'''++++++++++++++++++++++第五部分：输出++++++++++++++++++++'''
#输出函数
def Ou_Grnn(tr_indata,tr_outdata,te_indata,te_outdata,max_num):
    dict_num={}
    dict_data={}
    for i_num in range(1,max_num):
        grnn_out=Grnn_Network(tr_indata,tr_outdata,te_indata,te_outdata,i_num/100)
        if np.isnan(grnn_out[0]):
            break
        dict_num[i_num]=grnn_out[0]
        dict_data[i_num]=grnn_out[1]
        try:
            if error_num>grnn_out[0]:
                error_num=grnn_out[0]
            else:
                del dict_data[i_num]
        except NameError:
            error_num=grnn_out[0]
        
    #选择最小的值
    minkey=min(dict_num.items(),key=lambda x:x[1])[0]

    #图示输出

    fig=plt.figure(figsize=(15,8))
    fig.subplots_adjust(wspace=0.6,hspace=0.9)
    

    test_target=(te_outdata.T)[0]
    test_output=(dict_data[minkey].T)[0]
    

    #测试数据实际值与输出值对比图
    ax3=fig.add_subplot(3,1,1)
    ax3.scatter(range(len(test_target)),test_target,c='g',alpha=1.0,label='Y_target',marker='o')
    ax3.plot(test_output,c='r',alpha=2.0,label='Y_output')
    
    ax3.set_title('The Contrast between Real and Output')
    ax3.set_xlabel('Sequence')
    ax3.set_ylabel('Data')
    ax3.legend(loc='upper left')

    #图
    ax2=fig.add_subplot(3,1,2)
    ax2.scatter(range(len(test_output-test_target)),test_output-test_target,facecolor='g',alpha=0.5)
    ax2.plot(range(len(test_target)),[0]*len(test_target),c='r',alpha=1)
    ax2.set_title('error')
   
    
    #误差图
    ax1=fig.add_subplot(3,1,3)
    ax1.plot(list(dict_num.keys()),list(dict_num.values()),c='red',alpha=2)
    ax1.set_title('ErrorNumber[最合适的参数为%.4f]'%(minkey/100))

    ax1.set_xlabel('Spread paramete')
    ax1.set_ylabel('Error Number')
        
    return minkey


para=Ou_Grnn(GRNN_Train_Feature,GRNN_Train_Target,GRNN_Test_Feature,GRNN_Test_Target,1000)

#预测数据
def Pre_Function(tr_indata,tr_outdata,pre_indata,pre_outdata,selected_para):
    pre_dict=Grnn_Network(tr_indata,tr_outdata,pre_indata,pre_outdata,selected_para)
    #图示输出

    fig=plt.figure(figsize=(15,8))
    fig.subplots_adjust(wspace=0.6,hspace=0.9)
    
    predict_output=(pre_dict[1].T)[0]

    print(predict_output)
    

    #测试数据实际值与输出值对比图
    ax13=fig.add_subplot(2,1,1)
    ax13.scatter(range(len(pre_outdata)),pre_outdata,c='g',alpha=1.0,label='Y_target',marker='o')
    ax13.plot(predict_output,c='r',alpha=2.0,label='Y_output')
    
    ax13.set_title('The Contrast between Real and Output')
    ax13.set_xlabel('Sequence')
    ax13.set_ylabel('Data')
    ax13.legend(loc='upper left')

    #图
    ax12=fig.add_subplot(2,1,2)
    ax12.scatter(range(len(pre_outdata-predict_output)),pre_outdata-predict_output,facecolor='g',alpha=0.5)
    ax12.plot(range(len(pre_outdata)),[0]*len(pre_outdata),c='r',alpha=1)
    ax12.set_title('error')
        
    plt.show()
    return 'END'

    

'''++++++++++++++++++++++++最终的程序++++++++++++++++++++++++++++++'''
import time

for ii in range(1,1000):
    print(Pre_Function(GRNN_Train_Feature,GRNN_Train_Target,GRNN_Predict_Feature,GRNN_Predict_Target,ii/100))
    time.sleep(1)





                
        
 
















