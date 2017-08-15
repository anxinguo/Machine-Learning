'''
E-mail: anxinguoguo@126.com
=================================================================
程序说明：
1，基于python3.5.2实现多元线性回归[局部加权回归主要用于预测]；
2，运行程序的库：Pandas,Numpy,matplotlib,xlsxwriter,prettytable；
3，回归方程形如 y=a1*x1+a2*x2+...+an*xn+b
4，成本函数：最小二乘法
5，优化求解方法：梯度[随机]下降法,(根据样本数自动选择)
=================================================================
'''
print(__doc__)

'''++++++++++++++++++++第一部分：导入需要的库++++++++++++++++'''
import pandas as pd                    #读取数据
import numpy as np                     #程序运算
import matplotlib.pyplot as plt        #画图
from prettytable import PrettyTable    #打印表格样式
import xlsxwriter as xw                #输出至excel文件
plt.ion()
'''++++++++++++++++++++++第二部分：数据准备++++++++++++++++++++'''
#数据文件存储路径(需要更改)
DataPath=r'E:\4月14日工作数据\coding_NN_learning\多元回归\data.xlsx'

#写入数据文件的路径
WriteDataPath=r'E:\4月14日工作数据\coding_NN_learning\多元回归\write.xlsx'

#求解参数设置
OplrPara={'StudyRate':0.5,'IterTimes':5000,'ErrorNumber':0.001}

sheetname='train'
#读取数据的函数
def ReadData(path,sheetname):
    data_source=pd.read_excel(path,sheetname)
    data_value=data_source.values
    name_data=data_source.keys()[1:-1]
    x_data=np.array(data_value[...,1:-1],dtype=float)
    y_data=np.array(data_value[...,-1],dtype=float)
    return x_data,y_data,name_data

data_oplr=ReadData(r'%s'%DataPath,'train')
x_factor=data_oplr[0]#自变量数据
y_target=data_oplr[1]#因变量数据
name_factor=data_oplr[2]#自变量名称数据

data_oplr_pre=ReadData(r'%s'%DataPath,'predict')
x_real=data_oplr_pre[0]#实际要预测的数据


yy_data=ReadData(r'%s'%DataPath,'predict')[1]

'''++++++++++++++++++++++第三部分：根据散点图以及相关系数判断是否可以使用线性回归++++++++++++++++++++'''
#计算皮尔逊相关系数的函数
def PearsonCoefficient(data1,data2):
    SSxy=((data1-np.mean(data1))*(data2-np.mean(data2))).sum()
    SSxx=((data1-np.mean(data1))**2).sum()
    SSyy=((data2-np.mean(data2))**2).sum()
    r_coefficient=SSxy/((SSxx*SSyy)**0.5)
    return r_coefficient


#展示变量间的皮尔逊系数以及散点图
dongVar=locals()
fig = plt.figure(figsize=(8,8))
fig.subplots_adjust(hspace=0.85,wspace=0.3)
for i_figure in range(len(x_factor.T)):
    dongVar['axes%s'%i_figure]=fig.add_subplot(len(x_factor.T),1,i_figure+1)
    pe_num=PearsonCoefficient(x_factor.T[i_figure],y_target)
    dongVar['axes%s'%i_figure].scatter(x_factor.T[i_figure],y_target,c='black',alpha=0.5)
    dongVar['axes%s'%i_figure].set_title(r'Scatter Figure $x\_factor%s$'%(i_figure+1))
    dongVar['axes%s'%i_figure].set_xlabel('x_factor%s'%(i_figure+1))
    dongVar['axes%s'%i_figure].set_ylabel('y_target')
    dongVar['axes%s'%i_figure].text(np.mean(x_factor.T[i_figure])-min(x_factor.T[i_figure]),\
                                    max(y_target)-min(y_target),r'$Pearson$ $cofficient$$=$$%.3f$'%pe_num)
plt.show()    

#为x_factor数据添加最后一列1
one=np.ones(len(y_target))
one1=np.ones(len(x_real))
x_factor=np.append(x_factor,one.reshape(len(one),1),axis=1)
x_real=np.append(x_real,one1.reshape(len(x_real),1),axis=1)

'''++++++++++++++++++++++第四部分：线性回归++++++++++++++++++++'''

#随机梯度下降函数
def StochasticGradientDescent(datax,datay,SR=OplrPara['StudyRate'],IT=OplrPara['IterTimes'],ET=OplrPara['ErrorNumber']):
    iter_times=0
    weight_num=np.zeros(len(datax[0]),dtype=float)
    error_list=[]
    while iter_times<IT:
        weight_num_copy=weight_num.copy()
        der_item=np.zeros(len(datax[0]),dtype=float)
        for i_sample in range(len(datay)):
            der_item+=((((datax*weight_num).sum(axis=1)-datay).sum())*datax[i_sample])/((datax**2).sum(axis=0))
        weight_num-=SR*der_item/len(datay)
        error_number=((datay-((datax*weight_num).sum(axis=1)))**2).sum()/(2*len(datay))
        iter_times+=1
        if error_number<=ET:
            break
        print(error_number)
        error_list.append(error_number)
        try:
            if error_list[-1]>error_list[-2]:
                SR*=0.8
                weight_num=weight_num_copy
        except IndexError:
            pass       
    print(weight_num)
    line_output='回归方程为y='
    for i_line in range(len(weight_num)):
        if  weight_num[i_line]>0:
            if i_line==len(weight_num)-1:
                line_output+='+%.5f'%(weight_num[i_line])
            elif i_line==0:
                line_output+='%.5f*%s'%(weight_num[i_line],name_factor[i_line])
            else:
                line_output+='+%.5f*%s'%(weight_num[i_line],name_factor[i_line])
        else:
            if i_line==len(weight_num)-1:
                line_output+='%.5f'%(weight_num[i_line])
            else:
                line_output+='%.5f*%s'%(weight_num[i_line],name_factor[i_line])
    print(line_output)
    
    return weight_num,error_list


#梯度下降法函数
def GradientDescent(datax,datay,SR=OplrPara['StudyRate'],IT=OplrPara['IterTimes'],ET=OplrPara['ErrorNumber']):
    iter_times=0
    weight_num=np.zeros(len(datax[0]),dtype=float)
    error_list=[]
    while iter_times<IT:
        weight_num_copy=weight_num.copy()
        der_item=(((datax*weight_num).sum(axis=1)-datay).reshape(len(datay),1)*datax).sum(axis=0)/(datax**2).sum(axis=0)
        
        weight_num-=SR*der_item
        iter_times+=1
        error_number=((datay-((datax*weight_num).sum(axis=1)))**2).sum()/(2*len(datay))
        if error_number<=ET:
            break
        print(error_number)
        error_list.append(error_number)
        try:
            if error_list[-1]>error_list[-2]:
                SR*=0.8
                weight_num=weight_num_copy
        except IndexError:
            pass       
    print(weight_num)
    line_output='回归方程为y='
    for i_line in range(len(weight_num)):
        if  weight_num[i_line]>0:
            if i_line==len(weight_num)-1:
                line_output+='+%.5f'%(weight_num[i_line])
            elif i_line==0:
                line_output+='%.5f*%s'%(weight_num[i_line],name_factor[i_line])
            else:
                line_output+='+%.5f*%s'%(weight_num[i_line],name_factor[i_line])
        else:
            if i_line==len(weight_num)-1:
                line_output+='%.5f'%(weight_num[i_line])
            else:
                line_output+='%.5f*%s'%(weight_num[i_line],name_factor[i_line])
    print(line_output)
    return weight_num,error_list


#局部加权梯度下降法函数
def WeightGradientDescent(datax,datay,x_sample,SR=OplrPara['StudyRate'],IT=OplrPara['IterTimes'],ET=OplrPara['ErrorNumber']):
    iter_times=0
    weight_num=np.zeros(len(datax[0]),dtype=float)
    error_list=[]
    while iter_times<IT:
        weight_num_copy=weight_num.copy()
        der_item=((((datax*weight_num).sum(axis=1)-datay).reshape(len(datay),1)*datax)*\
                  np.exp(-1*(((datax-x_sample)**2).sum(axis=1)).reshape(len(datay),1)/5000000000000)).sum(axis=0)\
                  /(datax**2).sum(axis=0)
        
        weight_num-=SR*der_item
        iter_times+=1
        error_number=((datay-((datax*weight_num).sum(axis=1)))**2).sum()/(2*len(datay))
        if error_number<=ET:
            break
        print(error_number)
        error_list.append(error_number)
    
        try:
            if error_list[-1]>error_list[-2]:
                SR*=0.8
                weight_num=weight_num_copy
        except IndexError:
            pass
    print(weight_num)      
    line_output='回归方程为y='
    for i_line in range(len(weight_num)):
        if  weight_num[i_line]>0:
            if i_line==len(weight_num)-1:
                line_output+='+%.5f'%(weight_num[i_line])
            elif i_line==0:
                line_output+='%.5f*%s'%(weight_num[i_line],name_factor[i_line])
            else:
                line_output+='+%.5f*%s'%(weight_num[i_line],name_factor[i_line])
        else:
            if i_line==len(weight_num)-1:
                line_output+='%.5f'%(weight_num[i_line])
            else:
                line_output+='%.5f*%s'%(weight_num[i_line],name_factor[i_line])
 
    return weight_num,error_list

#print(GradientDescent(x_factor,y_target,x_real[0])[0])
        
'''++++++++++++++++++++++第五部分：结果输出++++++++++++++++++++'''            
            
def OutResult(datax,datay):
    if len(datay)>800:
        out_result=StochasticGradientDescent(datax,datay)
    else:
        out_result=GradientDescent(datax,datay)
    return out_result
    
Out_Result=OutResult(x_factor,y_target)



#求方差
def FangCha(data):
    return((data-np.mean(data))**2).sum()/len(data)

def ModelOut(out_result):
    weight_num=out_result[0]
    error_list=out_result[1]
    y_output=(x_factor*weight_num).sum(axis=1)

    

    fig=plt.figure(figsize=(16,12))
    #fig.SubplotParams(wspace=0.6,hspace=0.9)
    fig.subplots_adjust(hspace=0.45,wspace=0.3)

    #实际值与输出值对比
    #ax3=fig.add_subplot(2,2,1)
    ax3=plt.subplot2grid((4,4),(0,0),colspan=3,rowspan=2)
    ax3.plot(y_target,c='g',alpha=1.0,label='Y_target',marker='o')
    ax3.plot(y_output,c='r',alpha=1.0,label='Y_output',marker='o')
    
    ax3.set_title('The Contrast between Real and Output')
    ax3.set_xlabel('Real x_data')
    ax3.set_ylabel('Real y_data')
    ax3.legend(loc='upper left')
    
    #成本函数图
    #ax1=fig.add_subplot(3,2,4)
    ax1=plt.subplot2grid((4,4),(2,0),colspan=2,rowspan=2)
    ax1.plot(error_list,c='red',alpha=2)
    ax1.set_title('Cost Function Number')
    ax1.set_xlabel('Iter_Times')
    ax1.set_ylabel('Cost Number')
        
    #误差图
    #ax2=fig.add_subplot(3,2,5)
    ax2=plt.subplot2grid((4,4),(2,2),colspan=2,rowspan=2)
    ax2.scatter(range(len(y_output-y_target)),y_output-y_target,facecolor='g',alpha=0.5)
    ax2.plot(range(len(y_target)),[0]*len(y_target),c='r',alpha=1)
    ax2.set_title('Error Scatter (ERRORSUM: %.4f)'%((y_output-y_target)**2).sum())
    ax2.set_xlabel('Error number')
    ax2.set_ylabel('Times')


    SST=FangCha(y_target)*len(y_target)
    SSR=FangCha(y_output)*len(y_target)
    SSE=FangCha(y_target-y_output)*len(y_target)
    R2=SSR/SST
    
    #模型验证参数
    #ax2=fig.add_subplot(6,2,4)
    left1,width=0.1,0.5
    bottom1,height=0.1,0.5
    ax4=plt.subplot2grid((4,4),(0,3),colspan=1,rowspan=2)
    ax4.text(left1,bottom1,r'决定系数 $R^{2}$=%.2f'%(R2))
    ax4.set_title('模型验证参数')
    ax4.set_xticks([])
    ax4.set_yticks([])

    plt.show()
    if R2>0.8:
        return '模型可用于预测'
    else:
        return '模型欠缺'


print(ModelOut(Out_Result))

'''++++++++++++++++++++++第六部分：预测结果展示++++++++++++++++++++'''

#将数据写入excel文件中
def WriteExcel(WriteDataPath):
    workbook=xw.Workbook(r'%s'%WriteDataPath)
    worksheet=workbook.add_worksheet('Predict_number')

    y_output1=(x_real*Out_Result[0]).sum(axis=1)
    worksheet.write(0,0,'预测样本标识')
    for j_predict in range(len(x_real)):
        worksheet.write(j_predict+1,0,'%s'%(j_predict+1))
        worksheet.write(j_predict+1,len(x_real[0]),'%.2f'%y_output1[j_predict])
        
        for i_sd in range(len(x_real[j_predict])-1):
            worksheet.write(j_predict+1,i_sd+1,'%s'%x_real[j_predict][i_sd])
    worksheet.write(0,len(x_real[0]),'预测Y值')

    for i_sdr in range(len(x_real[0])-1):
        worksheet.write(0,i_sdr+1,'x_factor%s'%(i_sdr+1))
        
    workbook.close()
    return '预测结果储存在%s'%(WriteDataPath)
    
print(WriteExcel(WriteDataPath))

'''++++++++++++++++++++++第七部分：图示++++++++++++++++++++'''

y_predict=(x_real*Out_Result[0]).sum(axis=1)
fig,axes=plt.subplots(figsize=(6,6))
axes1=plt.subplot(2,1,1)
axes1.plot(y_predict,c='g',alpha=1.0,label='Y_预测值',marker='o')
axes1.plot(yy_data,c='b',alpha=1.0,label='Y_目标值',marker='o')

axes2=plt.subplot(2,1,2)
axes2.scatter(range(len(y_predict-yy_data)),y_predict-yy_data,c='g')
axes2.plot([0]*len(y_predict),c='r')

plt.show()








    
