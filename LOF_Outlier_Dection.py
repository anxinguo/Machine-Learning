'''
=================================================================
程序说明：
1，基于python3.5.2实现利用LOF进行异常值检测【根据箱线图标准以及多k值交叉验证】；
2，运行程序的库：Pandas,Numpy,matplotlib,xlsxwriter；
3，新数据加入需要重新计算；
=================================================================
'''
print(__doc__)

'''++++++++++++++++++++第一部分：导入需要的库++++++++++++++++'''
import pandas as pd                    #读取数据
import numpy as np                     #程序运算
import matplotlib.pyplot as plt        #画图
import xlsxwriter as xw                #输出至excel文件

'''++++++++++++++++++++++第二部分：数据准备++++++++++++++++++++'''

#数据文件存储路径
DataPath=r'E:\4月14日工作数据\coding_NN_learning\LOF异常值检测算法\LOF_train.xlsx'

#写入数据文件的路径
WriteDataPath=r'E:\4月14日工作数据\coding_NN_learning\LOF异常值检测算法\LOF_write.xlsx'

#读取数据的函数
def ReadData(path):
    data_source=pd.read_excel(path)
    header_name=list(data_source.keys())
    data_value=data_source.values
    code_data=np.array(data_value[...,0:1])
    num_data=np.array(data_value[...,1:],dtype=float)
    return code_data,num_data,header_name

code_LOF=(ReadData(r'%s'%DataPath)[0].T)[0]#数据编号
                       
data_lof=ReadData(r'%s'%DataPath)[1]#数据源

HEADER=ReadData(r'%s'%DataPath)[2]+['LOF_num','OUT_judge']#抬头

#定义数据处理的函数
def Normal_data(data_set):
    mean_data=data_set.mean(axis=0)
    std_data=data_set.std(axis=0)
    return (data_set-mean_data)/std_data

data_LOF=Normal_data(data_lof)


'''++++++++++++++++++++++第三部分：参数设置++++++++++++++++++++'''

K_NUM=20 #k邻域的参数

'''++++++++++++++++++++++第四部分：LOF算法实现++++++++++++++++++++'''

#定义欧式距离的函数
def EuclideanDistance(sigle_data,set_data):
    if set_data.ndim==2:
        ed=((set_data-sigle_data)**2).sum(axis=1)**0.5
    else:
        ed=((set_data-sigle_data)**2).sum()**0.5
    return ed

#定义选择K个紧邻数据的函数
def SelectK(edlist,namelist,knum):
    ed_dict={namelist[edcode]:edlist[edcode] for edcode in range(len(namelist))}
    sorted_dict=sorted(ed_dict.items(),key=lambda x:x[1],reverse=False)
    #包含的元素集合
    K_inset=[]
    #根据数据判断
    if sorted_dict[knum][1]<sorted_dict[knum+1][1]:
        K_inset=[sorted_dict[i_k][0] for i_k in range(knum+1) if sorted_dict[i_k][1]!=0]
          
    elif sorted_dict[knum][1]==sorted_dict[knum+1][1]:
        K_inset=[]
        for i_k in range(len(sorted_dict)):
            if sorted_dict[i_k][1]!=0:
                if sorted_dict[i_k][1]<=sorted_dict[knum][1]:
                    K_inset.append(sorted_dict[i_k][0])
                else:
                    break
    else:
        print('第四部分前2个函数出现错误')

    #含有最大距离的字典值
    max_num=sorted_dict[knum][1]
    return K_inset,max_num


#计算所有的LOF编号字典
def Get_LOF(data_set,namelist,knum=K_NUM):
    #数据编号最大值字典
    Dict_All_Max={namelist[ii]:SelectK(EuclideanDistance(data_set[ii],data_set),namelist,knum)[1]\
                  for ii in range(len(namelist))}
    #数据编号集合字典
    Dict_All_Set={namelist[jj]:SelectK(EuclideanDistance(data_set[jj],data_set),namelist,knum)[0]\
                  for jj in range(len(namelist))}
    
    #计算所有的IRDK编号字典
    Dict_All_Irdk={}
    Dict_All_Reach={}
    for ij in range(len(data_set)):
        shangzhi=0
        for i_se in Dict_All_Set[namelist[ij]]:
            shangzhi+=max(Dict_All_Max[i_se],EuclideanDistance(data_set[ij],data_set[list(namelist).index(i_se)]))
        irdk=len(Dict_All_Set[namelist[ij]])/shangzhi
        Dict_All_Irdk[namelist[ij]]=irdk
        Dict_All_Reach[namelist[ij]]=shangzhi

    #计算所有的LOF编号字典
    Dict_All_LOF={}
    for ilof in range(len(data_set)):
        all_irdk=0
        for loflof in Dict_All_Set[namelist[ilof]]:
            all_irdk+=Dict_All_Irdk[loflof]
        product_lof=all_irdk*Dict_All_Reach[namelist[ilof]]
        Dict_All_LOF[namelist[ilof]]=product_lof
            
    return Dict_All_LOF


LOF_MAIN=Get_LOF(data_LOF,code_LOF)#输出的最终的LOF分数字典  

'''++++++++++++++++++++++第五部分：图示++++++++++++++++++++'''
'''
plt.ion()
fig,axes = plt.subplots(1, 3)
fig.subplots_adjust(top=0.92,left=0.07,right=0.97,hspace=0.3,wspace=0.3)
((ax1,ax2,ax3))=axes 

all_data=list(LOF_MAIN.values())

ax1.violinplot(all_data,showmeans=False,showmedians=True)
ax1.set_title('小提琴图')
ax1.set_xlabel('code')
ax1.set_ylabel('LOF number')


ax2.boxplot(all_data)
ax2.set_title('箱线图')
ax2.set_xlabel('code')
ax2.set_ylabel('LOF number')


ax3.hist(all_data)
ax3.set_title('直方图')
ax3.set_xlabel('code')
ax3.set_ylabel('LOF number')

plt.show()
'''
'''++++++++++++++++++++++附加部分：多K值交叉验证++++++++++++++++++++'''
#参数设置
minKnum=2
maxKnum=20

#选取多个K值，根据箱线图的判断异常标准进行交叉判断

outlier_set={}
for K_NUM in range(minKnum,maxKnum+1):
    LOF_MAIN=Get_LOF(data_LOF,code_LOF,K_NUM)
    print(K_NUM)
    npvalue=np.array(list(LOF_MAIN.values()),dtype=float)
    standnum=np.percentile(npvalue,75)+1.5*(np.percentile(npvalue,75)-np.percentile(npvalue,25))
    for ikey,ivalue in LOF_MAIN.items():
        if ivalue>standnum:
            try:
                outlier_set[ikey]+=1
            except KeyError:
                outlier_set[ikey]=1
               
#选取交集
thelastset=[]
for sub_key,sub_value in outlier_set.items():
    if sub_value==maxKnum-minKnum+1:
        thelastset.append(sub_key)
#输出判断的集合
OUTPUT=['Outlier'if code_LOF[itf] in thelastset else 'Normal' for itf in range(len(code_LOF))]
    
'''++++++++++++++++++++++第六部分：输出结果至excel++++++++++++++++++++'''

#将判断异常的数据写入excel文件中
def WriteExcel(WriteDataPath):
    workbook=xw.Workbook(r'%s'%WriteDataPath)
    worksheet=workbook.add_worksheet('Outlier judge')
    format= workbook.add_format({'bold': True, 'font_color': 'red'})

    #抬头数据
    for i_header in range(len(HEADER)):
        worksheet.write(0,i_header,'%s'%(HEADER[i_header]))
    
    #编码数据
    for i_code in range(len(code_LOF)):
        worksheet.write(i_code+1,0,'%s'%code_LOF[i_code])
        
    #原始数据
    for j_initdata in range(len(data_lof)):
        for i_init in range(len(data_lof[j_initdata])):
            worksheet.write(j_initdata+1,i_init+1,'%s'%data_lof[j_initdata][i_init])

    #LOF分数数据
    #for i_score in range(len(list(LOF_MAIN.values()))):
        #worksheet.write(i_score+1,len(HEADER)-2,'%s'%LOF_MAIN[code_LOF[i_score]])

    #最终的结果判断数据
    for i_scoreee in range(len(OUTPUT)):
        worksheet.write(i_scoreee+1,len(HEADER)-1,'%s'%OUTPUT[i_scoreee])
        if OUTPUT[i_scoreee]=='Outlier':
            worksheet.set_row(i_scoreee+1,18,format)

    workbook.close()
    return '异常值判断结果储存在%s'%(WriteDataPath)
    
print(WriteExcel(WriteDataPath))

