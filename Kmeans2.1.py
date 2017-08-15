'''
E-mail: anxinguoguo@126.com
=================================================================
程序说明：
1，基于python3.5.2实现k_means++聚类优化算法；
2，运行程序的库：Pandas,Numpy,matplotlib,xlsxwriter；
3，++【初始值的优化】
4，优化【Hartigan and Wong：A K-Means Clustering Algorithm】
=================================================================
'''
print(__doc__)

'''++++++++++++++++++++第一部分：导入需要的库++++++++++++++++'''
import pandas as pd                    #读取数据
import numpy as np                     #程序运算
import matplotlib.pyplot as plt        #画图
import xlsxwriter as xw                #输出至excel文件

'''++++++++++++++++++++++第二部分：数据准备++++++++++++++++++++'''
#数据文件存储路径(需要更改)
DataPath=r'E:\4月14日工作数据\coding_NN_learning\K-means聚类算法\kmeans.xlsx'

#写入数据文件的路径
WriteDataPath=r'E:\4月14日工作数据\coding_NN_learning\K-means聚类算法\write.xlsx'


#数据集名称
DATA_list=['Iris','Wine','Yeast']

#读取数据的函数
def ReadData(path,sheetname):
    data_source=pd.read_excel(path,sheetname)
    data_value=data_source.values
    name_data=data_source.keys()[0:-1]
    x_data=np.array(data_value[...,0:-1],dtype=float)
    y_data=np.array(data_value[...,-1],dtype=str)
    return x_data,y_data,name_data

#数据
data_cart_train=ReadData(r'%s'%DataPath,DATA_list[1])
input_factor1=data_cart_train[0]                               #因子输入
output_type=data_cart_train[1]                                 #类别
name_factor=data_cart_train[2]

#定义求取标准差的函数(R-1和Python标准差区别)
def Std_np(listn):
     return (((listn-np.mean(listn))**2).sum()/(len(listn)-1))**0.5
    
#正态化数据函数
def Stand_Format(nplist):
    np_list_copy=nplist.copy().T
    for i_convert in range(len(np_list_copy)):
        np_list_copy[i_convert]=(np_list_copy[i_convert]-np.mean(np_list_copy[i_convert]))\
                                 /Std_np(np_list_copy[i_convert])
    return np_list_copy.T

input_factor=Stand_Format(input_factor1)
         
'''++++++++++++++++++++第二部分：参数设置++++++++++++++++'''
Count_Type=3
Max_type=20

'''++++++++++++++++++++第三部分：函数部分++++++++++++++++'''

#定义选择初始中心点列表的函数(k-means++)
def Select_Node(fdata,ctype=Count_Type):
    init_node=np.random.choice(len(fdata))#随机选取的第一个中心
    init_sample=fdata[init_node]
    
    center_list=[init_node]
    #按着k-means++算法寻找其他的中心
    while len(center_list)!=ctype:
        distance_np=[]
        if len(center_list)==1:
            max_dis=((fdata-init_sample)**2).sum(axis=1)
        else:
            for i_node in center_list:
                distance=((fdata-fdata[i_node])**2).sum(axis=1)
                distance_np.append(distance)
                
            max_dis=np.array(distance_np).min(axis=0)
        
        p_dis=max_dis/max_dis.sum()
   
        #选择最佳的中心
        node_dict={}
        for i_data in range(len(fdata)):
            if i_data==0:
                node_dict[i_data]=[0,p_dis[i_data]]
            else:
                node_dict[i_data]=[node_dict[i_data-1][1],node_dict[i_data-1][1]+p_dis[i_data]]
        #随机选取一个(0,1)之间的小数
        random_float=np.random.random()
        #选择概率最大的中心点
        for i_key in node_dict:
            if random_float>=node_dict[i_key][0] and random_float<node_dict[i_key][1]:
                if i_key not in center_list:
                    center_list.append(i_key)

    return center_list


#定义返回最小值下标的函数
def Return_code(nolist):
    listlist=np.array([list(range(len(nolist[0])))]*len(nolist))
    low_code=listlist[nolist==nolist.min(axis=1).reshape(len(nolist),1)]
    return low_code

#定义返回次最小值下标的函数
def Return_coder(nolist):
    lowlow_code=Return_code(nolist)
    copy_nolist=nolist.copy()
    for i_low in range(len(lowlow_code)):
        copy_nolist[i_low][lowlow_code[i_low]]+=max(nolist[i_low])
    return Return_code(copy_nolist)

#定义误差函数
def Get_error(np_list):
    mea_np=((np.array(np_list)-np.array(np_list).mean(axis=0))**2).sum()
    return mea_np

#定义字典键值互换以及合并的函数
def Key_value(ex_dict):
    swop_dict={}
    for key_i,value_j in ex_dict.items():
        if value_j[0] in swop_dict:
            swop_dict[value_j[0]].append(key_i)
        else:
            swop_dict[value_j[0]]=[key_i]
    return swop_dict
        
#定义k-means-HW优化函数
def K_MEANS_HW(fdata,ctype):
    Center_List=Select_Node(fdata,ctype)#初始节点的编号列表
    
    Center_Data=[]#初始选择的中心数据
    for i_code in Center_List:
        Center_Data.append(list(fdata[i_code]))

    #初始聚类阶段
    dis_set=[]
    for j_data in Center_Data:
        dis_data_node=(((fdata-j_data)**2).sum(axis=1))
        dis_set.append(dis_data_node)
    dis_set_select=np.array(dis_set).T
    
    #返回最小值以及次最小值的索引
    low_class=Return_code(dis_set_select)
    lower_class=Return_coder(dis_set_select)
    
    #数据点、类别字典
    data_type_dict={data_i:[low_class[data_i],lower_class[data_i]]for data_i in range(len(fdata))}

    #类别、数据点的字典
    type_data_dict=Key_value(data_type_dict)

    #类别、中心点字典
    type_center_dict={type_ii:(fdata[type_data_dict[type_ii],:]).mean(axis=0)\
                      for type_ii in type_data_dict}
    
    #自由类别的集合
    Live_Set=list(range(len(Center_List)))

    #所有类别的集合
    ALL_Set=list(range(len(Center_List)))
    
    #开始聚类算法
    while len(Live_Set)!=0:#自由类别集合不为空
        Change_Set=[] #记录产生变化的类别
        
        #优化转移阶段(OPTRA)
        #遍历每一条数据
        for op_data in range(len(fdata)):
            #如果所属类别在自由类别集合中
            if data_type_dict[op_data][0] in Live_Set:
                
                #计算此数据与所有其他类别的R2
                distance_R2={dis_i:len(type_data_dict[dis_i])*(((fdata[op_data]-type_center_dict[dis_i])**2).sum())\
                               /(len(type_data_dict[dis_i])+1)for dis_i in ALL_Set if dis_i!=data_type_dict[op_data][0]}

             
                smallest_type=min(distance_R2.items(),key=lambda x:x[1])[0]
                
                #计算此数据与本身类别的R2
                R2_self=len(type_data_dict[data_type_dict[op_data][0]])\
                         *((((fdata[op_data]-type_center_dict[data_type_dict[op_data][0]])**2).sum()))\
                         /(len(type_data_dict[data_type_dict[op_data][0]])-1)

               
                #判断两个R2的大小
                if distance_R2[smallest_type]>=R2_self:#改变原来的次最优类别
                    data_type_dict[op_data][1]=smallest_type
                  
                else:
                    #记录更改
                    Change_Set.append(data_type_dict[op_data][0])
                    Change_Set.append(smallest_type)
                    
                    #更新数据点、类别字典
                    nump=data_type_dict[op_data][0]
                    data_type_dict[op_data][0]=smallest_type
                    data_type_dict[op_data][1]=nump

                    #更新类别、数据点的字典
                    type_data_dict=Key_value(data_type_dict)

                    #更新类别、中心点字典
                    type_center_dict={type_ii:(fdata[type_data_dict[type_ii],:]).mean(axis=0) for type_ii in type_data_dict}

                                       
            else:#如果不在自由类别集合中
                
                #计算此数据与所有其他类别的R2
                distance_R2={dis_ii:len(type_data_dict[dis_ii])*(((fdata[op_data]-type_center_dict[dis_ii])**2).sum())\
                               /(len(type_data_dict[dis_ii])+1)for dis_ii in Live_Set if dis_ii!=data_type_dict[op_data][0]}
                smallest_type=min(distance_R2.items(),key=lambda x:x[1])[0]
                
                #计算此数据与本身类别的R2
                R2_self=len(type_data_dict[data_type_dict[op_data][0]])\
                         *((((fdata[op_data]-type_center_dict[data_type_dict[op_data][0]])**2).sum()))\
                         /(len(type_data_dict[data_type_dict[op_data][0]])-1)
                
                #判断两个R2的大小
                if distance_R2[smallest_type]>=R2_self:#改变原来的次最优类别
                    data_type_dict[op_data][1]=smallest_type
                    
                else:
                    #记录更改
                    Change_Set.append(data_type_dict[op_data][0])
                    Change_Set.append(smallest_type)
                    
                    #更新数据点、类别字典
                    nump=data_type_dict[op_data][0]
                    data_type_dict[op_data][0]=smallest_type
                    data_type_dict[op_data][1]=nump

                    #更新类别、数据点的字典
                    type_data_dict=Key_value(data_type_dict)

                    #更新类别、中心点字典
                    type_center_dict={type_ii:(fdata[type_data_dict[type_ii],:]).mean(axis=0) for type_ii in type_data_dict}
       
        Live_Set=list(set(Change_Set.copy()))
        print('mmmm')
        print(Live_Set)
        if len(Live_Set)==0:
            break
       
        #快速转移阶段(QTRAN)
        Transfer_Set=ALL_Set.copy()
        while Transfer_Set!=[]:
            Alla_Set=[]
            #遍历数据集
            Set_Change=[]#记录更改
            for qt_data in range(len(fdata)):
                if data_type_dict[op_data][0] in Transfer_Set \
                   or data_type_dict[op_data][1]in Transfer_Set:
                    #计算R1和R2
                    R1_rt=len(type_data_dict[data_type_dict[op_data][0]])*\
                           (fdata[qt_data]-type_center_dict[data_type_dict[op_data][0]]).sum()\
                           /(len(type_data_dict[data_type_dict[op_data][0]])-1)
                   
                    
                    R2_rt=len(type_data_dict[data_type_dict[op_data][1]])*\
                           (fdata[qt_data]-type_center_dict[data_type_dict[op_data][1]]).sum()\
                           /(len(type_data_dict[data_type_dict[op_data][1]])+1)
                  
                    #判断
                    if R1_rt>R2_rt and data_type_dict[op_data][0]!=data_type_dict[op_data][1]:

                        #记录更改
                        Set_Change.append(data_type_dict[op_data][0])
                        print(data_type_dict[op_data][0])
                        Set_Change.append(data_type_dict[op_data][1])
                        print(data_type_dict[op_data][1])

                        Alla_Set.append(data_type_dict[op_data][0])
                        Alla_Set.append(data_type_dict[op_data][1])
                        
                        #更新数据点、类别字典
                        nump=data_type_dict[op_data][0]
                        data_type_dict[op_data][0]=smallest_type
                        data_type_dict[op_data][1]=nump

                        #更新类别、数据点的字典
                        type_data_dict=Key_value(data_type_dict)

                        #更新类别、中心点字典
                        type_center_dict={type_iii:(fdata[type_data_dict[type_iii],:]).mean(axis=0) \
                                          for type_iii in type_data_dict}
                        
            #更改自由类别集合
            
            Transfer_Set=list(set(Set_Change.copy()))
            print('gggggggggggggg')
            print(Transfer_Set)
            
        Live_Se1t=list(set(Alla_Set.copy()))  
        Live_Set+=Live_Se1t
        print('nnnnnnn')
        print(Live_Set)
    
    return type_data_dict,type_center_dict


'''++++++++++++++++++++第四部分：聚类结果展示++++++++++++++++'''
kkk=K_MEANS_HW(input_factor,ctype=Count_Type)
    
for i_in in range(1,Count_Type+1):
    print('第%s类的中心为%s'%(i_in,kkk[1][i_in-1]))
    print('此类别包括%s'%output_type[kkk[0][i_in-1]])

    

        
        
                
    
        



















