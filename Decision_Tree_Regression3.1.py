'''
E-mail: anxinguoguo@126.com
=================================================================
程序说明：
1，基于python3.5.2实现决策树回归算法；【相比1.1,更改节点的编码方式】
2，运行程序的库：Pandas,Numpy,matplotlib,xlsxwriter；
3，剪枝方式【后剪枝、误差增加最小】
4，交叉验证最优子树
=================================================================
'''
print(__doc__)

'''++++++++++++++++++++第一部分：导入需要的库++++++++++++++++'''
import pandas as pd                    #读取数据
import numpy as np                     #程序运算
import matplotlib.pyplot as plt        #画图
import xlsxwriter as xw                #输出至excel文件
import time                            #运行效率问题

plt.ion()
start=time.clock()

'''++++++++++++++++++++++第二部分：数据准备++++++++++++++++++++'''
#数据文件存储路径(需要更改)
DataPath=r'E:\4月14日工作数据\coding_NN_learning\决策树回归算法\CART_data.xlsx'

#写入数据文件的路径
WriteDataPath=r'E:\4月14日工作数据\coding_NN_learning\决策树回归算法\write.xlsx'

sheetname='train'
#读取数据的函数
def ReadData(path,sheetname):
    data_source=pd.read_excel(path,sheetname)
    data_value=data_source.values
    name_data=data_source.keys()[0:-1]
    x_data=np.array(data_value[...,0:-1],dtype=float)
    y_data=np.array(data_value[...,-1],dtype=float)
    return x_data,y_data,name_data

#训练数据
data_cart_train=ReadData(r'%s'%DataPath,'train')
input_train=data_cart_train[0].T  #训练输入
output_train=data_cart_train[1]   #训练输出
name_train=data_cart_train[2]     #训练变量


#测试数据
data_cart_test=ReadData(r'%s'%DataPath,'test')
input_test=data_cart_test[0]      #测试输入
output_test=data_cart_test[1]     #测试输出
name_test=data_cart_test[2]       #测试输出


#预测数据
data_cart_predict=ReadData(r'%s'%DataPath,'predict')
input_predict=data_cart_predict[0]      #预测输入
output_predict=data_cart_predict[1]     #预测输出
name_predict=data_cart_predict[2]       #测试输出


#数据集合
TRAIN_DATA_SET=np.append(input_train,np.array([output_train]),axis=0)#训练数据集合

#停止生长的方差设置参数
Stop_Var=10

'''++++++++++++++++++++++第三部分：函数++++++++++++++++++++'''

'''----------------寻找最优分类变量以及分割数据集-----------'''
#间隔均值的函数(单因子)
def Middle_Ave(data_list):
    copy_data=data_list.copy()
    copy_data=np.array(sorted(copy_data))
    middle_num=list((copy_data[0:-1]+copy_data[1:])/2)
    return middle_num

#定义数据集合的函数(因子集合)
def Split_Set(data_set):
    ave_split=[]
    for i_set in range(len(data_set)-1):
        copy_list=set(data_set[i_set])
        ave_split.append(Middle_Ave(copy_list))
    return np.array(ave_split)


#获得最佳分割变量
def Get_Split(data_value,name_list):

    init_var=np.var(data_value[-1])
    
    split_set=Split_Set(data_value)
    for i_split in range(len(split_set)):

        for i_var in split_set[i_split]:
            dayu=data_value[-1][data_value[i_split]>i_var]
            xiaoyu=data_value[-1][data_value[i_split]<=i_var]
            var1=np.var(dayu)
            var2=np.var(xiaoyu) 
            var=var1+var2#分割后的方差
                        
            try:
                if var<split_var:
                    split_var=var     #最小方差
                    split_num=i_var   #分类属性值
                    split_name=i_split#最佳变量的编号
            except NameError:
                    split_var=var
                    split_num=i_var
                    split_name=i_split
                    
    if split_var>init_var: #分解后的方差必须小于分解前的。否则此数据集不再分解
        return False
    else:
    #遍历
        return [split_name,split_num,name_list[split_name]]


#定义分裂数据集的函数(因子+输出的集合,数据集的值)
def Divide_DataSet(data_set,name_list):
    best_thing=Get_Split(data_set,name_list)
    if best_thing:
        #分类参考
        par_split=best_thing[0]#变量编号
        num_split=best_thing[1]#分类属性值
        #数据集
        large_set=[]
        small_set=[]
        for i_set in range(len(data_set)):
            large_set.append(data_set[i_set][data_set[par_split]>num_split])
            small_set.append(data_set[i_set][data_set[par_split]<=num_split])
        return np.array(small_set),np.array(large_set),[name_list[par_split],num_split,par_split]
    else:
        return False

'''++++++++++++++++++++++第四部分：生长++++++++++++++++++++'''
#数据集原始字典
DATA_DICT={}
DATA_DICT[1]=TRAIN_DATA_SET

   
#定义生长树函数
def Growth_Tree(data_dict,name_list,stop_para=Stop_Var):
    son_tree=0                 #用于记录子树的数量
    divided_para={}            #用于存储分裂变量的字典
    divided_para[1]={}
    
    save_son_tree={}           #存储每一个节点的数据集字典
    start_node=1               #开始的节点编码
    
    node_ship={}               #存储节点关系的字典
       
    while len(data_dict)!=0:           #数据集字典始终改变
        copy_data_dict={}              #用于存储数据集的字典
                             
        for i_data in data_dict:#循环数据集
            
            zuiyou=Divide_DataSet(data_dict[i_data],name_list)#分裂数据集,【最优属性名称,最佳分点,最佳属性编号】

            if zuiyou:

                #数据集的左半部分
                start_node+=1
                save_son_tree[start_node]=zuiyou[0]
            
                #针对左半部分(数据集)
                if np.var(zuiyou[0][-1])<=stop_para: #判断新生成的数据集是否可再分类
                    son_tree+=1                      #子树数加一
                
                else:
                    copy_data_dict[start_node]=zuiyou[0]      #小于等于数据集
                
                #针对最佳变量
            
                divided_para[start_node]=divided_para[i_data].copy()#需要更改字典的键
            
                divided_para[start_node]['%d0'%zuiyou[2][2]]=[zuiyou[2][1],'0',zuiyou[2][2]]
            
                #数据集的右半部分
                start_node+=1
                save_son_tree[start_node]=zuiyou[1]
    
                #针对右半部分(数据集)
                if np.var(zuiyou[1][-1])<=stop_para:        #判断分裂后的数据集是否可再分类
                    son_tree+=1                             #子树数加一
                
                else:
                    copy_data_dict[start_node]=zuiyou[1]    #大于数据集

                #针对最佳变量
                divided_para[start_node]=divided_para[i_data].copy()
                divided_para[start_node]['%d1'%zuiyou[2][2]]=[zuiyou[2][1],'1',zuiyou[2][2]]

                node_ship[i_data]=[start_node-1,start_node]
            
        data_dict=copy_data_dict.copy()  #用于存储数据集的字典
 
    return node_ship,divided_para,save_son_tree #所有节点的关系，分割关系的字典，每一个节点的数据集

#输出生长树的所有分类节点关系、子树的节点编号、所有节点的数据集
growth_tree=Growth_Tree(DATA_DICT,name_train)


'''++++++++++++++++++++++第五部分：剪枝++++++++++++++++++++'''
def Recreate_Ship(dictd,num):

    de=dictd[num]

    for i in de:
        if i in dictd:
            de+=dictd[i]

    for irr in de:
        if irr in dictd:
            del dictd[irr]
            
    for ji in  dictd[num]:
        if ji in dictd:
            del dictd[ji]
            
    del dictd[num]
    
    return dictd


#转换符号的函数
def Convert_Symbol(a,b,ss):
    if ss=='0':
        return eval('a%sb'%'<=')
    elif ss=='1':
        return eval('a%sb'%'>')

#判断符合规则的函数
def Judge_Rule(data_sigle,laol,dicidedpa):
    for i_node in laol:
        sign_num=0
        for i_factor in dicidedpa[i_node]:
            if  Convert_Symbol(data_sigle[dicidedpa[i_node][i_factor][2]],dicidedpa[i_node][i_factor][0],dicidedpa[i_node][i_factor][1]):
                sign_num=1
            else:
                sign_num=0
                break
        if sign_num==1:
            return i_node#返回的是叶节点数据集字典的键


#根据节点关系判断叶子结点的函数
def Judge_Loafnodes(nodesship):#最终判断的数据集一定在叶节点中
    gennodes=list(nodesship.keys())
    allnodes=[]
    for inodes in nodesship.values():
        allnodes+=inodes
    return list(set(allnodes).difference(set(gennodes)))#返回叶子结点


#测试函数
def Test_Tree(tdata,nodesship,data_dict,dicidedpa):
    loafnodesq=Judge_Loafnodes(nodesship)    
    match_target=[]
    for i_data in tdata:
        d=Judge_Rule(i_data,loafnodesq,dicidedpa)
        match_target.append(np.mean(data_dict[d][-1]))#可换为中位值np.median
    return match_target


#计算误差函数
def Get_Error(outdata,tardata):
    return (((outdata-tardata)**2)/(2*len(outdata))).sum()

#剪枝函数[根据误差增长最小剪枝节点]
def Pruning_Tree(trindata,troutdata,nodeship,dividedpara,datanode):

    #记录最有子树序列
    record_best_sontree={}

    #代数
    iter_times=1
    #初始值
    deletenode=0
    
    node_ship=nodeship.copy()
    
    
    #判断剪枝停止的条件：
    while len(node_ship)!=1 or deletenode not in [2,3]:
            
        #记录误差的字典
        record_error={}
        for inkey in node_ship:
            if inkey!=1:
                #叶节点误差
                error_one=np.var(datanode[inkey][-1])
            
                #根节点误差
                error1=0
                for isonnode in node_ship[inkey]:
                    error1+=np.var(datanode[isonnode][-1])
                record_error[inkey]=error_one-error1

        #选择误差值最小的键
        minkey=min(record_error.items(),key=lambda x:x[1])[0]
       

        #在节点的关系中删除此节点
        node_ship=Recreate_Ship(node_ship,minkey)

        #储存此树
        record_best_sontree[iter_times]=node_ship.copy()

        #代数加1
        iter_times+=1

        #停止条件
        deletenode=minkey
        
    return record_best_sontree
       
SE_Tree=Pruning_Tree(input_train.T,output_train,growth_tree[0],growth_tree[1],growth_tree[2])
 

'''++++++++++++++++++++++第六部分：交叉验证++++++++++++++++++++'''           
    
#交叉验证函数
def Cross_Validation(vaindata,vaoudata,para_value,treese,data_dict):

    #输出叶节点的集合
    for itrr in treese:
        outdata_tr=Test_Tree(vaindata,treese[itrr],data_dict,para_value)
        error_tr=Get_Error(vaoudata,outdata_tr)+len(Judge_Loafnodes(treese[itrr]))*5
        print(itrr,error_tr)
        try:
            if smallest>error_tr:
                Last_Tree=treese[itrr]
                Last_Code=itrr
                smallest=error_tr
                
        except NameError:
            smallest=error_tr
            Last_Tree=treese[itrr]
            Last_Code=itrr
            
    print(Last_Code)
    #测试数据图示
    outdata_te=Test_Tree(vaindata,Last_Tree,data_dict,para_value,)
    #输出图示
    fig=plt.figure(figsize=(15,8))
    
    #测试数据实际值与输出值对比图
    ax22=fig.add_subplot(1,1,1)
    ax22.scatter(range(len(outdata_te)),vaoudata,c='g',alpha=1.0,label='Y_target',marker='o')
    ax22.plot(outdata_te,c='r',alpha=2.0,label='Y_output')
    
    ax22.set_title('The Contrast between Real and Output')
    ax22.set_xlabel('Sequence')
    ax22.set_ylabel('Data')
    ax22.legend(loc='upper left')

    plt.show()

    return Last_Tree

LAST_TREE=Cross_Validation(input_test,output_test,growth_tree[1],SE_Tree,growth_tree[2])


'''++++++++++++++++++++++第七部分：模型预测++++++++++++++++++++''' 

#预测函数
def Predict_Plot(prindata,proudata,para_value,lasttree,data_dict):
    #决策树回归预测值
    outdata_pr=Test_Tree(prindata,lasttree,data_dict,para_value)
    #输出图示
    fig=plt.figure(figsize=(15,8))
    fig.subplots_adjust(wspace=0.6,hspace=0.9)
    
    #测试数据实际值与输出值对比图
    ax13=fig.add_subplot(2,1,1)
    ax13.scatter(range(len(proudata)),proudata,c='g',alpha=1.0,label='Y_target',marker='o')
    ax13.plot(outdata_pr,c='r',alpha=2.0,label='Y_output')
    
    ax13.set_title('Predict and Output data')
    ax13.set_xlabel('Sequence')
    ax13.set_ylabel('Data')
    ax13.legend(loc='upper left')

    #误差
    ax12=fig.add_subplot(2,1,2)
    ax12.scatter(range(len(proudata)),proudata-outdata_pr,facecolor='g',alpha=0.5)
    ax12.plot(range(len(proudata)),[0]*len(proudata),c='r',alpha=1)
    ax12.set_title('error')
        
    plt.show()

    return '++++++++++++++++++++++结束++++++++++++++++++++' 


print(Predict_Plot(input_predict,output_predict,growth_tree[1],LAST_TREE,growth_tree[2]))
    
end=time.clock()

print(-start+end)




                 






