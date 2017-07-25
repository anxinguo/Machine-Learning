'''GSOM算法
'''
#引入库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#引入数据
Data_Path=r'E:\4月14日工作数据\coding_NN_learning\GSOM算法'                     #数据文件存储路径
Data_Test=['shui11','dongwu16','shui7','dongwu101','shui3','shuizhan']         #数据集

#读取数据
Read_Data=pd.read_excel(r'%s\测试数据.xlsx'%Data_Path,sheetname=Data_Test[5])   #转换数据集
Gsom_Data=Read_Data.values
Gsom_Para=(Read_Data.columns).values[1:]                                       #属性数据
Gsom_input=np.array(Gsom_Data[...,1:],dtype=np.float)                          #输入数据
Gsom_tag=np.array(Gsom_Data[...,0],dtype=np.str)                               #数据编号

#GSOM变量(生长阶段)
learn_rate=0.8                        #学习率
spread_factor=0.4                     #网络扩展因子（负->节点扩展阈值）
neighbourhood_radius=3                #邻域半径
node_count=3.8                        #节点数因子小于4（负->学习率）
neighbourhood_decay=0.1               #邻域衰减因子（负->学习率）
initial_radius=3                      #初始邻域半径因子(正->学习率)
error_distribute=0.39                 #误差分布因子
node_grow_count=0                     #生长阶段终止条件
delete_node_redundancy=5              #删除冗余的代数

#GSOM变量(平滑阶段)
learn_rate_smooth=0.02                #学习率
neighbourhood_radius_smooth=0         #邻域半径
node_factor_smooth=node_count         #节点数因子
neighbourhood_decay_smooth=0.8        #邻域衰减因子
initial_radius_smooth=1               #初始邻域半径因子
node_error_change=1/10**4             #平滑阶段终止条件

#GSOM变量(优化阶段)
iteration_times=50                    #迭代次数

'''
初始化阶段
'''
#GSOM变量
node_position=[(0,0),(1,0),(1,1),(0,1)]                                                      #四个nodes的二维坐标 
node_vector=np.random.random((len(node_position),len(Gsom_input[0])))                        #随机0-1间向量
vector_dictonary={tuple(node_position[i]):node_vector[i]for i in range(len(node_position))}  #坐标与向量字典
error_dictonary={tuple(node_position[i]):0 for i in range(len(node_position))}               #坐标与误差字典
father_dictonary={tuple(node_position[i]):0 for i in range(len(node_position))}              #父节点字典

#网络扩展阈值
growth_thershold=-len(Gsom_input[0])*np.log(spread_factor)

#原始坐标
copy_node_position=node_position.copy()

#数据归一到[0,1]之间
def Norm_data(data_input):
    max_num=data_input.max(axis=0)
    min_num=data_input.min(axis=0)
    data_input_norm=(data_input-min_num)/(max_num-min_num)
    return data_input_norm

#输入数据单位化
gsom_data_norm=Norm_data(Gsom_input)

'''
生长阶段
'''
#计算欧式距离
def Distance_vector(va,vb):
    if va.ndim==2 and vb.ndim==1:
        v_sub=va-vb
        v_pro=np.sqrt((v_sub**2).sum(axis=1))
        return v_pro
    elif va.ndim==1 and vb.ndim==1:
        v_sub=va-vb
        v_pro=np.sqrt((v_sub**2).sum(axis=0))
        return v_pro   

#菱形邻域中符合条件节点二维坐标
def Position_neighbourhood(win_position_p,iter_times_p,node_position_p,neighbourhood_radius_p=neighbourhood_radius):
    satisfy_position_list=[]
    abs_number=max(abs(win_position_p[0]),abs(win_position_p[1]))+neighbourhood_radius_p
    new_win_position=(win_position_p[0]+abs_number,win_position_p[1]+abs_number)
    zero_number,one_number=new_win_position[0],new_win_position[1]
    
    first_zero=zero_number-neighbourhood_radius_p+iter_times_p
    second_zero=zero_number+neighbourhood_radius_p-iter_times_p+1
    
    first_one=one_number-neighbourhood_radius_p+iter_times_p
    second_one=one_number+neighbourhood_radius_p-iter_times_p+1
    all_position=[(a-abs_number,b-abs_number) for a in range(first_zero,second_zero) for b in range(first_one,second_one)]
    square_set=list(set(all_position).intersection(set(node_position_p)))
    rhombus_set=[]
    for i_grid in square_set:
        n0,n1=i_grid[0],i_grid[1]
        if abs(n0-win_position_p[0])+abs(n1-win_position_p[1])<=neighbourhood_radius_p-iter_times_p:
            rhombus_set.append(i_grid)
    return rhombus_set

#矩形邻域中符合条件节点二维坐标
def Position_neighbourhood_rectangle(win_position_p,iter_times_p,node_position_p,neighbourhood_radius_p=neighbourhood_radius):
    satisfy_position_list=[]
    abs_number=max(abs(win_position_p[0]),abs(win_position_p[1]))+neighbourhood_radius_p
    new_win_position=(win_position_p[0]+abs_number,win_position_p[1]+abs_number)
    zero_number,one_number=new_win_position[0],new_win_position[1]
    
    first_zero=zero_number-neighbourhood_radius_p+iter_times_p
    second_zero=zero_number+neighbourhood_radius_p-iter_times_p+1
    
    first_one=one_number-neighbourhood_radius_p+iter_times_p
    second_one=one_number+neighbourhood_radius_p-iter_times_p+1
    all_position=[(a-abs_number,b-abs_number) for a in range(first_zero,second_zero) for b in range(first_one,second_one)]
    square_set=list(set(all_position).intersection(set(node_position_p)))
    return square_set


#学习率函数(原则：距离越大，学习率越小；次数越多，学习率越小；节点数越多，学习率越大)
def Learn_rate_factor(iter_times_p,node_count_p,node_position_p,win_position_p,\
                         learn_rata_p=learn_rate,start_node_p=node_count,de_num=neighbourhood_decay,init_r_p=initial_radius):
    position_winner_np=np.array([win_position_p[0],win_position_p[1]])
    node_position_np=np.array(node_position_p)
    position_distance=((node_position_np-position_winner_np)**2).sum(axis=1)+iter_times_p
    return learn_rata_p*(1-start_node_p/node_count_p)*np.exp(-(position_distance/(init_r_p*np.exp(-de_num*iter_times_p))))

#返回字典中值的最大值
def Max_number(node_order_dict):
    values_dict=node_order_dict.values()
    return max(values_dict)+1

#返回直接节点的二维坐标
def Return_direct(position_sigle):
    n0,n1=position_sigle[0],position_sigle[1]
    return [(n0-1,n1),(n0+1,n1),(n0,n1-1),(n0,n1+1)]

#非边缘节点,返回1；否则，返回缺失坐标集合
def Judge_boundary(position_win,position_all):
    n0,n1=position_win[0],position_win[1]
    if (n0-1,n1) in position_all and (n0+1,n1) in position_all\
       and (n0,n1-1) in position_all and (n0,n1+1) in position_all:
        return 1
    else:
        node_miss=[]  
        node_direct=Return_direct(position_win)
        for i_miss in node_direct:
            if i_miss not in position_all:
                node_miss.append(i_miss)
    return node_miss

#返回最小距离的二维坐标
def Min_position1(position_all,vector_all,vector_sigle):
    position_all=np.array(position_all)
    vector_distance=Distance_vector(vector_all,vector_sigle)
    return tuple((position_all[vector_distance==min(vector_distance)])[0])

#字典组合
def Combine_np(position_np,dictonary_all):
    vector_np=[]
    for i_key in position_np:
        vector_np.append(dictonary_all[i_key])
    return np.array(vector_np)

#返回最小距离的二维坐标
def Min_position(position_all,vector_position_dict,vector_sigle):
    vector_all=Combine_np(position_all,vector_position_dict)
    position_all=np.array(position_all)
    vector_distance=Distance_vector(vector_all,vector_sigle)
    return tuple((position_all[vector_distance==min(vector_distance)])[0])
 
#根据二维坐标集合选择相应的向量集合
def Update_vector_set(dict_position_vector,position_neigh_bour):
    update_vector=[]
    for key in position_neigh_bour:
        if key in dict_position_vector:
            update_vector.append(dict_position_vector[key])
    return np.array(update_vector)

#数值转化
def Convert_number(vector_np):
    for i_convert in range(len(vector_np)):
        if vector_np[i_convert]<0 or vector_np[i_convert]>1:
            vector_np[i_convert]=0.5
    return vector_np        
        
#分布误差的递归函数
def Recursion_distribute_error(set_position,error_dict,position_vector_dict,father_dict,\
                               grow_count_p=0,error_dis_p=error_distribute,expand_factor_p=growth_thershold):
    for i_recursion in set_position:
        #误差累计值大于阈值
        if error_dict[i_recursion]>expand_factor_p:
            #非边缘节点
            if Judge_boundary(i_recursion,set_position)!=1:
                #生长新的节点
                n0,n1=i_recursion[0],i_recursion[1]                         #winner节点坐标
                copy_set_position=set_position.copy()
                for i_insert in Judge_boundary(i_recursion,copy_set_position):
                    grow_count_p+=1
                    set_position.append(i_insert)
                    fir_posi_mis,sec_posi_mis=i_insert[0],i_insert[1]       #新增节点坐标
                    fir,sec=2*fir_posi_mis-n0,2*sec_posi_mis-n1             #区分Case的验证节点坐标
                    #Case b
                    if (fir,sec) in copy_set_position:
                        change_vector=Convert_number((position_vector_dict[i_recursion]+position_vector_dict[(fir,sec)])/2)
                        position_vector_dict[i_insert]=change_vector
                        father_dictonary[i_insert]=i_recursion
                        error_dict[i_insert]=0
                    #Case a和c
                    else:
                        for j_node in Return_direct(i_recursion):
                            if j_node not in Judge_boundary(i_recursion,copy_set_position):
                                stand_node_posi=j_node
                                break
                        try:
                            change_vector=Convert_number(2*position_vector_dict[i_recursion]-position_vector_dict[stand_node_posi])
                        except UnboundLocalError:
                            change_vector=[0.5]*len(position_vector_dict[i_recursion])
                        position_vector_dict[i_insert]=change_vector
                        error_dict[i_insert]=0
                        father_dictonary[i_insert]=i_recursion
                return Recursion_distribute_error(set_position,error_dict,position_vector_dict,father_dict,grow_count_p) 
    #返回坐标集合、向量集合以及累计误差字典
    return set_position,error_dict,position_vector_dict,grow_count_p,father_dict

#生长函数
def Gsom_growth(data_set,position_start,dictonary_start,error_start,father_start,\
                bias_start=growth_thershold,grow_node_start=node_grow_count,dele_node_num=delete_node_redundancy):
    win_count_node={}                                                              #保留二维坐标成为winner的次数    
    stand_num=grow_node_start+1
    de_times=0
    while stand_num>grow_node_start:                                               #根据新生节点个数终止循环
        np.random.shuffle(data_set)                                                #样本随机
        for i_data in data_set:                                                    #单样本循环
            stand_num=0
            win_position_node=Min_position(position_start,dictonary_start,i_data)  #获取距离单样本的最小距离的二维坐标
            win_count_node[win_position_node]=1
            t_iter=0                                                               #单样本迭代次数
            n_count=len(position_start)                                            #网络中含有的节点个数
            #计算winner与该输入的误差
            winner_error_change=Distance_vector(i_data,dictonary_start[win_position_node])
            #非边缘节点
            if Judge_boundary(win_position_node,position_start)==1:
                #分布误差
                for i_judge in Return_direct(win_position_node):
                    error_start[i_judge]+=winner_error_change
                    function_num=Recursion_distribute_error(position_start,error_start,dictonary_start,father_start)
                    position_start=function_num[0]
                    error_start=function_num[1]
                    dictonary_start=function_num[2]
                    stand_num+=function_num[3]
                    father_start=function_num[4]
            #边缘节点
            else:
                #存储这个winner节点与输入信息的误差：
                error_start[win_position_node]+=winner_error_change
                #大于阈值，生长节点
                if error_start[win_position_node]>bias_start:
                    fir_posi,sec_posi=win_position_node[0],win_position_node[1]#winner边缘节点的二维坐标
                    copy_position_start=position_start.copy()#此时的节点的二维坐标集合
                    for i_insert in Judge_boundary(win_position_node,copy_position_start):
                        stand_num+=1
                        position_start.append(i_insert)
                        fir_posi_mis,sec_posi_mis=i_insert[0],i_insert[1]      #新增节点的二维坐标
                        fir,sec=2*fir_posi_mis-fir_posi,2*sec_posi_mis-sec_posi#区分Case的验证节点坐标
                        #Case b
                        if (fir,sec) in copy_position_start:
                            change_vector=Convert_number((dictonary_start[win_position_node]+dictonary_start[(fir,sec)])/2)
                            dictonary_start[i_insert]=change_vector
                            father_dictonary[i_insert]=win_position_node
                            error_start[i_insert]=0 
                        #Case a和c
                        else:
                            for j_node in Return_direct(win_position_node):
                                if j_node not in Judge_boundary(win_position_node,copy_position_start):
                                    stand_node_posi=j_node
                                    break
                            try:
                                change_vector=Convert_number(2*dictonary_start[win_position_node]-dictonary_start[stand_node_posi])
                            except UnboundLocalError:
                                change_vector=[0.5]*len(dictonary_start[win_position_node])
                            dictonary_start[i_insert]=change_vector
                            father_dictonary[i_insert]=win_position_node
                            error_start[i_insert]=0
                else:
                    pass
            #更改win权重值以及邻域的权重值，直到邻域中只包含winner节点
            while len(Position_neighbourhood(win_position_node,t_iter,position_start))!=0:
                near_node_stright=Position_neighbourhood(win_position_node,t_iter,position_start)           #winner二维坐标的邻域
                change_vector_node=Update_vector_set(dictonary_start,near_node_stright)                     #在邻域内的向量集合
                learn_rate=Learn_rate_factor(t_iter,n_count,near_node_stright,win_position_node)            #学习率LR
                change_vector_node+=learn_rate.reshape(len(near_node_stright),1)*(i_data-change_vector_node)#更改节点的向量
                t_iter+=1
                #改变字典中对应的值
                for i_dict in range(len(near_node_stright)):
                    dictonary_start[near_node_stright[i_dict]]=change_vector_node[i_dict]
        de_times+=1
        if de_times%(dele_node_num+1)==0:
        #每隔几代删除从未成为winner的节点
            for i_delete in position_start:
                if i_delete not in win_count_node:
                    try:
                        position_start.remove(i_delete)
                        del dictonary_start[i_delete]
                        del father_start[i_delete]
                    except ValueError:
                        pass
    return position_start,dictonary_start,father_start

'''
平滑阶段
'''
def Gsom_smooth(data,posi_end,dict_end,tag_list,\
                dec_lr_smo=learn_rate_smooth,r_num_smo=node_factor_smooth,near_hoo_smo=neighbourhood_radius_smooth,\
                de_num_smo=neighbourhood_decay_smooth,init_r_smo=initial_radius_smooth,iter_stop_error=node_error_change):
    win_count_node_smooth={}#二维坐标成为winner的次数
    winner_node_smooth={}   #不同数据击中的node
    vector_set_smooth=[]
    n_count=len(posi_end)
    error_stop_list=[1,0]
    while error_stop_list[-2]-error_stop_list[-1]>iter_stop_error or error_stop_list[-2]<error_stop_list[-1]:
        error=0
        for j_data in data:
            hit_count=0
            win_posi_node=Min_position(posi_end,dict_end,j_data)#获取距离单样本的最小距离的二维坐标
            try:
                winner_node_smooth[win_posi_node]+=[tag_list[hit_count]]
            except KeyError:
                winner_node_smooth[win_posi_node]=[tag_list[hit_count]]
            hit_count+=1
            t_iter=0#单样本迭代次数
            #存储次数
            try:
                win_count_node_smooth[win_posi_node]+=1
            except KeyError:
                win_count_node_smooth[win_posi_node]=1
            #更改win权重值以及邻域的权重值，直到邻域中只包含winner节点
            while len(Position_neighbourhood(win_posi_node,t_iter,posi_end,near_hoo_smo))!=0:
                near_node_stright=Position_neighbourhood(win_posi_node,t_iter,posi_end)                     #winner二维坐标的邻域
                change_vector_node=Update_vector_set(dict_end,near_node_stright)                            #在邻域内的向量集合
                learn_rate=Learn_rate_factor(t_iter,n_count,near_node_stright,win_posi_node,\
                                             dec_lr_smo,r_num_smo,de_num_smo,init_r_smo)                    #学习率LR
                change_vector_node+=learn_rate.reshape(len(near_node_stright),1)*(j_data-change_vector_node)#更改节点的向量
                t_iter+=1
                #改变字典中对应的值
                for j_dict in range(len(near_node_stright)):
                    dict_end[near_node_stright[j_dict]]=change_vector_node[j_dict]
            error+=Distance_vector(dict_end[win_posi_node],j_data)
        error_stop_list.append(error)
        print(error)
    return dict_end,posi_end,win_count_node_smooth,winner_node_smooth

'''
开始自生长自适应映射网络的训练
'''
copy_gsom_data_norm=gsom_data_norm.copy()
while iteration_times!=0:
    growth_net=Gsom_growth(copy_gsom_data_norm,node_position,vector_dictonary,error_dictonary,father_dictonary)
    iteration_times-=1
    smooth_net=Gsom_smooth(gsom_data_norm,growth_net[0],growth_net[1],Gsom_tag)

'''
聚类结果展示阶段
'''
#返回站点编号以及对应的节点的字典函数
def Station_node(sta_code,posi_vector_dict,posi_set,input_data):
    code_node={}
    vector_set_code=Update_vector_set(posi_vector_dict,posi_set)
    for i_code in range(len(sta_code)):
        try:
            code_node[Min_position1(posi_set,vector_set_code,input_data[i_code])].append(sta_code[i_code])
        except KeyError:
            code_node[Min_position1(posi_set,vector_set_code,input_data[i_code])]=[sta_code[i_code]]
    return code_node

station_code_node=Station_node(Gsom_tag,smooth_net[0],smooth_net[1],gsom_data_norm)

#求取节点二维坐标的四个极值
def Extremum(posi_num_set):
    posi_num_set=np.array(posi_num_set)
    x_max=max(posi_num_set[...,0])
    x_min=min(posi_num_set[...,0])
    y_max=max(posi_num_set[...,-1])
    y_min=min(posi_num_set[...,-1])
    return x_max,x_min,y_max,y_min,

#Win节点以及非Win节点展示图
def Win_show(position_all,code_node_dict,copy_start_posi=copy_node_position):
    extremum=Extremum(position_all)
    fig_node,ax=plt.subplots()
    ax.set_xlim(0,extremum[0]-extremum[1]+2)
    ax.set_ylim(0,extremum[2]-extremum[3]+2)
    ax.grid(False)
    ax.set_axis_off()
    win_num,no_win=0,0
    legend_sign=0
    for i_win in position_all:
        a_win=i_win[0]
        b_win=i_win[1]
        if i_win in code_node_dict:
            ax.plot([a_win-extremum[1]+1,a_win-extremum[1]+1],[b_win-extremum[3]+1,b_win-extremum[3]+1],\
                    "ko",lw=5,ms=10)                
        else:
            ax.plot([a_win-extremum[1]+1,a_win-extremum[1]+1],[b_win-extremum[3]+1,b_win-extremum[3]+1],\
                    "wo",lw=5,ms=10)
        if i_win in copy_start_posi:
            ax.plot([a_win-extremum[1]+1.1,a_win-extremum[1]+1.1],[b_win-extremum[3]+1.1,b_win-extremum[3]+1.1],\
                    "ro-",mec="r",lw=5,ms=5)
    ax.scatter(a_win-extremum[1]+1,b_win-extremum[3]+1,c='r',label='$Four$ $Start$ $Nodes$',alpha=0.7)
    ax.scatter(a_win-extremum[1]+1,b_win-extremum[3]+1,c='w',label='$Non$_$Winner$ $Nodes$',alpha=0.7)
    ax.scatter(a_win-extremum[1]+1,b_win-extremum[3]+1,c='k',label='$Winner$ $Nodes$',alpha=0.7)            
    plt.title('$Winner$ $and$ $Non$_$Winner$ $Nodes$')
    ax.legend(loc='upper right',shadow=True,fontsize=10) 
    return fig_node

'''
Winner节点以及Non_Winner节点展示图
'''
Fig_node=Win_show(smooth_net[1],station_code_node)

Fig_node.show()

#击中节点的数据图函数
def Hit_count(code_node_list,posi_list,):
    extremum=Extremum(posi_list)
    fig_hit,ax=plt.subplots()
    ax.set_xlim(0,extremum[0]-extremum[1]+2)
    ax.set_ylim(0,extremum[2]-extremum[3]+2)
    ax.grid(True)
    ax.set_axis_off()
    for i_tag in posi_list:
        a_tag=i_tag[0]
        b_tag=i_tag[1]
        if i_tag in code_node_list:
            ax.text(a_tag-extremum[1]+1,b_tag-extremum[3]+1,'%s'%len(code_node_list[i_tag]),\
                    color='k',size=10+len(code_node_list[i_tag]),va='center',ha='center')
            ax.plot([a_tag-extremum[1]+1,a_tag-extremum[1]+1],[b_tag-extremum[3]+1,b_tag-extremum[3]+1],\
                    "yH-",mec="k",lw=5,ms=10*len(code_node_list[i_tag]))
        else:
            ax.plot([a_tag-extremum[1]+1,a_tag-extremum[1]+1],[b_tag-extremum[3]+1,b_tag-extremum[3]+1],\
                    "wH-",mec="k",lw=5,ms=9)
    plt.title('$Hit$_$Nodes$')
    
    return fig_hit
'''
击中节点的数据图
'''
Fig_hit=Hit_count(station_code_node,smooth_net[1])

Fig_hit.show()
            
#网络生长骨架图
def Grow_structure(posi_list,father_dict,node_code_dict,start_position_list=copy_node_position):
    start_path={tuple(start_position_list[i]):tuple(start_position_list[(i+1)%4]) for i in range(len(start_position_list))}  
    extremum=Extremum(posi_list)
    fig_skeleton,ax=plt.subplots()
    ax.set_xlim(0,extremum[0]-extremum[1]+2)
    ax.set_ylim(0,extremum[2]-extremum[3]+2)
    ax.set_axis_off()
    ax.grid(False)
    for i_tag in posi_list:
        a_tag=i_tag[0]
        b_tag=i_tag[1]
        if i_tag in start_position_list:
            ax.plot([a_tag-extremum[1]+1.1,a_tag-extremum[1]+1.1],[b_tag-extremum[3]+1.1,b_tag-extremum[3]+1.1],\
                    "ro-",mec="k",lw=5,ms=5)
        else:
            i_son=father_dict[i_tag]
            c_son=i_son[0]
            d_son=i_son[1]
            ax.plot([a_tag-extremum[1]+1,c_son-extremum[1]+1],[b_tag-extremum[3]+1,d_son-extremum[3]+1],\
                    "ko-",mec="k",lw=2,ms=2)
        if i_tag in node_code_dict:
            ax.plot([a_tag-extremum[1]+1,a_tag-extremum[1]+1],[b_tag-extremum[3]+1,b_tag-extremum[3]+1],\
                    "ko-",mec="k",lw=5,ms=10)
        else:
            ax.plot([a_tag-extremum[1]+1,a_tag-extremum[1]+1],[b_tag-extremum[3]+1,b_tag-extremum[3]+1],\
                    "wo-",mec="k",lw=5,ms=10)
        if i_tag in start_path:
            i_bro=start_path[i_tag]
            c_bro=i_bro[0]
            d_bro=i_bro[1]
            ax.plot([a_tag-extremum[1]+1,c_bro-extremum[1]+1],[b_tag-extremum[3]+1,d_bro-extremum[3]+1],\
                    'ko--',lw=2,ms=2)
    ax.scatter(a_tag-extremum[1]+1,b_tag-extremum[3]+1,c='r',label='$Four$ $Start$ $Nodes$',alpha=0.7)
    ax.scatter(a_tag-extremum[1]+1,b_tag-extremum[3]+1,c='w',label='$Non$_$Winner$ $Nodes$',alpha=0.7)
    ax.scatter(a_tag-extremum[1]+1,b_tag-extremum[3]+1,c='k',label='$Winner$ $Nodes$',alpha=0.7)
    plt.title('$Growth$_$Skeleton$')
    ax.legend(loc='upper right',shadow=True,fontsize=10) 
    return fig_skeleton

'''
网络生长骨架图
'''
Fig_skeleton=Grow_structure(smooth_net[1],growth_net[2],station_code_node)
Fig_skeleton.show()

'''
删除不必要轨迹的函数
'''
#组合元素的函数(字典键)
def Combine_element(dict_father):
    combine_list=[]
    for i_ele in dict_father:
        try:
            dict_father[i_ele]+=1
        except TypeError:
            combine_list.append(dict_father[i_ele])
            combine_list.append(i_ele)
    times_dict={}
    for i_times in dict_father:
        times_dict[i_times]=combine_list.count(i_times)
    return times_dict

#判断删除字典键的函数
def Decide_delete_key(dict_father,win_node_dict):
    combine_dict=Combine_element(dict_father)
    for i_delete in combine_dict:
        if combine_dict[i_delete]==1:
            if i_delete not in win_node_dict:
                del dict_father[i_delete]
                return Decide_delete_key(dict_father,win_node_dict)
    return dict_father 

#组合元素的函数(字典值)
def Combine_element_item(dict_father):
    combine_list=[]
    for i_ele in dict_father:
        try:
            dict_father[i_ele]+=1
        except TypeError:
            combine_list.append(dict_father[i_ele])
            combine_list.append(i_ele)
    times_dict={}
    copy_combine_list=combine_list.copy()
    for i_times in set(copy_combine_list):
        times_dict[i_times]=combine_list.count(i_times)
    return times_dict

#判断删除字典值的函数
def Decide_delete_item(dict_father_item,win_node_dict_item):
    combine_dict=Combine_element_item(dict_father_item)
    copy_dict_father_item=dict_father_item.copy()
    for i_item in combine_dict:
        if combine_dict[i_item]==1 and i_item not in win_node_dict_item:
            for j_item in dict_father_item:
                if dict_father_item[j_item]==i_item:
                    del copy_dict_father_item[j_item]
                    dict_father_item=copy_dict_father_item
                    return Decide_delete_item(dict_father_item,win_node_dict_item)
    return dict_father_item 
    
copy_father=growth_net[2].copy()
delete_key_dict=Decide_delete_key(copy_father,station_code_node)
copy_delete_key_dict=delete_key_dict.copy()
delete_item_dict=Decide_delete_item(copy_delete_key_dict,station_code_node)

'''
根据邻域确定的分组函数（适用于完全扩展的结构）
'''
def Base_group1(dict_result):
    dict_base={}
    for i_base in dict_result:
        dict_base[i_base]=[i_base]
    return dict_base

based_group=Base_group1(station_code_node)

#矩形邻域中符合条件节点二维坐标
def Base_group2(win_posi,iter_t,posi_set,near_para=1):
    satisfy_posi=[]
    abs_num=max(abs(win_posi[0]),abs(win_posi[1]))+near_para
    new_win_posi=(win_posi[0]+abs_num,win_posi[1]+abs_num)
    fir_num0=new_win_posi[0]-(near_para-iter_t)
    sec_num0=new_win_posi[0]+(near_para-iter_t)+1
    fir_num1=new_win_posi[1]-(near_para-iter_t)
    sec_num1=new_win_posi[1]+(near_para-iter_t)+1   
    all_posi=[(a-abs_num,b-abs_num) for a in range(fir_num0,sec_num0) for b in range(fir_num1,sec_num1)]#决定分组的关键语句
    #all_posi=Return_direct(win_posi)
    return list(set(all_posi).intersection(set(posi_set)))

#邻域分组函数   
def Divide_nearhood_group(win_dict_group,dict_result,merge_num=1):
    last_list=[]
    for i_nearhood in win_dict_group:
        for j_nearhood in Base_group2(i_nearhood,0,list(win_dict_group.keys()),near_para=merge_num):
            if j_nearhood in win_dict_group:
                dict_result[i_nearhood].append(j_nearhood)
        last_list.append(dict_result[i_nearhood])
    return last_list

near_group=Divide_nearhood_group(station_code_node,based_group)

    
#列表函数
def Divide_path_group(posi_posi_dict):
    list_group=[]
    for i_divide in posi_posi_dict:
        try:
            posi_posi_dict[i_divide]+=0
        except TypeError:
            list_group.append([i_divide,posi_posi_dict[i_divide]])
    return list_group

need_combine_list=Divide_path_group(delete_item_dict)

#含有交集的汇总            
def Gather_group(posi_list_set):
    for i_gather in range(len(posi_list_set)-1):
        for j_gather in range(i_gather+1,len(posi_list_set)):
            i_list=posi_list_set[i_gather]
            j_list=posi_list_set[j_gather]
            inter_list=list(set(i_list).intersection(set(j_list)))
            if inter_list!=[]:
                posi_list_set.append(list(set(i_list+j_list)))
                posi_list_set.remove(i_list)
                posi_list_set.remove(j_list)
                return Gather_group(posi_list_set)
    return posi_list_set

#根据初步分组确定最终的分组
def Last_group(combine_list):
    last_group_list=[]
    for j_combine in  Gather_group(combine_list):
        last_group_list.append(j_combine)
    return last_group_list

group_end_near=Last_group(near_group)

#删除不必要的生长轨迹图
def Grow_structure_delete(posi_list,father_dict,node_code_dict,tag_sign='Null',start_position_list=copy_node_position):
    start_path={tuple(start_position_list[i]):tuple(start_position_list[(i+1)%4]) for i in range(len(start_position_list))}  
    extremum=Extremum(posi_list)
    fig_delete,ax=plt.subplots()
    ax.set_xlim(0,extremum[0]-extremum[1]+2)
    ax.set_ylim(0,extremum[2]-extremum[3]+2)
    ax.set_axis_off()
    ax.grid(False)
    for i_tag in posi_list:
        a_tag=i_tag[0]
        b_tag=i_tag[1]
        if i_tag in start_position_list:
            ax.plot([a_tag-extremum[1]+1.1,a_tag-extremum[1]+1.1],[b_tag-extremum[3]+1.1,b_tag-extremum[3]+1.1],\
                    "ro-",mec="k",lw=5,ms=5)
        else:
            try:
                i_son=father_dict[i_tag]
                c_son=i_son[0]
                d_son=i_son[1]
                ax.plot([a_tag-extremum[1]+1,c_son-extremum[1]+1],[b_tag-extremum[3]+1,d_son-extremum[3]+1],\
                        "ko-",mec="k",lw=2,ms=2)
            except KeyError:
                pass
            
        if i_tag in node_code_dict:
            ax.plot([a_tag-extremum[1]+1,a_tag-extremum[1]+1],[b_tag-extremum[3]+1,b_tag-extremum[3]+1],\
                    "ko-",mec="k",lw=5,ms=10)
        else:
            ax.plot([a_tag-extremum[1]+1,a_tag-extremum[1]+1],[b_tag-extremum[3]+1,b_tag-extremum[3]+1],\
                    "wo-",mec="k",lw=5,ms=10)
            
        if tag_sign=='Tag':
            if i_tag in node_code_dict:
                ax.annotate('%s'%node_code_dict[i_tag],xy=(a_tag-extremum[1]+1,b_tag-extremum[3]+1))
            plt.title('$Growth$_$Skeleton$ $with$ $Winner$ $Nodes$ $and$ $Codes$ ')    
        else:
            plt.title('$Growth$_$Skeleton$ $with$ $Winner$ $Nodes$')
        if i_tag in start_path:
            i_bro=start_path[i_tag]
            c_bro=i_bro[0]
            d_bro=i_bro[1]
            ax.plot([a_tag-extremum[1]+1,c_bro-extremum[1]+1],[b_tag-extremum[3]+1,d_bro-extremum[3]+1],\
                    'ko--',lw=2,ms=2)
    ax.scatter(a_tag-extremum[1]+1,b_tag-extremum[3]+1,c='r',label='$Four$ $Start$ $Nodes$',alpha=0.7)
    ax.scatter(a_tag-extremum[1]+1,b_tag-extremum[3]+1,c='w',label='$Non$_$Winner$ $Nodes$',alpha=0.7)
    ax.scatter(a_tag-extremum[1]+1,b_tag-extremum[3]+1,c='k',label='$Winner$ $Nodes$',alpha=0.7)
    ax.legend(loc='upper right',fontsize=10) 
    return fig_delete

Fig_delete2=Grow_structure_delete(smooth_net[1],delete_item_dict,station_code_node)

Fig_delete2.show()

#分组之后的编号信息
def After_group_node(group_list,station_node):
    station_node_dara=[]
    for i_list in range(len(group_list)):
        sigle_group=[]
        for j_list in group_list[i_list]:
            if j_list in station_node:
                for ji_list in station_node[j_list]:
                    if ji_list not in sigle_group:
                        sigle_group.append(ji_list)
        station_node_dara.append(sigle_group)
    return station_node_dara

after_group_node=After_group_node(group_end_near,station_code_node)

#最长组员数
def Long_count(group_list):
    length=0
    for i_count in group_list:
        if length<len(i_count):
            length=len(i_count)
    return length
    

#不同类别分布图具有图例的
def Type_cluster_legend(group_list,posi_list,win_po_node,\
                        node_setm,win_sign='Null',num_stand=12,peace_num=0.5):
    length_list=Long_count(node_setm)
    n_l=int(length_list)/num_stand+len(node_setm)-1
    color_count=len(group_list)
    extremum=Extremum(posi_list)
    color_list=['#666699','#0099CC','#99CC00','#99CC99','#CCCC33','#009933',\
                '#FF6666','#009966','#CC6600','#FFCC33','#6699FF','#339966',\
                '#990066','#FF6600','#33CC99','#990033','#FF0033','#663366',\
                '#CCCCFF','#FF9933','#009999','#CC9966','#CCCC00','#99CCFF',\
                '#CC99CC','#336699','#996699','#003399','#999999','#FFFF00',\
                '#CC3399','#66CCCC']
    fig_legend,ax=plt.subplots()
    ax.set_xlim(-n_l*peace_num,extremum[0]-extremum[1]+3)
    ax.set_ylim(-n_l*peace_num,extremum[2]-extremum[3]+3)
    ax.set_axis_off()
    all_win_node=[]
    for i_plot in range(color_count):
        for i_point in group_list[i_plot]:
            a=i_point[0]
            b=i_point[1]
            all_win_node.append(i_point)
            ax.broken_barh([(a-extremum[1]+1,1)],(b-extremum[3]+1,1),facecolors=color_list[i_plot%len(color_list)])
        if len(node_setm[i_plot])>num_stand:
            ax.scatter(-n_l,0,c='%s'%color_list[i_plot],label='%s'%node_setm[i_plot][:num_stand],alpha=0.7,marker='s')
            i_iter=1
            while i_iter<int(len(node_setm[i_plot])/num_stand)+1:
                if node_setm[i_plot][i_iter*num_stand:(i_iter+1)*num_stand]!=[]:
                    ax.scatter(-n_l,0,c='white',label='%s'%node_setm[i_plot][i_iter*num_stand:(i_iter+1)*num_stand],alpha=0.0)
                    i_iter+=1
        else:
            ax.scatter(-n_l,0,c='%s'%color_list[i_plot%len(color_list)],label='%s'%node_setm[i_plot],alpha=0.7,marker='s')
        plt.legend(loc='lower center',shadow=True,fontsize=10)           
    if win_sign=='Win':
        for i_plot in posi_list:
            if i_plot not in all_win_node:
                a=i_plot[0]
                b=i_plot[1]
                ax.broken_barh([(a-extremum[1]+1,1)],(b-extremum[3]+1,1),facecolors='whitesmoke')
                plt.title('$Cluster$ $Color$ $with$ $All$ $Nodes$')
    else:
        plt.title('$Cluster$ $Color$ $with$ $Winner$ $Nodes$')
          
    return fig_legend

Fig_dis=Type_cluster_legend(group_end_near,smooth_net[1],station_code_node,after_group_node,'Win')
Fig_dis.show()

Fig_dis=Type_cluster_legend(group_end_near,smooth_net[1],station_code_node,after_group_node)
Fig_dis.show()

#win节点集合
win_node_list=list(station_code_node.keys())

#win节点向量集合
win_node_vector_dict={}
for i_dict in station_code_node:
    win_node_vector_dict[i_dict]=smooth_net[0][i_dict]
   
#节点与向量字典
def Node_vector(node_dict,i_num):
    node_vector_i={}
    for i_node in node_dict:
        node_vector_i[i_node]=(node_dict[i_node])[i_num]
    return node_vector_i

#属性与节点向量字典
def Para_node(para_data,node_vector_dict):
    para_node_vector={}
    for i_para in range(len(para_data)):
        para_node_vector[para_data[i_para]]=Node_vector(node_vector_dict,i_para)
    return para_node_vector

para_node_vector_dict=Para_node(Gsom_Para,smooth_net[0])

para_node_vector_dict_win=Para_node(Gsom_Para,win_node_vector_dict)

#属性与向量集合字典
def Para_vector(pardetor_dict):
    para_vector_dict={}
    for i_dict in pardetor_dict:
        para_vector_dict[i_dict]=[]
        for j_dict in pardetor_dict[i_dict]:
            para_vector_dict[i_dict].append(pardetor_dict[i_dict][j_dict])
        para_vector_dict[i_dict]=np.array(para_vector_dict[i_dict])
    return para_vector_dict

#返回分位值
def Return_Quartile(np_data):
    min_num=np_data.min(axis=0)
    one_num=np.percentile(np_data,25,axis=0)
    half_num=np.percentile(np_data,50,axis=0)
    three_num=np.percentile(np_data,75,axis=0)
    max_num=np_data.max(axis=0)
    num_list=[min_num,one_num,half_num,three_num,max_num]
    return num_list

#返回级别
def Return_class(np_data,num_i):
    min_num=np_data.min(axis=0)
    one_num=np.percentile(np_data,25,axis=0)
    half_num=np.percentile(np_data,50,axis=0)
    three_num=np.percentile(np_data,75,axis=0)
    max_num=np_data.max(axis=0)
    if num_i>=three_num:
        return 3
    elif num_i>=half_num:
        return 2
    elif num_i>=one_num:
        return 1
    else:
        return 0

#获得每个属性的对应的最值
def Para_num(data_value):
    para_num_min=data_value.min(axis=0)
    para_num_max=data_value.max(axis=0)
    return para_num_max,para_num_min

#反归一公式
def Anti_norm(num,min_num,max_num):
    return num*(max_num-min_num)+min_num
 
#最终的属性图(反归一向量值)
def End_para_figure_change(para_data,posi_list,panove_dict,pave_dict,data_value):
    count_num=int(np.sqrt(len(para_data)))+1
    extremum=Extremum(posi_list)
    fig_para,axes=plt.subplots()
    fig_para.subplots_adjust(top=0.92,left=0.07,right=0.94,hspace=0.3,wspace=0.2)
    axes.set_xlim(0,extremum[0]-extremum[1]+3)
    axes.set_ylim(0,extremum[2]-extremum[3]+3)
    num_para=Para_num(data_value)
    axes.grid(False)
    color_list=['lightgreen','olivedrab','darkolivegreen','black']
    #color_list=['gainsboro','silver','dimgray','black']
    for i_para in range(len(para_data)):                        #每一个属性
        ax=plt.subplot(count_num,count_num,i_para+1)
        ax.set_xlim(0,extremum[0]-extremum[1]+3)
        ax.set_ylim(0,extremum[2]-extremum[3]+3)
        ax.set_axis_off()
        for j_para in panove_dict[para_data[i_para]]:           #属性的每一个二维坐标
            a=j_para[0]
            b=j_para[1]
            single_para_data=para_data[i_para]
            col_sign=Return_class(pave_dict[single_para_data],panove_dict[single_para_data][j_para])
            ax.broken_barh([(a-extremum[1]+1,1)],(b-extremum[3]+1,1),\
                           facecolors=color_list[col_sign])
        plt.title('%s'%single_para_data)
        label_num=Return_Quartile(pave_dict[para_data[i_para]])
        ax.scatter(-2,0,c='%s'%color_list[0],label='%.2e'%Anti_norm(label_num[0],num_para[1][i_para],num_para[0][i_para]),alpha=0.7,marker='s')
        ax.scatter(-2,0,c='%s'%color_list[1],label='%.2e'%Anti_norm(label_num[1],num_para[1][i_para],num_para[0][i_para]),alpha=0.7,marker='s')
        ax.scatter(-2,0,c='%s'%color_list[2],label='%.2e'%Anti_norm(label_num[2],num_para[1][i_para],num_para[0][i_para]),alpha=0.7,marker='s')
        ax.scatter(-2,0,c='%s'%color_list[3],label='%.2e'%Anti_norm(label_num[3],num_para[1][i_para],num_para[0][i_para]),alpha=0.7,marker='s')
        ax.legend(shadow=True,fontsize=8,bbox_to_anchor=(1.253,0.6))
    return fig_para

Para_all_vector_change=End_para_figure_change(Gsom_Para,smooth_net[1],para_node_vector_dict_win,Para_vector(para_node_vector_dict_win),Gsom_input)
Para_all_vector_change.show()









#站点与真实数据指标字典
Real_station_data={Gsom_tag[i_tag]:Gsom_input[i_tag] for i_tag in range(len(Gsom_tag))}

#节点向量真实数据集合
def Real_node_vector_dict(win_node_sta,sta_real_data):
    real_win_node_vector_dict={}
    for i_real in win_node_sta:
        vector_set=[]
        for j_real in win_node_sta[i_real]:
            vector_set.append(list(sta_real_data[j_real]))
        real_win_node_vector_dict[i_real]=vector_set
    return real_win_node_vector_dict

real_Node_vector=Real_node_vector_dict(station_code_node,Real_station_data)

#平均值节点向量数据组合
def Mean_node_vector(real_dict_para):
    mean_node_vector={}
    for i_mean in real_dict_para:
        if len(real_dict_para[i_mean])==1:
            mean_node_vector[i_mean]=np.array(real_dict_para[i_mean][0])
        else:
            np_list=np.array(real_dict_para[i_mean])
            mean_list=np_list.mean(axis=0)
            mean_node_vector[i_mean]=np.array(mean_list)
    return mean_node_vector

mean_real_Node_vector=Mean_node_vector(real_Node_vector)


#十六进制的控制函数
def Convert_hex(ten_num):
    hex_num=str(hex(ten_num))
    if len(hex_num)==3:
        return '0'+hex_num[2:]
    else:
        return hex_num[2:]

#颜色的十进制表示函数
def Decimal_list(color_num):
    Color_list_decimal=[]
    all_color_list=[[r,g,b] for r in [0,color_num] \
                    for g in [0,color_num] for b in [0,color_num]]
    for i_sin in all_color_list:
        if i_sin.count(i_sin[0])!=3:
            Color_list_decimal.append(i_sin)
    return Color_list_decimal
            

#转换数字函数字典
def Number_change(list_num,a_up,b_low,color_num=8):
    length_np=len(list_num)
    copy_list=list_num.copy()
    length_lis=len(list_num)
    s=int((a_up-b_low)/length_lis)
    if s>color_num:
        num_list=list(range(a_up,b_low,-color_num))
    elif s==1:
        num_list=list(range(a_up,b_low,-1))
    else:
        num_list=list(range(a_up,b_low,1-s))   
    copy_list.sort()
    number_numcolor_dict={copy_list[i]:num_list[:length_lis][i] for i in range(length_lis)}
    return number_numcolor_dict


#定义颜色块位置字典
def Location_color(list_num,y_max):
    copy_list=list_num.copy()
    length_key=len(list_num)
    float_height=y_max/length_key
    local_list=[]
    for i_height in range(length_key):
        local_list.append(i_height*float_height)
    max_loca=max(local_list)
    copy_list.sort()
    location_numcolor_dict={copy_list[i]:local_list[i] for i in range(length_key)}
    odd_location_numcolor_dict={copy_list[i]:i for i in range(length_key)}
    return location_numcolor_dict,float_height,max_loca,odd_location_numcolor_dict
      

#色系扩展属性图
def Color_plus_para(posi_list,win_node_sw,node_vect,para_data,color_name=250,color_list_index=4):
    decimal_color_list=Decimal_list(color_name)#颜色控制函数
    color_list=decimal_color_list[color_list_index]#色系选择
    count_num=int(np.sqrt(len(para_data)))+1
    extremum=Extremum(posi_list)
    fig_para,axes=plt.subplots()
    fig_para.subplots_adjust(top=0.92,left=0.07,right=0.94,hspace=0.3,wspace=0.2)
    axes.set_xlim(0,extremum[0]-extremum[1]+5)
    axes.set_ylim(0,extremum[2]-extremum[3]+5)
    axes.grid(False)
    for i_para in range(len(para_data)):#每一个属性
        single_para_data=para_data[i_para]
        ax=plt.subplot(count_num,count_num,i_para+1)
        ax.set_xlim(0,extremum[0]-extremum[1]+5)
        ax.set_ylim(0,extremum[2]-extremum[3]+5)
        ax.set_axis_off()
        para_num_list=[]
        for i_win in win_node_sw:
            para_num_list.append(node_vect[i_win][i_para])
        color_name_list=Number_change(para_num_list,color_name,50,color_num=10)
        color_location_list_all=Location_color(para_num_list,extremum[2]-extremum[3]+5)
        color_location_list=color_location_list_all[0]
        for j_para in win_node_sw:
            a=j_para[0]
            b=j_para[1]
            R,G,B=color_list[0],color_list[1],color_list[2]
            if R==0:
                R=color_name_list[node_vect[j_para][i_para]]
                        
            if G==0:
                G=color_name_list[node_vect[j_para][i_para]]
                        
            if B==0:
                B=color_name_list[node_vect[j_para][i_para]]
            color_set=Convert_hex(R)+Convert_hex(G)+Convert_hex(B)
            ax.broken_barh([(a-extremum[1]+1,1)],(b-extremum[3]+1,1),\
                              facecolors='#%s'%color_set)
            
            ax.broken_barh([(extremum[0]-extremum[1]+3,0.5)],(color_location_list[node_vect[j_para][i_para]],\
                                                              color_location_list_all[1]), facecolors='#%s'%color_set)
            if color_location_list_all[3][node_vect[j_para][i_para]]%2==0:
                ax.annotate('%.2e'%node_vect[j_para][i_para], (extremum[0]-extremum[1]+3,0.5),\
                            xytext=((extremum[0]-extremum[1]+7)/(extremum[0]-extremum[1]+5),\
                                    (color_location_list[node_vect[j_para][i_para]]+color_location_list_all[1])/(extremum[2]-extremum[3]+5)),\
                            textcoords='axes fraction',fontsize=12,horizontalalignment='right', verticalalignment='top')

            
        plt.title('%s'%single_para_data)
    return fig_para
    
big_small_diff=Color_plus_para(smooth_net[1],station_code_node,mean_real_Node_vector,Gsom_Para)

big_small_diff.show()



'''
实现不同类别大差异，同一类别小差异

#分组节点向量组合
def Group_node_vector(group_list,node_vector):
    vect_list=[]
    set_item=[]
    num_item=1
    for i_group in group_list:
        if i_group not in set_item:
            vect_list.append(node_vector[i_group])
            num_item+=1
            set_item.append(i_group)
    vect_np=np.array(vect_list)
    if num_item==1:
        return vect_np
    else:
        transpose_vect_np=np.transpose(vect_np)
        return transpose_vect_np
#颜色的十进制表示函数
def Decimal_list(end_para=80,diff_para=50):
    Color_list_decimal=[]
    for i_color in range(255,end_para,-diff_para):
        all_color_list=[[r,g,b] for r in [0,i_color] for g in [0,i_color] for b in [0,i_color]]
        for i_sin in all_color_list:
            if i_sin.count(i_sin[0])!=3:
                Color_list_decimal.append(i_sin)
    return Color_list_decimal
                
decimal_color_list=Decimal_list()#颜色控制函数

#大小差异图
def Big_small_class(posi_list,class_group,node_vector,para_data,color_list=decimal_color_list):
    count_num=int(np.sqrt(len(para_data)))+1
    extremum=Extremum(posi_list)
    fig_para,axes=plt.subplots()
    fig_para.subplots_adjust(top=0.92,left=0.07,right=0.94,hspace=0.3,wspace=0.2)
    axes.set_xlim(0,extremum[0]-extremum[1]+3)
    axes.set_ylim(0,extremum[2]-extremum[3]+3)
    axes.grid(False)
    for i_para in range(len(para_data)):                        #每一个属性
        ax=plt.subplot(count_num,count_num,i_para+1)
        ax.set_xlim(0,extremum[0]-extremum[1]+3)
        ax.set_ylim(0,extremum[2]-extremum[3]+3)
        ax.set_axis_off()
        for i_class  in range(len(class_group)):#每一个类别
            color_list_sigle=color_list[i_class]#实现大差异颜色
            class_list=class_group[i_class]#区分大类别的list
            para_num=Group_node_vector(class_list,node_vector)[i_para]#大类别中的标签参数
            if len(para_num)!=1:
                color_num_list=Number_change(para_num,250,50)
                for j_para in class_group[i_class]:           #类别中的每一个赢节点
                    a=j_para[0]
                    b=j_para[1]
                    single_para_data=para_data[i_para]
                    R,G,B=color_list_sigle[0],color_list_sigle[1],color_list_sigle[2]
                    if R==0:
                        R=color_num_list[node_vector[j_para][i_para]]
                        
                    if G==0:
                        G=color_num_list[node_vector[j_para][i_para]]
                        
                    if B==0:
                        B=color_num_list[node_vector[j_para][i_para]]

                    color_set=Convert_hex(R)+Convert_hex(G)+Convert_hex(B)
                    ax.broken_barh([(a-extremum[1]+1,1)],(b-extremum[3]+1,1),\
                                   facecolors='#%s'%color_set)
            else:
                for j_para in class_group[i_class]:           #类别中的一个赢节点
                    a=j_para[0]
                    b=j_para[1]
                    single_para_data=para_data[i_para]
                    R,G,B=color_list_sigle[0],color_list_sigle[1],color_list_sigle[2]
                    color_set=Convert_hex(R)+Convert_hex(G)+Convert_hex(B)

                    ax.broken_barh([(a-extremum[1]+1,1)],(b-extremum[3]+1,1),\
                                   facecolors='#%s'%color_set)
         
        plt.title('%s'%single_para_data)
        #label_num=Return_Quartile(pave_dict[para_data[i_para]])
        #ax.scatter(-2,0,c='%s'%color_list[0],label='%.2e'%Anti_norm(label_num[0],num_para[1][i_para],num_para[0][i_para]),alpha=0.7,marker='s')
        #ax.scatter(-2,0,c='%s'%color_list[1],label='%.2e'%Anti_norm(label_num[1],num_para[1][i_para],num_para[0][i_para]),alpha=0.7,marker='s')
        #ax.scatter(-2,0,c='%s'%color_list[2],label='%.2e'%Anti_norm(label_num[2],num_para[1][i_para],num_para[0][i_para]),alpha=0.7,marker='s')
        #ax.scatter(-2,0,c='%s'%color_list[3],label='%.2e'%Anti_norm(label_num[3],num_para[1][i_para],num_para[0][i_para]),alpha=0.7,marker='s')
        #ax.legend(shadow=True,fontsize=8,bbox_to_anchor=(1.253,0.6))
    return fig_para
    
big_small_diff=Big_small_class(smooth_net[1],group_end_near,mean_real_Node_vector,Gsom_Para)

big_small_diff.show()
'''





















