"""
==============================================================
★此代码库切勿外传★

代码库功能更新中......

使用过程中若发现BUG或有什么修改建议，请及时联系代码包装人员
==============================================================
"""

"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
使用方法：

① 将Codebase.py文件放入工程文件根目录

② import Codebase 或者 from Codebase import XXX

③ Codebase.XXX() 或者 XXX()
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

"""
【目录】
1. PlotDCA() 绘制决策曲线
2. KFold_Validation() k折交叉验证
3. Valuestandard() 数据标准化
4. ConfusionMatrix() 混淆矩阵
5. PlotROC() 绘制ROC曲线
6. PlotPRC() 绘制PRC曲线
7. DelongTest() Delong检验
8. PlotLearningCurve() 绘制学习曲线
9. CompareModel() 模型比较
10. PlotCC() 校准曲线
11. NNCalibrationModel() 神经网络概率校准后的伪模型
12. Plot_Learning_Curve() 绘制学习曲线,神经网络除外
13. Bootstrap_PRC() 自助法检验两PRC曲线下面积的差异是否显著

待更新....
"""

"""======================== Import ========================"""
import numpy as np
import pandas as pd
import copy
import os
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import brier_score_loss
from tensorflow.keras.metrics import AUC
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LinearRegression
import math
import scipy.stats
from scipy import stats
from scipy.stats import chi2
from keras.models import load_model
import warnings

"""======================== PlotDCA ========================"""
class PlotDCA:
    """
    决策曲线 3.0（Decision Curve Analysis, DCA）的绘制；需要ConfusionMatrix7.2及以上做为前置
    
    参数介绍
    ========
    Model_Name_List：是所拟合模型和模型命名字符串的组合列表，如：Model_Name_List = [(rf, 'RF'),(dnn,'DNN')]，默认为None
    ========
    
    方法介绍
    ========
    plot()方法：用于决策曲线的绘制，是除构造方法init外，其他所有方法的先决方法（需要最先被调用），具体见plot()说明
    
    save()方法：用于保存绘制的决策曲线，需要先调用plot()方法（即先要有图才能保存），具体见save()说明
    
    value()方法：用于展示决策曲线绘制过程的详细数据，pt、TP、FP、Net Benefit的值，需要先调用plot()方法，具体见value()说明
    ========
    """
    
    def __init__(self, Model_Name_List=None):
        if type(Model_Name_List) != type([]):
            self.__init_state = False
            raise ValueError("需要参数“Model_Name_List”，且类型为“list”")
        else:
            self.__init_state = True
            self.__plot_state = False
            self.__model_list = Model_Name_List
            
        return None

    def plot(self, y_data=None, color_list=None, x_max=1.0, y_max=1.0, xlabel='Threshold Probability', ylabel='Net Benefit', 
             title='Decision Curve Analysis of Models'):
        
        """
        plot方法用于DCA的绘制
        
        参数介绍
        ========
        color_list：模型决策曲线的颜色，类型为list，其数目至少等于模型数量，默认为None
        
        x_max：决策曲线图横坐标的的最大值，类型为float，默认为1.0
        
        y_max：决策曲线图纵坐标的的最大值，类型为float，默认为1.0
        
        xlabel：X轴的名字，类型为str，默认为“Threshold Probability”
        
        ylabel：Y轴的名字，类型为str，默认为“Net Benefit”
        
        title：标题的名称，类型为str，默认为“Decision Curve Analysis of Models”
        ========
        """
        
        self.__plot_state = False
        
        if type(y_data) == type(None):
            raise ValueError("需要参数“y_data”")
        else:
            self.__y = y_data
            
        self.__x_max = x_max
        self.__y_max = y_max
        self.__xlabel = xlabel
        self.__ylabel = ylabel
        self.__title = title
        
        if type(color_list) == type(None):
            self.__defultcolor = True
            if len(self.__model_list) <= 8:
                self.__color_list = ['#f68993', '#54de98', '#fbfc7e', '#f1a974', '#d0f37c', '#5579fe', '#e79dfa', '#8bf0f1']
                self.__colorstate = False
            else:
                self.__colorstate = True
        else:
            if len(self.__model_list) > len(color_list):
                raise ValueError("颜色数量应大于等于模型数量，模型数量为%s，颜色的数量仅为%s" % (len(self.__model_list),len(color_list)))
            else:
                self.__defultcolor = False
                self.__color_list = color_list
        
        plt.close()
        plt.figure(figsize=(6.4, 4.8), dpi=300)
        
        self.__cycle = 0
        self.__detailed_value = {}
        self.__pt_list = []
        self.__jd_list = []
        self.__index = []
        for i in range(0,100,1):
            self.__pt_value = i/100
            self.__jd_value = (np.sum(self.__y)-(len(self.__y)-np.sum(self.__y))*self.__pt_value/(1-self.__pt_value))/len(self.__y)
            self.__pt_list.append(self.__pt_value)
            self.__jd_list.append(self.__jd_value)
        
        for self.__cm, self.__name in self.__model_list:
            self.__proba_model = self.__cm.dca()
            self.__y_proba = np.copy(self.__proba_model)
                
            self.__TP_list = []
            self.__FP_list = []
            self.__NB_list = []
            self.__Y = self.__y
            self.__index.append(self.__name)
            
            self.__y_proba = self.__y_proba.ravel()
            
            for m in range(0,100,1):
                self.__pt = m/100
                self.__y_pred = np.zeros(self.__y_proba.shape[0])
                for n in range(self.__y_proba.shape[0]):
                    if self.__y_proba[n] >= self.__pt:
                        self.__y_pred[n] = 1
                    else:
                        self.__y_pred[n] = 0
                        
                MC = confusion_matrix(self.__Y,self.__y_pred)
                self.__TP = MC[1,1]
                self.__FP = MC[0,1]
                self.__NB = (self.__TP-(self.__FP * self.__pt/(1-self.__pt)) )/self.__Y.shape[0]
                
                self.__TP_list.append(self.__TP)
                self.__FP_list.append(self.__FP)
                self.__NB_list.append(self.__NB)
                
            #模型的净收益
            if self.__defultcolor:
                if self.__colorstate:
                    plt.plot(self.__pt_list, self.__NB_list, lw=1, linestyle='-',label=self.__name)
                else:
                    plt.plot(self.__pt_list, self.__NB_list, color=self.__color_list[self.__cycle], lw=1, linestyle='-',label=self.__name)
            else:
                plt.plot(self.__pt_list, self.__NB_list, color=self.__color_list[self.__cycle], lw=1, linestyle='-',label=self.__name)
            
            self.__dictionary = {'阈值（pt）':self.__pt_list, 'TP':self.__TP_list, 'FP':self.__FP_list, 'Net Benefit(NB)':self.__NB_list}
            self.__dataframe = pd.DataFrame(self.__dictionary)
            self.__detailed_value.update({self.__name: self.__dataframe})
            self.__cycle = self.__cycle + 1
            
        #所有人都不治疗的净收益
        plt.plot(self.__pt_list, np.zeros(len(self.__pt_list)), color='#383c3b', lw=1, linestyle='--',label='None')
        #所有人都治疗的净收益
        plt.plot(self.__pt_list, self.__jd_list, color='#21336e', lw=1, linestyle='dotted',label='All')
        
        
        plt.xlim([0.0, self.__x_max])
        plt.ylim([-0.1, self.__y_max])
        plt.xlabel(self.__xlabel)
        plt.ylabel(self.__ylabel)
        plt.title(self.__title)
        plt.legend(loc="best")
        self.__fig = plt.gcf()
        plt.show()
        
        self.__plot_state = True
        
        return None
    
    def save(self, path="DCA.png", dpi=300):
        
        """
        save方法用于保存绘制的决策曲线
        
        参数介绍
        ========
        
        path：保存路径，类型为str，格式为“/file1/file2/image_name.png”；默认保存在当前路径，名字为"DCA.png"
        
        dpi：每英寸上，所能印刷的网点数，类型为int，默认为300
        
        ========
        """
        
        if self.__plot_state == False:
            raise RuntimeError("尚未绘制决策曲线图，请先调用plot()方法绘制")
        
        self.__path = path
        self.__dpi = dpi
        
        self.__fig.savefig(self.__path, dpi=self.__dpi, bbox_inches = 'tight')
        print("决策曲线已保存，路径为：%s" % self.__path)
        
        return None
    
    def value(self, m_name=None):
        
        """
        value方法用于显示绘图过程中的详细数据
        
        参数介绍
        ========
        m_name：所要查询的模型的名字，类型为str，与对象实例化时，传入的“Model_Name_List”一致，或者为“all”，默认为None
        ========
        """
        
        if type(m_name) == type(None):
            raise ValueError("需要参数“m_name”")
        elif self.__plot_state == False:
            raise RuntimeError("尚未绘制决策曲线图，请先调用plot()方法绘制")
        else:
            self.__mname = m_name
            
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
            
        if self.__mname != "all":
            print("%s的决策曲线详细数据如下所示：" % self.__mname)
            print(" ")
            display(self.__detailed_value[self.__mname])
        else:
            for d in range(0, len(self.__detailed_value)):
                print("%s的决策曲线详细数据如下所示：" % self.__index[d])
                print(" ")
                display(self.__detailed_value[self.__index[d]])
                print(" ")
                
        return None
        
"""========================================================="""

"""===================== KFoldValidation ==================="""
class KFoldValidation():
    """
    KFoldValidation，用于验证在特定参数情况下，模型是是否过拟合，评估标准为Accuracy
    
    参数介绍
    ========
    model：所搭建的，具有fit方法的，待验证的模型，默认为None；★注意：若 dnn_model=True ，该参数失效，并需传入dnn_path参数
    
    X：训练集的数据，类型为DataFrame或Numpy数组，默认为None
    
    Y：训练集的标签，类型为DataFrame、Series或Numpy数组，默认为None
    
    k：为折叠的次数，类型为int，默认为 3
    
    r：为随机种子，类型为int，默认为None
    
    dnn_model：是否为基于keras的神经网络模型，类型为bool，默认为False；若为True，则需传入dnn_path参数
    
    |++++ 以下参数只有 dnn_model=Ture 时有用 ++++|
    
    dnn_path：实例化且未fit的神经网络模型所存储的位置，类型为str，默认为None；★当 dnn_model=Ture 时的必须参数！
    
    dnn_batch_size：一次训练所抓取的数据样本数量，类型为int，默认为64
    
    dnn_epochs：神经网络训练的迭代次数，类型为int，默认为50
    
    dnn_shuffle：打乱，类型为str，默认为'True'
    
    dnn_verbos：是否显示每次迭代的进度条，默认为0，即不显示
    
    dnn_weight：阳性样本权重,类型为int，默认为1
    
    ========
    
    """
    
    def __init__(self, model=None, x=None, y=None, k=3, r=None, dnn_model=False, dnn_path=None, 
                 dnn_batch_size=64, dnn_epochs=50, dnn_shuffle='True', dnn_verbos=0, dnn_weight=1):
        
        # 参数检查
        # 1.常规参数检查
        # model
        if dnn_model:
            self.__dnn = True
            if type(dnn_path) == type('dnn path'):
                self.__model = copy.deepcopy(dnn_path)
            else:
                raise ValueError("dnn_path参数错误，请检查，需要的类型为str（传入的类型为：%s）" % type(dnn_path))
        else:
            self.__dnn = False
            if type(model) == type(None):
                raise ValueError("需要 model 参数！")
            else:
                self.__model = copy.deepcopy(model)
        # x        
        if type(x) == type(None):
            raise ValueError("需要 x 参数！")
        else:
            if type(x) == type(pd.DataFrame()):
                self.__x = copy.deepcopy(x.values)
            else:
                if type(x) == type(np.array([])):
                    self.__x = copy.deepcopy(x)
                else:
                    raise ValueError("x 参数类型错误，请检查！x 的类型为DataFrame或Numpy数组，而传入类型为：%s" % type(x))
        # y
        if type(y) == type(None):
            raise ValueError("需要 y 参数！")
        else:
            if type(y) == type(pd.DataFrame()):
                self.__y = copy.deepcopy(y.values)
            else:
                if type(y) == type(pd.Series(1)):
                    self.__y = copy.deepcopy(y.values)
                else:
                    if type(y) == type(np.array([])):
                        self.__y = copy.deepcopy(y)
                    else:
                        raise ValueError("y 参数类型错误，类型为DataFrame、Series或或Numpy数组，传入类型为：%s" % type(x))
                        
        # k
        if type(k) == type(int(1)):
            self.__k = k
        else:
            raise ValueError("k 参数类型错误，请检查！k 的类型为int，而传入类型为：%s" % type(k))
        
        # r
        if type(r) == type(None):
            self.__r = r
        else:
            if type(r) == type(int(1)):
                self.__r = r
            else:
                raise ValueError("r 参数类型错误，请检查！r 的类型为int，而传入类型为：%s" % type(r))
        
        # 2.Keras相关参数检查
        # dnn_batch_size
        if type(dnn_batch_size) == type(int(64)):
            self.__batch_size = dnn_batch_size
        else:
            raise ValueError("dnn_batch_size的类型必须为int，而传入的类型为%s" % type(dnn_batch_size))
        
        # dnn_epochs
        if type(dnn_epochs) == type(int(100)):
            self.__epochs = dnn_epochs
        else:
            raise ValueError("dnn_epochs的类型必须为int，而传入的类型为%s" % type(dnn_epochs))
            
        # dnn_shuffle
        if  type(dnn_shuffle) == type("False"):
            if dnn_shuffle == "False":
                self.__shuffle = dnn_shuffle
            else:
                if dnn_shuffle == "True":
                    self.__shuffle = dnn_shuffle
                else:
                    raise ValueError("dnn_shuffle的类容为“True”或“False”")
        else:
            raise ValueError("dnn_shuffle的类型必须为str")
            
        # dnn_verbos    
        self.__verbose = dnn_verbos
            
        # dnn_weight
        if type(dnn_weight) == type(int(1)):
            self.__weight = dnn_weight
        else:
            if type(dnn_weight) == type(float(0.1)):
                    self.__weight = dnn_weight
            else:
                raise ValueError("dnn_weight的类型必须为int或float")
                
        self.__switch = False
        
        return None
    
    def once(self):
        """onece()方法用于进行1次K折交叉验证"""
        
        # 变量初始化
        self.__k_times = 1
        self.__train_acc_s = 0
        self.__test_acc_s = 0
        self.__Model = None
        
        self.__k_list = []
        self.__train_acc = []
        self.__test_acc = []
        
        # 数据集划分
        self.__data = StratifiedKFold(n_splits=self.__k, shuffle=True, random_state=self.__r).split(self.__x, self.__y)
        
        for self.__train, self.__test in self.__data:
            
            # k列表
            if self.__k_times == 1:
                self.__k_list.append('%s st' % self.__k_times)
            elif self.__k_times == 2:
                self.__k_list.append('%s nd' % self.__k_times)
            elif self.__k_times == 3:
                self.__k_list.append('%s rd' % self.__k_times)
            else:
                self.__k_list.append('%s th' % self.__k_times)
                
            # 数据集附于
            self.__x_train, self.__x_test = self.__x[self.__train], self.__x[self.__test]
            self.__y_train, self.__y_test = self.__y[self.__train], self.__y[self.__test]
            
            # 避免模型被重复fit和权重更新
            assert self.__Model == None, "代码有Bug，请联系打包者"
            if self.__dnn:
                self.__Model = load_model(self.__model)
            else:
                self.__Model = copy.deepcopy(self.__model)
                
            # 训练模型
            if self.__dnn:
                self.__Model.fit(self.__x_train,self.__y_train, batch_size=self.__batch_size, epochs=self.__epochs, 
                                 shuffle=self.__shuffle, verbose=self.__verbose, class_weight={0:1,1:self.__weight})
            else:
                self.__Model.fit(self.__x_train,self.__y_train)
                
            
            # 训练集准确度
            if self.__dnn:
                self.__y_predict_train = self.__Model.predict(self.__x_train)
                self.__y_predict_train[self.__y_predict_train >= 0.5] = 1
                self.__y_predict_train[self.__y_predict_train < 0.5] = 0
            else:
                self.__y_predict_train = self.__Model.predict(self.__x_train)
                
            self.__acc_train = accuracy_score(self.__y_train, self.__y_predict_train)
            self.__train_acc_s = self.__train_acc_s + self.__acc_train
            self.__train_acc.append(self.__acc_train)
            
            # 验证集准确度
            if self.__dnn:
                self.__y_predict_test = self.__Model.predict(self.__x_test)
                self.__y_predict_test[self.__y_predict_test >= 0.5] = 1
                self.__y_predict_test[self.__y_predict_test < 0.5] = 0
            else:
                self.__y_predict_test = self.__Model.predict(self.__x_test)
                
            self.__acc_test = accuracy_score(self.__y_test, self.__y_predict_test)
            self.__test_acc_s = self.__test_acc_s + self.__acc_test
            self.__test_acc.append(self.__acc_test)
            
            self.__k_times = self.__k_times + 1
            self.__Model = None
            
        self.__k_times = self.__k_times - 1
        
        self.__train_mean = self.__train_acc_s / self.__k_times
        self.__test_mean = self.__test_acc_s / self.__k_times
        
        self.__column_1 = self.__k_list+[" "]+["平均值"]
        self.__column_2 = self.__train_acc+[" "]+[self.__train_mean]
        self.__column_3 = self.__test_acc+[" "]+[self.__test_mean]
        
        self.__data_dictionary = {'KFlod Times':self.__column_1, '训练集Accuracy':self.__column_2, '测试集Accuracy':self.__column_3}
        self.__data_frame = pd.DataFrame(self.__data_dictionary)
        
        if self.__switch:
            return self.__train_mean, self.__test_mean
        else:
            display(self.__data_frame)
            return None
        
    
    def cycle(self, times=20):
        """
        cycle()方法用于多次进行K折交叉验证
        
        参数介绍
        ========
        times：循环的次数，类型为int，默认为20
        
        ========
        """
        # 参数检查
        if type(times) == type(int(20)):
            self.__times = times
        else:
            raise ValueError("times 参数类型错误，类型为int，传入类型为：%s" % type(times))
            
        # 变量初始化
        self.__a_train_s = 0
        self.__a_test_s = 0
        
        self.__c_list = []
        self.__a_train = []
        self.__a_test = []
        
        # 关闭once展示
        self.__switch = True
        
        # 循环
        for self.__i in range(0, self.__times):
            self.__a1, self.__a2 = self.once()
            self.__a_train_s = self.__a_train_s + self.__a1
            self.__a_test_s = self.__a_test_s + self.__a2
            self.__a_train.append(self.__a1)
            self.__a_test.append(self.__a2)
            if self.__i == 0:
                self.__c_list.append('1 st')
            elif self.__i == 1:
                self.__c_list.append('2 nd')
            elif self.__i == 2:
                self.__c_list.append('3 rd')
            else:
                self.__j = self.__i + 1
                self.__c_list.append('%s th' % self.__j)
                
        # 求平均
        self.__train_c = self.__a_train_s / self.__times
        self.__test_c = self.__a_test_s / self.__times
        
        # 输出
        self.__column_4 = self.__c_list+[" "]+["平均值"]
        self.__column_5 = self.__a_train+[" "]+[self.__train_c]
        self.__column_6 = self.__a_test+[" "]+[self.__test_c]
        
        self.__data_dictionary_1 = {'Cycle Times':self.__column_4, '训练集准确度':self.__column_5, '验证集准确度':self.__column_6}
        self.__data_frame_1 = pd.DataFrame(self.__data_dictionary_1)
        display(self.__data_frame_1)
        
        # 还原
        self.__switch = False
        
        return None
    
"""========================================================="""
    
"""===================== Valuestandard ====================="""
class Valuestandard:
    """
    Valuestandard用于数据标准化
    
    参数介绍
    ========
    skip_list：当Skip为True时才生效，表示不需要进行标准化的变量的序号列表，类型为list，默认为None
    
    onehot_list：当OneHot为True时才生效，表示需要进行独热编码的变量的序号列表，类型为list，默认为None
    
    categories_list：当OneHot为True时才生效，表示需要进行独热编码的变量的种类数列表，与onehot_list相互对应，类型为list，默认为None
    
    Skip：是否跳过部分变量，不对其进行标准化，如：二分类变量等，类型为bool，默认为False，即将全部变量标准化
    
    OneHot：是否使用独热编码，类型为bool，默认为False，即不使用独热编码
    ========
    """
    
    def __init__(self, skip_list=None, onehot_list=None, categories_list=None, Skip=False, OneHot=False):
        
        self.__Skip = Skip
        self.__OneHot = OneHot
        
        if self.__Skip:
            if type(skip_list) == type(None):
                raise ValueError("需要参数：skip_list")
            elif type(skip_list) != type([]):
                raise ValueError("参数“skip_list”的类型必须为：list")
            else:
                self.__list1 = skip_list
        
        if self.__OneHot:
            if type(onehot_list) == type(None):
                raise ValueError("需要参数：onehot_list")
            elif type(onehot_list) != type([]):
                raise ValueError("参数“onehot_list”的类型必须为：list")
            else:
                self.__list2 = onehot_list
        
        if self.__OneHot:
            if type(categories_list) == type(None):
                raise ValueError("需要参数：categories_list")
            elif type(categories_list) != type([]):
                raise ValueError("参数“categories_list”的类型必须为：list")
            else:
                for self.__k in range(len(categories_list)):
                    if self.__k == 0:
                        self.__list3 = [list(range(categories_list[self.__k]))]
                    else:
                        self.__list3.append(list(range(categories_list[self.__k])))
                
        self.__fitstate = False
        self.__fitstate_2 = False
        
        return None
    
    def fit_transform(self, data_1=None):
        """
        fit_transform 方法用于标准化数据
        
        参数介绍
        ========
        data_1：待标准化的数据集，类型为DataFrame，默认为None
        ========
        """
        
        if self.__fitstate:
            raise RuntimeError("数据已被标准化，请勿重复标准化；若要重新标准化，请重新实例化Valuestandard")
        
        if type(data_1) == None:
            raise ValueError("需要待标准化的数据集：data_1")
        elif type(data_1) != type(pd.DataFrame()):
            raise ValueError("数据集的类型必须为：DataFrame，而传入数据的类型为：%s" % type(data_1))
        else:
            self.__Data_1 = data_1.copy()
            self.__data = self.__Data_1.copy()
            
        if self.__OneHot:
            self.__Oname_list = self.__data.columns[self.__list2]
            self.__data_1 = self.__data[self.__Oname_list].copy()
            self.__data = self.__data.drop(self.__Oname_list, axis=1)
            
            for self.__i in range(len(self.__list2)):
                self.__transdata = self.__data_1[self.__Oname_list[self.__i]].copy()
                self.__quantity = self.__list3[self.__i]
                self.__LE = LabelEncoder()
                self.__transdata = self.__LE.fit_transform(self.__transdata)
                self.__OH = OneHotEncoder(categories=[self.__quantity], sparse=False)
                self.__transdata = self.__OH.fit_transform(self.__transdata.reshape(-1, 1))
                self.__transdata = pd.DataFrame(self.__transdata)
                self.__transdata.rename(columns=lambda s: self.__Oname_list[self.__i] + '_' + str(s), inplace=True)
                self.__transdata = self.__transdata.astype(np.int64)
                
                if self.__i == 0:
                    self.__data_onehot = self.__transdata.copy()
                else:
                    self.__data_onehot = pd.concat([self.__data_onehot, self.__transdata], axis=1)
                
                self.__transdata = None
                self.__quantity = None
                self.__OH = None
                self.__LE = None
                
            self.__Oname_list = None
            self.__data_1 = None
        
        if self.__Skip:
            self.__Sname_list = self.__Data_1.columns[self.__list1]
            self.__data_skip = self.__data[self.__Sname_list].copy()
            self.__data_o2 = self.__data[self.__Sname_list].copy()
            self.__data = self.__data.drop(self.__Sname_list, axis=1)
            if self.__data.shape[1] != 0:
                self.__s = True
            else:
                self.__s = False
            
            self.__Sname_list = None
        else:
            self.__s = True 
            
        if self.__s:
            self.__Dname_list = self.__data.columns
            self.__data_original = self.__data.copy()
            self.__Key = StandardScaler()
            self.__data_standard = self.__Key.fit_transform(self.__data)
            self.__data_standard = pd.DataFrame(self.__data_standard)
            self.__data_standard.columns = self.__Dname_list
            self.__Dname_list = None
        else:
            self.__data_standard = self.__data.copy()
            self.__data_original = self.__data.copy()
        
        if self.__Skip:
            self.__data_standard = pd.concat([self.__data_standard, self.__data_skip], axis=1)
            self.__data_original = pd.concat([self.__data_original, self.__data_o2], axis=1)
            self.__data_skip = None
            self.__data_o2 = None
        
        if self.__OneHot:
            self.__data_standard = pd.concat([self.__data_standard, self.__data_onehot], axis=1)
            self.__data_original = pd.concat([self.__data_original, self.__data_onehot], axis=1)
            self.__data_onehot = None
        
        self.__data = None
        self.__fitstate = True
        
        return None
    
    def transform(self, data_2=None):
        """
        transform 方法用于，以fit_transform的标准，标准化数据，如测试集数据等
        
        参数介绍
        ========
        data_2：待标准化的数据集，类型为DataFrame，默认为None
        
        ★可以调用多次，但只保存最后一次调用的数据★
        ========
        """
        if self.__fitstate != True:
            raise RuntimeError("缺少数据标准化标准，请先调用 fit_transform() 方法")
        
        if type(data_2) == None:
            raise ValueError("需要待标准化的数据集：data_2")
        elif type(data_2) != type(pd.DataFrame()):
            raise ValueError("数据集的类型必须为：DataFrame，而传入数据的类型为：%s" % type(data_2))
        else:
            self.__Data_2 = data_2.copy()
            self.__data = self.__Data_2.copy()
            
        if self.__OneHot:
            self.__Oname_list = self.__data.columns[self.__list2]
            self.__data_1 = self.__data[self.__Oname_list].copy()
            self.__data = self.__data.drop(self.__Oname_list, axis=1)
            
            for self.__i in range(len(self.__list2)):
                self.__transdata = self.__data_1[self.__Oname_list[self.__i]].copy()
                self.__quantity = self.__list3[self.__i]
                self.__LE = LabelEncoder()
                self.__transdata = self.__LE.fit_transform(self.__transdata)
                self.__OH = OneHotEncoder(categories=[self.__quantity], sparse=False)
                self.__transdata = self.__OH.fit_transform(self.__transdata.reshape(-1, 1))
                self.__transdata = pd.DataFrame(self.__transdata)
                self.__transdata.rename(columns=lambda s: self.__Oname_list[self.__i] + '_' + str(s), inplace=True)
                self.__transdata = self.__transdata.astype(np.int64)
                
                if self.__i == 0:
                    self.__data_onehot = self.__transdata.copy()
                else:
                    self.__data_onehot = pd.concat([self.__data_onehot, self.__transdata], axis=1)
                
                self.__transdata = None
                self.__quantity = None
                self.__OH = None
                self.__LE = None
                
            self.__Oname_list = None
            self.__data_1 = None
        
        if self.__Skip:
            self.__Sname_list = self.__Data_2.columns[self.__list1]
            self.__data_skip = self.__data[self.__Sname_list].copy()
            self.__data_o2 = self.__data[self.__Sname_list].copy()
            self.__data = self.__data.drop(self.__Sname_list, axis=1)
            if self.__data.shape[1] != 0:
                self.__st = True
            else:
                self.__st = False
            
            self.__Sname_list = None
        else:
            self.__st = True
            
        if self.__st:
            self.__Dname_list = self.__data.columns
            self.__data_original_2 = self.__data.copy()
            self.__data_standard_2 = self.__Key.transform(self.__data)
            self.__data_standard_2 = pd.DataFrame(self.__data_standard_2)
            self.__data_standard_2.columns = self.__Dname_list
            self.__Dname_list = None
        else:
            self.__data_standard_2 = self.__data.copy()
            self.__data_original_2 = self.__data.copy()
        
        if self.__Skip:
            self.__data_standard_2 = pd.concat([self.__data_standard_2, self.__data_skip], axis=1)
            self.__data_original_2 = pd.concat([self.__data_original_2, self.__data_o2], axis=1)
            self.__data_skip = None
            self.__data_o2 = None
        
        if self.__OneHot:
            self.__data_standard_2 = pd.concat([self.__data_standard_2, self.__data_onehot], axis=1)
            self.__data_original_2 = pd.concat([self.__data_original_2, self.__data_onehot], axis=1)
            self.__data_onehot = None
        
        self.__data = None
        self.__fitstate_2 = True
        
        return None
        
    
    def standard_data_1(self):
        """
        standard_data_1 用于返回fit_transform()方法标准化后的数据，需先调用fit_transform()方法
        """
        if self.__fitstate != True:
            raise RuntimeError("没有数据被标准化，请先调用 fit_transform() 方法")
        
        return self.__data_standard
    
    def standard_data_2(self):
        """
        standard_data_2 用于返回transform()方法标准化后的数据，需先调用transform()方法
        """
        if self.__fitstate_2 != True:
            raise RuntimeError("没有数据被标准化，请先调用 transform() 方法")
        
        return self.__data_standard_2
    
    def original_data_1(self):
        """
        用于返回fit_transform()方法标准化后的原始数据，需先调用fit_transform()方法
        """
        if self.__fitstate != True:
            raise RuntimeError("请先调用 fit_transform() 方法")
        
        return self.__data_original
    
    def original_data_2(self):
        """
        用于返回transform方法标准化后的原始数据，需先调用transform方法
        """
        if self.__fitstate_2 != True:
            raise RuntimeError("请先调用 transform() 方法")
            
        return self.__data_original_2

"""========================================================="""

"""==================== ConfusionMatrix ===================="""
class ConfusionMatrix:
    """
    ConfusionMatrix 7.2 用于2分类混淆矩阵相关指标的计算和图表输出，并可与PlotROC、PlotPRC、PlotCC配合使用；置信水平为95%
    
    参数介绍
    ========
    
    model：训练好的模型，模型预测接口为“model.predict”或“model.predict_proba”，默认为None
    
    x_data：不带标签的数据集，类型为DataFrame，标准化状态与模型训练时保持一致，默认为None
    
    y_data：数据集的标签，类型为DataFrame或Series，默认为None
    
    custom_threshold：自定义阈值,类型为float，默认为None，若不设置，将采用模型预测方法下的默认阈值；★当设置Optimal_Threshold参数时，该参数失效
    
    DNN_Model：model是否为基于Keras的神经网络模型，类型为bool，默认为False
    
    Optimal_Threshold：是否使用最佳阈值，类型为str，默认为False；
                       
                       ① 若 Optimal_Threshold = "ROC" 表示以ROC曲线为基准，以Youden指数寻找最佳阈值
                       ② 若 Optimal_Threshold = "PRC" 表示以PRC曲线为基准，以F1-score指数寻找最佳阈值
    
    HL_test：是否进行Hoslem-Lemeshow拟合优度检验，类型为int，默认为False；设置该参数就表示进行模型拟合优度检验；参数表示分组数
    
    AUPRC_mod：表示计算AUPRC的方法，类型为str，默认未“trapezoidal”，即梯形规则法
                ①“trapezoidal”：梯形规则法
                ②“interpolate”：非线性插值法
    
    ========
    
    方法介绍
    ========
    
    metrics()方法：用于展示模型的评价指标，如ROC曲线下面积、准确率，敏感度等
    
    cm_set()方法：用于设置混淆矩阵的相关参数，如标题、坐标轴标签、分类标签和颜色等
    
    plot()方法：用于绘制混淆矩阵
    
    tc_set()方法：用于设置阈值校准图的相关参数，如标题、坐标轴标签颜色等
    
    threshold_calibration()方法：用于绘制校准阈值图，使用最佳阈值时才能使用
    
    save()方法：用于保存生成的混淆矩阵图
    =========
    
    可访问对象介绍
    =========
    
    _threshold：混淆矩阵的阈值,保留3位小数
    =========
    """
    
    def __init__(self, model=None, x_data=None, y_data=None, custom_threshold=None, 
                 DNN_Model=False, Optimal_Threshold=False, HL_test=False, AUPRC_mod='trapezoidal'):
        
        self.__optimal = False
        self.__OROC = False
        self.__OPRC = False
        self.__OTstate = False
        
        # 参数检查
        if type(model) == type(None):
            raise ValueError("缺少model参数，请传入model参数")
        else:
            self.__model = model
            
        if type(x_data) == type(None):
            raise ValueError("缺少x_data参数，请传入x_data参数")
        else: 
            if type(x_data) != type(pd.DataFrame()):
                raise ValueError("x_data的类型必须为DataFrame，而传入的类型为%s" % type(x_data))
            else:
                self.__x = x_data
                
        if type(y_data) == type(None):
            raise ValueError("缺少y_data参数，请传入y_data参数")
        else: 
            if type(y_data) != type(pd.DataFrame()):
                if type(y_data) != type(pd.Series(1)):
                    raise ValueError("y_data的类型必须为DataFrame或Series，而传入的类型为%s" % type(y_data))
                else:
                    self.__y = y_data
            else:
                self.__y = y_data
                
        if Optimal_Threshold != False:
            if Optimal_Threshold == "ROC":
                self.__optimal = True
                self.__OROC = True
            else: 
                if Optimal_Threshold == "PRC":
                    self.__optimal = True
                    self.__OPRC = True
                else:
                    raise ValueError("Optimal_Threshold参数只能为：'ROC'、'PRC'或者False")
            #检查
            assert self.__OROC != self.__OPRC, "代码有Bug，请联系包装者检查"
        else:
            assert self.__OROC != True, "代码有Bug，请联系包装者检查"
            assert self.__OPRC != True, "代码有Bug，请联系包装者检查"
            assert self.__optimal != True, "代码有Bug，请联系包装者检查"
                
        if HL_test:
            if type(HL_test) != type(int(1)):
                raise ValueError("HL_test的类型必须为int，而传入的类型为%s" % type(HL_test))
            else:
                self.__g = HL_test
                self.__HL_switch = True
        else:
            self.__HL_switch = False
            
        if AUPRC_mod == 'trapezoidal':
            self.__modstate = 1
        elif AUPRC_mod == 'interpolate':
            self.__modstate = 2
        else:
            raise ValueError("AUPRC_mod参数错误，其值只能为“trapezoidal”或“interpolate”")
        
        # 参数计算
        if self.__optimal:
            self.__OT = True
            if self.__OROC:
                if DNN_Model:
                    self.__yscore = self.__model.predict(self.__x)
                    self.__FPR, self.__TPR, self.__thresholds = roc_curve(self.__y,self.__yscore)
                    self.__precision, self.__recall, self.__thresholds2 = precision_recall_curve(self.__y,self.__yscore)
                    self.__Y = self.__TPR - self.__FPR
                    self.__youdenindex = np.argmax(self.__Y)
                    self.__optimal_threshold = self.__thresholds[self.__youdenindex]
                    self.__point = [self.__FPR[self.__youdenindex], self.__TPR[self.__youdenindex]]
                    self.__ypred = self.__model.predict(self.__x)
                    self.__OTstate = True
                else:
                    self.__yscore = self.__model.predict_proba(self.__x)[:, 1]
                    self.__FPR, self.__TPR, self.__thresholds = roc_curve(self.__y,self.__yscore)
                    self.__precision, self.__recall, self.__thresholds2 = precision_recall_curve(self.__y,self.__yscore)
                    self.__Y = self.__TPR - self.__FPR
                    self.__youdenindex = np.argmax(self.__Y)
                    self.__optimal_threshold = self.__thresholds[self.__youdenindex]
                    self.__point = [self.__FPR[self.__youdenindex], self.__TPR[self.__youdenindex]]
                    self.__ypred = self.__model.predict_proba(self.__x)[:, 1]
                    self.__OTstate = True
            
            if self.__OPRC:
                if DNN_Model:
                    self.__yscore = self.__model.predict(self.__x)
                    self.__FPR, self.__TPR, self.__thresholds = roc_curve(self.__y,self.__yscore)
                    self.__precision, self.__recall, self.__thresholds2 = precision_recall_curve(self.__y,self.__yscore)
                    self.__Y = (2 * self.__precision * self.__recall) / (self.__precision + self.__recall)
                    self.__f1_index = np.argmax(self.__Y)
                    self.__optimal_threshold = self.__thresholds2[self.__f1_index]
                    self.__point = [self.__recall[self.__f1_index], self.__precision[self.__f1_index]]
                    self.__ypred = self.__model.predict(self.__x)
                    self.__OTstate = True
                else:
                    self.__yscore = self.__model.predict_proba(self.__x)[:, 1]
                    self.__FPR, self.__TPR, self.__thresholds = roc_curve(self.__y,self.__yscore)
                    self.__precision, self.__recall, self.__thresholds2 = precision_recall_curve(self.__y,self.__yscore)
                    self.__Y = self.__Y = (2 * self.__precision * self.__recall) / (self.__precision + self.__recall)
                    self.__f1_index = np.argmax(self.__Y)
                    self.__optimal_threshold = self.__thresholds2[self.__f1_index]
                    self.__point = [self.__recall[self.__f1_index], self.__precision[self.__f1_index]]
                    self.__ypred = self.__model.predict_proba(self.__x)[:, 1]
                    self.__OTstate = True
                
            self.__ypred[self.__ypred >= self.__optimal_threshold] = 1
            self.__ypred[self.__ypred < self.__optimal_threshold] = 0
            self.__roc = auc(self.__FPR,self.__TPR)
            if self.__modstate ==1:
                self.__prc = auc(self.__recall, self.__precision)
            elif self.__modstate ==2:
                self.__iprauc = AUC(curve='PR')
                self.__iprauc.update_state(self.__y,self.__yscore)
                self.__prc =  np.float64(self.__iprauc.interpolate_pr_auc().numpy())
            else:
                assert False, "代码有Bug，请联系包装者检查"
                
            # 检查
            assert self.__OTstate, "代码有Bug，请联系包装者检查"
        else:
            self.__OT = False
            if type(custom_threshold) == type(None):
                self.__custom_threshold = 0.5
                if DNN_Model:
                    self.__ypred = self.__model.predict(self.__x)
                    self.__ypred[self.__ypred >= 0.5] = 1
                    self.__ypred[self.__ypred < 0.5] = 0
                    self.__yscore = self.__model.predict(self.__x)
                    self.__FPR, self.__TPR, self.__thresholds = roc_curve(self.__y,self.__yscore)
                    self.__precision, self.__recall, self.__thresholds2 = precision_recall_curve(self.__y,self.__yscore)
                else:
                    self.__ypred = self.__model.predict(self.__x)
                    self.__yscore = self.__model.predict_proba(self.__x)[:, 1]
                    self.__FPR, self.__TPR, self.__thresholds = roc_curve(self.__y,self.__yscore)
                    self.__precision, self.__recall, self.__thresholds2 = precision_recall_curve(self.__y,self.__yscore)
                self.__roc = auc(self.__FPR,self.__TPR)
                if self.__modstate ==1:
                    self.__prc = auc(self.__recall, self.__precision)
                elif self.__modstate ==2:
                    self.__iprauc = AUC(curve='PR')
                    self.__iprauc.update_state(self.__y,self.__yscore)
                    self.__prc =  np.float64(self.__iprauc.interpolate_pr_auc().numpy())
                else:
                    assert False, "代码有Bug，请联系包装者检查"

            else:
                self.__custom_threshold = custom_threshold
                if DNN_Model:
                    self.__ypred = self.__model.predict(self.__x)
                    self.__yscore = self.__model.predict(self.__x)
                    self.__FPR, self.__TPR, self.__thresholds = roc_curve(self.__y,self.__yscore)
                    self.__precision, self.__recall, self.__thresholds2 = precision_recall_curve(self.__y,self.__yscore)
                else:
                    self.__ypred = self.__model.predict_proba(self.__x)[:, 1]
                    self.__yscore = self.__model.predict_proba(self.__x)[:, 1]
                    self.__FPR, self.__TPR, self.__thresholds = roc_curve(self.__y,self.__yscore)
                    self.__precision, self.__recall, self.__thresholds2 = precision_recall_curve(self.__y,self.__yscore)
                
                self.__ypred[self.__ypred >= self.__custom_threshold] = 1
                self.__ypred[self.__ypred < self.__custom_threshold] = 0
                self.__roc = auc(self.__FPR,self.__TPR)
                if self.__modstate ==1:
                    self.__prc = auc(self.__recall, self.__precision)
                elif self.__modstate ==2:
                    self.__iprauc = AUC(curve='PR')
                    self.__iprauc.update_state(self.__y,self.__yscore)
                    self.__prc =  np.float64(self.__iprauc.interpolate_pr_auc().numpy())
                else:
                    assert False, "代码有Bug，请联系包装者检查"
        
        self.__cm = confusion_matrix(self.__y,self.__ypred)
        
        if DNN_Model:
            self.__scorecopy = self.__model.predict(self.__x)
            self.__yscore = self.__scorecopy.flatten()
            # 检查
            for self.__checkpoint in range(0, len(self.__yscore)):
                assert self.__yscore[self.__checkpoint] == self.__scorecopy[self.__checkpoint], "代码有bug，请联系包装者检查"
                    
        # DeLong 计算AUROC置信区间
        self.__alpha = 0.95
        self.__y_true = self.__y.to_numpy()
        assert np.array_equal(np.unique(self.__y_true), [0, 1])
        self.__order = (-self.__y_true).argsort()
        self.__label_1_count = int(self.__y_true.sum())
        self.__ordered_sample_weight = None
        self.__predictions_sorted_transposed = self.__yscore[np.newaxis, self.__order]
        
        self.__m = self.__label_1_count
        self.__n = self.__predictions_sorted_transposed.shape[1] - self.__m
        self.__positive_examples = self.__predictions_sorted_transposed[:, :self.__m]
        self.__negative_examples = self.__predictions_sorted_transposed[:, self.__m:]
        self.__k = self.__predictions_sorted_transposed.shape[0]
        
        self.__tx = np.empty([self.__k, self.__m], dtype=float)
        self.__ty = np.empty([self.__k, self.__n], dtype=float)
        self.__tz = np.empty([self.__k, self.__m + self.__n], dtype=float)
        
        for self.__r in range(self.__k):
            self.__tx[self.__r, :] = self.compute_midrank(self.__positive_examples[self.__r, :])
            self.__ty[self.__r, :] = self.compute_midrank(self.__negative_examples[self.__r, :])
            self.__tz[self.__r, :] = self.compute_midrank(self.__predictions_sorted_transposed[self.__r, :])
        
        self.__delong_aucs = self.__tz[:, :self.__m].sum(axis=1) / self.__m / self.__n - float(self.__m + 1.0) / 2.0 / self.__n
        self.__v01 = (self.__tz[:, :self.__m] - self.__tx[:, :]) / self.__n
        self.__v10 = 1.0 - (self.__tz[:, self.__m:] - self.__ty[:, :]) / self.__m
        
        self.__sx = np.cov(self.__v01)
        self.__sy = np.cov(self.__v10)
        
        self.__auroc_cov = self.__sx / self.__m + self.__sy / self.__n
        assert len(self.__delong_aucs) == 1, "代码有Bug，请告知包装者"
        self.__delong_auc = self.__delong_aucs[0]
        
        self.__auc_std = np.sqrt(self.__auroc_cov)
        self.__lower_upper_q = np.abs(np.array([0, 1])-(1-self.__alpha)/2)
        
        self.__roc_ci = stats.norm.ppf(self.__lower_upper_q, loc=self.__delong_auc, scale=self.__auc_std)
        self.__roc_ci[self.__roc_ci > 1] = 1
        
        # Logit Method 计算AUPRC置信区间
        self.__postivesample = self.__y.sum()
        self.__exponent1 = math.log(self.__prc/(1-self.__prc), math.e) - 1.96*math.pow(self.__postivesample*self.__prc*(1-self.__prc), -1/2)
        self.__exponent2 = math.log(self.__prc/(1-self.__prc), math.e) + 1.96*math.pow(self.__postivesample*self.__prc*(1-self.__prc), -1/2)
        self.__prc_ci = [math.exp(self.__exponent1) / (1 + math.exp(self.__exponent1)), 
                         math.exp(self.__exponent2) / (1 + math.exp(self.__exponent2))]
        
        # 混淆矩阵参数
        self.__classtag = ["Negative", "Postive"]
        self.__title = "Confusion Matrix"
        self.__xlabel = "Predict Label"
        self.__ylabel = "True Label"
        self.__fcolor = 20
        self.__reverse = True
        
        # 阈值校准图参数
        self.__title2 = "Threshold Calibration of Model"
        
        if self.__OROC:
            self.__xlabel2 = "False Postive Rate (FPR)"
            self.__ylabel2 = "True Postive Rate (TPR)"
        
        if self.__OPRC:
            self.__xlabel2 = "Recall (True Postive Rate)"
            self.__ylabel2 = "Precision"
        
        self.__linecolor = "#83b7e0"
        self.__pointcolor = "#ff8383"
        self.__modelname = "Model"
        
        
        if self.__OT:
            self._threshold = round(self.__optimal_threshold, 3)
        else:
            self._threshold = round(self.__custom_threshold, 3)
                
        
        self.__plotstate = False
        self.__tcstate = False
        
        return None
    
    def compute_midrank(self, x):
        """
        compute_midrank()方法，用于DeLong置信区间中间值计算，仅供内部调用，外部调用无意义
        
        """
        
        self.__J = np.argsort(x)
        self.__Z = x[self.__J]
        self.__N = len(x)
        self.__T = np.zeros(self.__N, dtype=float)
        self.__i = 0
        while self.__i < self.__N:
            self.__j = self.__i
            while self.__j < self.__N and self.__Z[self.__j] == self.__Z[self.__i]:
                self.__j += 1
            self.__T[self.__i:self.__j] = 0.5*(self.__i + self.__j - 1)
            self.__i = self.__j
        self.__T2 = np.empty(self.__N, dtype=float)
        self.__T2[self.__J] = self.__T + 1
        
        return self.__T2
    
    def metrics(self):
        """metrics()方法用于展示该模型的评价指标"""
        
        if self.__OT:
            print('The Optimal Threshold：%.3f' % self.__optimal_threshold)
            print('')
        else:
            print('The Threshold：%.3f' % self.__custom_threshold)
            print('')
            
        # Brier Score
        self.__brier = brier_score_loss(self.__y, self.__yscore, pos_label=self.__y.max())
        
        # Hoslem-Lemeshow Test
        if self.__HL_switch:
            self.__hl1 = pd.DataFrame(self.__y)
            self.__hl2 = pd.DataFrame(self.__yscore.reshape(-1,1))
            self.__data_hl = pd.concat([self.__hl2, self.__hl1], axis =1)
            self.__data_hl.columns = ['prob', 'label']
            self.__data_hl = self.__data_hl.sort_values('prob')
            self.__data_hl['decile'] = pd.qcut(self.__data_hl['prob'], self.__g, duplicates='drop')
            self.__ys = self.__data_hl['label'].groupby(self.__data_hl.decile).sum()
            self.__yt = self.__data_hl['label'].groupby(self.__data_hl.decile).count()
            self.__yn = self.__yt - self.__ys
            self.__yps = self.__data_hl['prob'].groupby(self.__data_hl.decile).sum()
            self.__ypt = self.__data_hl['prob'].groupby(self.__data_hl.decile).count()
            self.__ypn = self.__ypt - self.__yps
            self.__hltest = ( ((self.__ys - self.__yps)**2 / self.__yps) + ((self.__yn - self.__ypn)**2 / self.__ypn) ).sum()
            self.__pval = 1 - chi2.cdf(self.__hltest, self.__g-2)
            self.__df = self.__g - 2        
        
        print('AUROC：%.3f(%.3f-%.3f)'% (self.__roc, self.__roc_ci[0], self.__roc_ci[1]))
        print('---------')
        print('NOTE: The Confidence Interval for AUROC was Calculated by Delong Method (Delong et al, 1988)')
        print('---------')
        print('')
        print('AUPRC：%.3f(%.3f-%.3f)'% (self.__prc, self.__prc_ci[0], self.__prc_ci[1]))
        if self.__modstate == 1:
            print("Method: Trapezoidal Rule")
        elif self.__modstate == 2:
            print("Method: Interpolation (Davis & Goadrich 2006)")
        else:
            assert False, "Error!"
        print('---------')
        print('NOTE: The Confidence Interval for AUPRC was Calculated by Logit Method (Boyd et al, 2013)')
        print('---------')
        print('')
        print('Brier Score：%.3f' % (self.__brier))
        print('')
        if self.__HL_switch:
            print('Hoslem-Lemeshow Test')
            print('HL-chi2：{}'.format(self.__hltest))
            print('df：%.4f' % (self.__df))
            print('P-value：%.4f' % (self.__pval))
        print('')
        print('Sensitivity：%.3f'%(self.__cm[1,1]/(self.__cm[1,1]+self.__cm[1,0])))
        print('Specifity：%.3f'%(self.__cm[0,0]/(self.__cm[0,0]+self.__cm[0,1])))
        self.__sen = self.__cm[1,1]/(self.__cm[1,1]+self.__cm[1,0])
        self.__spe = self.__cm[0,0]/(self.__cm[0,0]+self.__cm[0,1])
        print('+LR：%.3f' % (self.__sen/(1-self.__spe)))
        print('-LR：%.3f' % ((1-self.__sen)/self.__spe))
        print('PPV：%.3f'%(self.__cm[1,1]/(self.__cm[1,1]+self.__cm[0,1])))
        print('NPV：%.3f'%(self.__cm[0,0]/(self.__cm[0,0]+self.__cm[1,0])))
        print('Accuracy：%.3f'%accuracy_score(self.__y,self.__ypred)) 
        print('Precision：%.3f'%precision_score(self.__y,self.__ypred))
        print('Recall：%.3f'%recall_score(self.__y,self.__ypred))
        print('F1-Value：%.3f'%f1_score(self.__y,self.__ypred))
    
    def cm_set(self, class_tag=None, title=None, x_label=None, y_label=None, reverse=None, Font_color_threshold=None):
        """
        cm_set()方法用于设置混淆矩阵的相关参数
        
        参数介绍
        ========
        
        class_tag：分类标签，类型为元素个数为2的字符串列表，默认为None，即使用默认设置；格式：class_tag=["Negative", "Postive"]
        
        title：混淆矩阵的标题，类型为str，默认为None，即“Confusion Matrix”
        
        x_label：x轴的标签，类型为str，默认为None，即“Predict Label”
        
        y_label：y轴的标签，类型为str，默认为None，即“True Label”
        
        reverse：分类标签的排序，类型为bool，默认为None，即“True，倒序”，False为正序
        
        Font_color_threshold：当某类的样本的数量大于多少时，字体颜色变为白色，类型为int，默认为None，即20
        
        ★关于内置的色彩映射函数，请参考matplotlib官网，并在此类源代码的“cmap=plt.cm.Blues”处修改
        ========
        """
        
        if type(class_tag) != type(None):
            self.__classtag = class_tag
        if type(title) != type(None):
            self.__title = title
        if type(x_label) != type(None):
            self.__xlabel = x_label
        if type(y_label) != type(None):
            self.__ylabel = y_label
        if type(Font_color_threshold) != type(None):
            self.__fcolor = Font_color_threshold
        if type(reverse) != type(None):
            self.__reverse = reverse
            
        return None
    
    def plot(self):
        """plot()方法用于混淆矩阵的绘制"""        
        
        self.__guess = self.__classtag
        self.__fact = self.__classtag
        self.__classes = list(set(self.__fact))
        self.__classes.sort(reverse=self.__reverse)
        self.__r=[[self.__cm[0,0],self.__cm[1,0]],[self.__cm[0,1],self.__cm[1,1]]]
        
        plt.close()
        plt.figure(figsize=(12,10))
        self.__confusion =self.__r
        plt.imshow(self.__confusion, cmap=plt.cm.Blues)
        self.__indices = range(len(self.__confusion))
        self.__indices2 = range(3)
        plt.xticks(self.__indices, self.__classes,rotation=40, fontsize=18)
        plt.yticks([0.00,1.00], self.__classes, fontsize=18)
        plt.ylim(1.5, -0.5)
        
        plt.title(self.__title,fontdict={'weight':'normal','size': 18})
        self.__cb=plt.colorbar()
        self.__cb.ax.tick_params(labelsize=18)
        plt.xlabel(self.__xlabel,fontsize=18)
        plt.ylabel(self.__ylabel,fontsize=18)
        
        for self.__n1 in range(len(self.__confusion)):
            for self.__n2 in range(len(self.__confusion[self.__n1])):
                if self.__confusion[self.__n1][self.__n2] > self.__fcolor:
                    self.__focolor="w"
                else:
                    self.__focolor="black"
                plt.text(self.__n1, self.__n2, self.__confusion[self.__n1][self.__n2], fontsize=18, 
                         color =self.__focolor, verticalalignment='center', horizontalalignment='center')
        
        self.__confusion_matrix = plt.gcf()
        
        if self.__OT:
            print('This confusion matrix is based on the threshold of %.3f' % self.__optimal_threshold)
            print('')
        else:
            print('This confusion matrix is based on the threshold of %.3f' % self.__custom_threshold)
            print('')
        
        plt.show()
        self.__plotstate = True
        
        return None
    
    def tc_set(self, title=None, x_label=None, y_label=None, model_name=None, roc_color=None, point_color=None):
        """
        tc_set()方法用于设置阈值校准图的相关参数
        
        参数介绍
        ========
        title：校准图的标题，类型为str，默认为None，即“Threshold Calibration of Model”
        
        x_label：x轴的标签，类型为str，默认为None，即默认标签
        
        y_label：y轴的标签，类型为str，默认为None，即默认标签
        
        model_name：模型的名字，类型为str，默认为None，即“Model”
        
        roc_color：ROC曲线的颜色，类型为str，默认为None，即“#83b7e0”
        
        point_color：校准阈值点的颜色，类型为str，默认为None，即“#ff8383”
        ========
        """
        if type(title) != type(None):
            self.__title2 = title
        if type(x_label) != type(None):
            self.__xlabel2 = x_label
        if type(y_label) != type(None):
            self.__ylabel2 = y_label
        if type(model_name) != type(None):
            self.__modelname = model_name
        if type(roc_color) != type(None):
            self.__linecolor = roc_color
        if type(point_color) != type(None):
            self.__pointcolor = point_color
        
        return None
    
    def threshold_calibration(self):
        """threshold_calibration()方法用于阈值校准图绘制"""
        
        if self.__OT:
            plt.close()
            plt.figure(figsize=(15,15), dpi=300, facecolor='w')
            plt.subplot(224)
            
            if self.__OROC:
                plt.plot([0, 1], [0, 1], color="#d0c9cb", lw=1.5, linestyle='--')
                plt.plot(self.__FPR, self.__TPR, color=self.__linecolor, lw=2, 
                         label=self.__modelname + (' ( AUROC = %0.3f )' % self.__roc))
                plt.plot([0, self.__point[0]], [self.__point[1], self.__point[1]], color="#142158", lw=1, linestyle=':')
                plt.plot([self.__point[0], self.__point[0]], [0, self.__point[1]], color="#142158", lw=1, linestyle=':')
                plt.plot(self.__point[0], self.__point[1], marker='o', color=self.__pointcolor)
                plt.text(self.__point[0]+0.05, self.__point[1]-0.05, f'The optimal threshold: {self.__optimal_threshold:.3f}')
                plt.text(self.__point[0]+0.05, self.__point[1]-0.1, f'FPR: {self.__FPR[self.__youdenindex]:.3f}')
                plt.text(self.__point[0]+0.05, self.__point[1]-0.15, f'TPR (Recall): {self.__TPR[self.__youdenindex]:.3f}')
            
            if self.__OPRC:
                plt.plot(self.__recall, self.__precision, color=self.__linecolor, lw=2, 
                         label=self.__modelname + (' ( AUPRC = %0.3f )' % self.__prc))
                plt.plot([0, self.__point[0]], [self.__point[1], self.__point[1]], color="#142158", lw=1, linestyle=':')
                plt.plot([self.__point[0], self.__point[0]], [0, self.__point[1]], color="#142158", lw=1, linestyle=':')
                plt.plot(self.__point[0], self.__point[1], marker='o', color=self.__pointcolor)
                plt.text(self.__point[0]-0.4, self.__point[1]-0.05, f'The optimal threshold: {self.__optimal_threshold:.3f}')
                plt.text(self.__point[0]-0.4, self.__point[1]-0.1, f'Recall: {self.__recall[self.__f1_index]:.3f}')
                plt.text(self.__point[0]-0.4, self.__point[1]-0.15, f'Precision: {self.__precision[self.__f1_index]:.3f}')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel(self.__xlabel2)
            plt.ylabel(self.__ylabel2)
            plt.title(self.__title2)
            plt.legend(loc="best")
            self.__TC = plt.gcf()
            plt.show()
            
            self.__tcstate = True
        else:
            raise ValueError("未对模型进行最佳阈值校准操作")
        
        return None
        
        
    
    def save(self, dpi=300, CM=False, TC=False, path_cm="confusion matrix.png", path_tc="threshold calibration.png"):
        """
        save()方法用于保存生成的混淆矩阵图
        
        参数介绍
        ========
        
        dpi：类型为int，默认为300
        
        CM：是否保存当前与之下的混淆矩阵，需要先调用plot()方法，类型未bool，默认为False
        
        TC：是否保存阈值校准图，需要先调用threshold_calibration()方法，类型未bool，默认为False
        
        path_cm：混淆矩阵保存的路径和名字，格式为，“/file1/file2/image_name.png”；默认保存在当前路径，名字为“confusion matrix.png”
        
        path_tc：阈值校准图的保存路径和名字，格式为，“/file1/file2/image_name.png”；默认保存在当前路径，名字为“threshold calibration.png”
        
        ========
        """
        if CM:
            if self.__plotstate:
                self.__pathcm = path_cm
                self.__dpi = dpi
                self.__confusion_matrix.savefig(self.__pathcm, dpi=self.__dpi, bbox_inches = 'tight')
                print("混淆矩阵已保存，路径为：%s" % self.__pathcm)
            else:
                raise RuntimeError("尚未绘制混淆矩阵图，请先调用plot()方法绘制")
        
        if TC:
            if self.__tcstate:
                self.__pathtc = path_tc
                self.__dpi = dpi
                self.__TC.savefig(self.__pathtc, dpi=self.__dpi, bbox_inches = 'tight')
                print("阈值调整图已保存，路径为：%s" % self.__pathtc)
            else:
                raise RuntimeError("尚未绘制阈值校准图，请先调用threshold_calibration()方法绘制")
            
        return None
    
    def roc(self):
        """★此方法为PlotROC的接口，仅在ConfusionMatrix中调用无意义★"""
        
        return self.__FPR, self.__TPR, self.__roc
    
    def prc(self):
        """★此方法为PlotPRC的接口，仅在ConfusionMatrix中调用无意义★"""
        
        return self.__precision, self.__recall, self.__prc
    
    def delong_test(self):
        """★此方法为DelongTest的接口，仅在ConfusionMatrix中调用无意义★"""
        
        return self.__y, self.__yscore
    
    def cc(self):
        """★此方法为PlotCC的接口，仅在ConfusionMatrix中调用无意义★"""
        
        return self.__y, self.__yscore
    
    def bootstrap_prc(self):
        """★此方法为Bootstrap_PRC的接口，仅在ConfusionMatrix中调用无意义★"""
        
        return self.__y, self.__yscore, self.__prc, self.__modstate, self.__prc_ci
        
    def dca(self):
        """★此方法为PlotDCA的接口，仅在ConfusionMatrix中调用无意义★"""
        
        return self.__yscore
        
"""========================================================="""

"""======================== PlotROC ========================"""
class PlotROC:
    """
    PlotROC用于ROC曲线的绘制，需要配合ConfusionMatrix使用
    
    参数介绍
    ========
    CM_Name_List：是待作ROC曲线模型的ConfusionMatrix和模型命名字符串的组合列表,类型为list，默认为None，格式为↓
                  
                  CM_Name_List = [(cm_lr, 'LR'),(cm_dnn,'DNN')]
                  
    ========
    
    方法介绍
    ========
    
    roc_set()方法：用于设置ROC曲线的相关属性
    
    plot()方法：用于绘制ROC曲线

    save()方法：用于保存绘制的曲线
    
    ========
    
    """
    def __init__(self, CM_Name_List=None):
        
        #参数检查
        if type(CM_Name_List) == type(None):
            raise ValueError("缺少CM_Name_List参数")
        elif type(CM_Name_List) != type([]):
            raise ValueError("CM_Name_List参数的类型应该为list，而传入的类型为%s" % type(CM_Name_List))
        else:
            self.__list = CM_Name_List
            
        #ROC曲线相关参数预设
        self.__title = "ROC Curve"
        self.__xlabel = "False Positive Rate"
        self.__ylabel = "True Positive Rate"
        self.__colorlist = ['#ff6d6d', '#79a4ea', '#14f474', '#f5ff63', '#b651f0', '#e4b628', '#1de497', '#e41d6c']
        self.__colorswitch = False
        self.__systemcolor = False
        
        self.__plotstate = False
        
        return None
    
    def roc_set(self, title=None, xlabel=None, ylabel=None, colorlist=None):
        """
        roc_set()方法用于设置ROC曲线的相关属性
        
        参数介绍
        ========
        title：曲线的标题，类型为str，默认为None，即“ROC Curve”
        
        xlabel：X轴标题，类型为str，默认为None，即“False Positive Rate”
        
        ylabel：Y轴标题，类型为str，默认为None，即“True Positive Rate”
        
        colorlist：ROC曲线的颜色列表，类型为list，颜色的数目需大于等于曲线的数目，默认为None，即采用默认配色
        ========
        """
        if type(title) != type(None):
            if type(title) == type("String"):
                self.__title = title
            else:
                raise ValueError("title的类型应为str，而传入的类型为%s" % type(title))
        
        if type(xlabel) != type(None):
            if type(xlabel) == type("String"):
                self.__xlabel = xlabel
            else:
                raise ValueError("xlabel的类型应为str，而传入的类型为%s" % type(xlabel))
        
        if type(ylabel) != type(None):
            if type(ylabel) == type("String"):
                self.__ylabel = ylabel
            else:
                raise ValueError("ylabel的类型应为str，而传入的类型为%s" % type(ylabel))
                
        if type(colorlist) != type(None):
            if type(colorlist) == type([]):
                if len(self.__list) > len(colorlist):
                    raise ValueError("待绘制的ROC曲线数目为%s，colorlist的数目也至少为%s，而传入颜色数为%s" % (len(self.__list), len(self.__list), len(colorlist)))
                else:
                    self.__colorswitch = True
                    self.__colorlist = colorlist
            else:
                raise ValueError("colorlist的类型应为list，而传入的类型为%s" % type(colorlist))
        else:
            if len(self.__list) > 8:
                self.__systemcolor = True
        
        return None
    
    def plot(self):
        """plot()方法用于绘制ROC曲线"""
        
        plt.close()
        plt.figure(figsize=(15,15), dpi=300, facecolor='w')
        plt.subplot(224)
        
        self.__colorindex = 0
        
        plt.plot([0, 1], [0, 1], color='#d0c9cb', lw=1.25, linestyle='--')
        for self.__cm, self.__name in self.__list:
            self.__fpr, self.__tpr, self.__roc = self.__cm.roc()
            if self.__colorswitch:
                plt.plot(self.__fpr, self.__tpr, color=self.__colorlist[self.__colorindex], lw=1.5, 
                         alpha = 0.7, label=self.__name + (' ( AUROC = %0.3f )' % self.__roc))
            else:
                if self.__systemcolor:
                    plt.plot(self.__fpr, self.__tpr, lw=1.5, alpha = 0.7, 
                             label=self.__name + (' ( AUROC = %0.3f )' % self.__roc))                
                else:
                    plt.plot(self.__fpr, self.__tpr, color=self.__colorlist[self.__colorindex], lw=1.5, 
                             alpha = 0.7, label=self.__name + (' ( AUROC = %0.3f )' % self.__roc))
            self.__colorindex = self.__colorindex + 1
        
        self.__colorindex = None
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(self.__xlabel)
        plt.ylabel(self.__ylabel)
        plt.title(self.__title)
        plt.legend(loc="best")
        self.__ROCPlot = plt.gcf()
        plt.show()
        
        self.__plotstate = True
        
        return None
    
    def save(self, path="ROC.png", dpi=300):
        """
        save()方法用于保存绘制的ROC曲线
        
        参数介绍
        ========
        
        path：保存路径，类型为str，格式为“/file1/file2/image_name.png”；默认保存在当前路径，名字为"ROC.png"
        
        dpi：每英寸上，所能印刷的网点数，类型为int，默认为300
        
        ========
        """
        
        if self.__plotstate:
            self.__path = path
            self.__dpi = dpi
            
            self.__ROCPlot.savefig(self.__path, dpi=self.__dpi, bbox_inches = 'tight')
            print("ROC曲线已保存，路径为：%s" % self.__path)
        else:
            raise RuntimeError("尚未绘制ROC曲线，请先调用plot()方法")
            
        return None
        
"""========================================================="""

"""======================== PlotPRC ========================"""
class PlotPRC:
    """
    PlotPRC用于PRC曲线的绘制，需要配合ConfusionMatrix使用
    
    参数介绍
    ========
    CM_Name_List：是待作PRC曲线模型的ConfusionMatrix和模型命名字符串的组合列表,类型为list，默认为None，格式为↓
                  
                  CM_Name_List = [(cm_lr, 'LR'),(cm_dnn,'DNN')]
                  
    ========
    
    方法介绍
    ========
    
    prc_set()方法：用于设置ROC曲线的相关属性
    
    plot()方法：用于绘制ROC曲线

    save()方法：用于保存绘制的曲线
    
    ========
    
    """
    def __init__(self, CM_Name_List=None):
        
        #参数检查
        if type(CM_Name_List) == type(None):
            raise ValueError("缺少CM_Name_List参数")
        elif type(CM_Name_List) != type([]):
            raise ValueError("CM_Name_List参数的类型应该为list，而传入的类型为%s" % type(CM_Name_List))
        else:
            self.__list = CM_Name_List
            
        #PRC曲线相关参数预设
        self.__title = "Precision-Recall Curve"
        self.__xlabel = "Recall"
        self.__ylabel = "Precision"
        self.__colorlist = ['#ff6d6d', '#79a4ea', '#14f474', '#f5ff63', '#b651f0', '#e4b628', '#1de497', '#e41d6c']
        self.__colorswitch = False
        self.__systemcolor = False
        
        self.__plotstate = False
        
        return None
    
    def prc_set(self, title=None, xlabel=None, ylabel=None, colorlist=None):
        """
        prc_set()方法用于设置PRC曲线的相关属性
        
        参数介绍
        ========
        title：曲线的标题，类型为str，默认为None，即“Precision-Recall Curve”
        
        xlabel：X轴标题，类型为str，默认为None，即“Recall”
        
        ylabel：Y轴标题，类型为str，默认为None，即“Precision”
        
        colorlist：PRC曲线的颜色列表，类型为list，颜色的数目需大于等于曲线的数目，默认为None，即采用默认配色
        ========
        """
        if type(title) != type(None):
            if type(title) == type("String"):
                self.__title = title
            else:
                raise ValueError("title的类型应为str，而传入的类型为%s" % type(title))
        
        if type(xlabel) != type(None):
            if type(xlabel) == type("String"):
                self.__xlabel = xlabel
            else:
                raise ValueError("xlabel的类型应为str，而传入的类型为%s" % type(xlabel))
        
        if type(ylabel) != type(None):
            if type(ylabel) == type("String"):
                self.__ylabel = ylabel
            else:
                raise ValueError("ylabel的类型应为str，而传入的类型为%s" % type(ylabel))
                
        if type(colorlist) != type(None):
            if type(colorlist) == type([]):
                if len(self.__list) > len(colorlist):
                    raise ValueError("待绘制的PRC曲线数目为%s，colorlist的数目也至少为%s，而传入颜色数为%s" % (len(self.__list), len(self.__list), len(colorlist)))
                else:
                    self.__colorswitch = True
                    self.__colorlist = colorlist
            else:
                raise ValueError("colorlist的类型应为list，而传入的类型为%s" % type(colorlist))
        else:
            if len(self.__list) > 8:
                self.__systemcolor = True
        
        return None
    
    def plot(self):
        """plot()方法用于绘制PRC曲线"""
        
        plt.close()
        plt.figure(figsize=(15,15), dpi=300, facecolor='w')
        plt.subplot(224)
        
        self.__colorindex = 0
        
        for self.__cm, self.__name in self.__list:
            self.__precision, self.__recall, self.__prc = self.__cm.prc()
            if self.__colorswitch:
                plt.plot(self.__recall, self.__precision, color=self.__colorlist[self.__colorindex], lw=1.5, 
                         alpha = 0.7, label=self.__name + (' ( AUPRC = %0.3f )' % self.__prc))
            else:
                if self.__systemcolor:
                    plt.plot(self.__recall, self.__precision, lw=1.5, alpha = 0.7, 
                             label=self.__name + (' ( AUPRC = %0.3f )' % self.__prc))                
                else:
                    plt.plot(self.__recall, self.__precision, color=self.__colorlist[self.__colorindex], lw=1.5, 
                             alpha = 0.7, label=self.__name + (' ( AUPRC = %0.3f )' % self.__prc))
            self.__colorindex = self.__colorindex + 1
        
        self.__colorindex = None
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel(self.__xlabel)
        plt.ylabel(self.__ylabel)
        plt.title(self.__title)
        plt.legend(loc="best")
        self.__PRCPlot = plt.gcf()
        plt.show()
        
        self.__plotstate = True
        
        return None
    
    def save(self, path="PRC.png", dpi=300):
        """
        save()方法用于保存绘制的PRC曲线
        
        参数介绍
        ========
        
        path：保存路径，类型为str，格式为“/file1/file2/image_name.png”；默认保存在当前路径，名字为"PRC.png"
        
        dpi：每英寸上，所能印刷的网点数，类型为int，默认为300
        
        ========
        """
        
        if self.__plotstate:
            self.__path = path
            self.__dpi = dpi
            
            self.__PRCPlot.savefig(self.__path, dpi=self.__dpi, bbox_inches = 'tight')
            print("PRC曲线已保存，路径为：%s" % self.__path)
        else:
            raise RuntimeError("尚未绘制PRC曲线，请先调用plot()方法")
            
        return None
        
"""========================================================="""

"""====================== DelongTest ======================="""
class DelongTest:
    """
    DelongTest用于AUROC显著性检验，需要配合ConfusionMatrix使用（4.0及其以上）
    
    参数介绍
    ========
    CM_Name_List：是待检验模型的ConfusionMatrix和模型命名字符串的组合列表,类型为list，默认为None，格式为↓
                  
                  CM_Name_List = [(cm_lr, 'LR'),(cm_svm, 'SVM'), (cm_dnn,'DNN')]
                  
    ========
    
    方法介绍
    ========
    
    test()方法：执行Delong检验
    
    ========
    
    """
    def __init__(self, CM_Name_List=None):
        
        # 参数检查
        if type(CM_Name_List) == type(None):
            raise ValueError("缺少CM_Name_List参数")
        elif type(CM_Name_List) != type([]):
            raise ValueError("CM_Name_List参数的类型应该为list，而传入的类型为%s" % type(CM_Name_List))
        else:
            self.__list = CM_Name_List
            
        # 参数转换
        self.__testlist = []
        for self.__cm, self.__name in self.__list:
            self.__y_true, self.__y_pred = self.__cm.delong_test()
            self.__newtuple = (self.__y_pred, self.__name)
            self.__testlist.append(self.__newtuple)
        
        self.__y_pred = None
        self.__name = None
        
        return None
    
    
    def compute_midrank(self, x):
        self.__J = np.argsort(x)
        self.__Z = x[self.__J]
        self.__N = len(x)
        self.__T = np.zeros(self.__N, dtype=float)
        self.__i = 0
        while self.__i < self.__N:
            self.__j = self.__i
            while self.__j < self.__N and self.__Z[self.__j] == self.__Z[self.__i]:
                self.__j += 1
            self.__T[self.__i:self.__j] = 0.5*(self.__i + self.__j - 1)
            self.__i = self.__j
        self.__T2 = np.empty(self.__N, dtype=float)
        self.__T2[self.__J] = self.__T + 1
        
        return self.__T2
    
    
    def compute_midrank_weight(self, x, sample_weight):
        self.__J = np.argsort(x)
        self.__Z = x[self.__J]
        self.__cumulative_weight = np.cumsum(sample_weight[self.__J])
        self.__N = len(x)
        self.__T = np.zeros(self.__N, dtype=float)
        self.__i = 0
        while self.__i < self.__N:
            self.__j = self.__i
            while self.__j < self.__N and self.__Z[self.__j] == self.__Z[self.__i]:
                self.__j += 1
            self.__T[self.__i:self.__j] = self.__cumulative_weight[self.__i:self.__j].mean()
            self.__i = self.__j
        self.__T2 = np.empty(self.__N, dtype=float)
        self.__T2[self.__J] = self.__T
    
        return self.__T2
    
    
    def fastDeLong_weights(self, predictions_sorted_transposed, label_1_count, sample_weight):
        
        # Short variables are named as they are in the paper
    
        self.__m = label_1_count

        self.__n = predictions_sorted_transposed.shape[1] - self.__m

        self.__positive_examples = predictions_sorted_transposed[:, :self.__m]

        self.__negative_examples = predictions_sorted_transposed[:, self.__m:]

        self.__k = predictions_sorted_transposed.shape[0]


        self.__tx = np.empty([self.__k, self.__m], dtype=float)

        self.__ty = np.empty([self.__k, self.__n], dtype=float)

        self.__tz = np.empty([self.__k, self.__m + self.__n], dtype=float)

        for self.__r in range(self.__k):

            self.__tx[self.__r, :] = self.compute_midrank_weight(self.__positive_examples[self.__r, :], sample_weight[:self.__m])

            self.__ty[self.__r, :] = self.compute_midrank_weight(self.__negative_examples[self.__r, :], sample_weight[self.__m:])

            self.__tz[self.__r, :] = self.compute_midrank_weight(predictions_sorted_transposed[self.__r, :], sample_weight)

        self.__total_positive_weights = sample_weight[:self.__m].sum()

        self.__total_negative_weights = sample_weight[self.__m:].sum()

        self.__pair_weights = np.dot(sample_weight[:self.__m, np.newaxis], sample_weight[np.newaxis, self.__m:])

        self.__total_pair_weights = self.__pair_weights.sum()

        self.__aucs = (sample_weight[:self.__m]*(self.__tz[:, :self.__m] - self.__tx)).sum(axis=1) / self.__total_pair_weights

        self.__v01 = (self.__tz[:, :self.__m] - self.__tx[:, :]) / self.__total_negative_weights

        self.__v10 = 1. - (self.__tz[:, self.__m:] - self.__ty[:, :]) / self.__total_positive_weights

        self.__sx = np.cov(self.__v01)

        self.__sy = np.cov(self.__v10)

        self.__delongcov = self.__sx / self.__m + self.__sy / self.__n

        
        return self.__aucs, self.__delongcov
    
    
    def fastDeLong_no_weights(self, predictions_sorted_transposed, label_1_count):
        
        # Short variables are named as they are in the paper

        self.__m = label_1_count
        self.__n = predictions_sorted_transposed.shape[1] - self.__m
        self.__positive_examples = predictions_sorted_transposed[:, :self.__m]
        self.__negative_examples = predictions_sorted_transposed[:, self.__m:]
        self.__k = predictions_sorted_transposed.shape[0]



        self.__tx = np.empty([self.__k, self.__m], dtype=float)
        self.__ty = np.empty([self.__k, self.__n], dtype=float)
        self.__tz = np.empty([self.__k, self.__m + self.__n], dtype=float)

        for self.__r in range(self.__k):

            self.__tx[self.__r, :] = self.compute_midrank(self.__positive_examples[self.__r, :])
            self.__ty[self.__r, :] = self.compute_midrank(self.__negative_examples[self.__r, :])
            self.__tz[self.__r, :] = self.compute_midrank(predictions_sorted_transposed[self.__r, :])

        self.__aucs = self.__tz[:, :self.__m].sum(axis=1) / self.__m / self.__n - float(self.__m + 1.0) / 2.0 / self.__n
        self.__v01 = (self.__tz[:, :self.__m] - self.__tx[:, :]) / self.__n
        self.__v10 = 1.0 - (self.__tz[:, self.__m:] - self.__ty[:, :]) / self.__m

        self.__sx = np.cov(self.__v01)
        self.__sy = np.cov(self.__v10)
        self.__delongcov = self.__sx / self.__m + self.__sy / self.__n

        return self.__aucs, self.__delongcov
    
    
    def fastDeLong(self, predictions_sorted_transposed, label_1_count, sample_weight=None):
        
        if sample_weight is None:
            
            return self.fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    
        else:
        
            return self.fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)
    
    
    def calc_pvalue(self, aucs, sigma):
        
        self.__l = np.array([[1, -1]])

        self.__z = np.abs(np.diff(aucs)) / (np.sqrt(np.dot(np.dot(self.__l, sigma), self.__l.T)) + 1e-8)
        self.__pvalue = 2 * (1 - scipy.stats.norm.cdf(np.abs(self.__z)))
        #  print(10**(np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)))
        
        return self.__pvalue
    
    
    def compute_ground_truth_statistics(self, ground_truth, sample_weight=None):
        assert np.array_equal(np.unique(ground_truth), [0, 1])
        self.__order = (-ground_truth).argsort()
        self.__label_1_count = int(ground_truth.sum())
        if sample_weight is None:
            self.__ordered_sample_weight = None
        else:
            self.__ordered_sample_weight = sample_weight[self.__order]

        return self.__order, self.__label_1_count, self.__ordered_sample_weight
    
    
    def delong_roc_variance(self, ground_truth, predictions):
        
        self.__sample_weight = None
    
        self.__order, self.__label_1_count, self.__ordered_sample_weight = self.compute_ground_truth_statistics(ground_truth, self.__sample_weight)
        self.__predictions_sorted_transposed = predictions[np.newaxis, self.__order]
        self.__aucs, self.__delongcov = self.fastDeLong(self.__predictions_sorted_transposed, self.__label_1_count)

        assert len(self.__aucs) == 1, "There is a bug in the code, please forward this to the developers"
        
        return self.__aucs[0], self.__delongcov
    
    
    def delong_roc_test(self, ground_truth, predictions_one, predictions_two):
        
        self.__sample_weight = None
        self.__order, self.__label_1_count,self.__ordered_sample_weight = self.compute_ground_truth_statistics(ground_truth)
        self.__predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, self.__order]
        self.__aucs, self.__delongcov = self.fastDeLong(self.__predictions_sorted_transposed, self.__label_1_count, 
                                                        self.__sample_weight)

        return self.calc_pvalue(self.__aucs, self.__delongcov)
    
    
    def delong_roc_ci(self, y_true, y_pred):
        
        self.__aucs, self.__auc_cov = self.delong_roc_variance(y_true, y_pred)
        self.__auc_std = np.sqrt(self.__auc_cov)
        self.__lower_upper_q = np.abs(np.array([0, 1]) - (1 - self.__alpha) / 2)
        self.__ci = stats.norm.ppf(self.__lower_upper_q, loc=self.__aucs, scale=self.__auc_std)
        self.__ci[self.__ci > 1] = 1
        
        return self.__aucs, self.__ci
    

    def test(self, alpha=0.95):
        """
        test()方法用于启动Delong检验
        
        参数介绍
        ========
        alpha：置信水平，类型为float，默认为0.95，即95%的置信区间
        
        """
        assert type(alpha) == type(0.95),"alpha参数错误，请检查！"
        self.__alpha = alpha
        
        self.__outputlist = []
        
        # 两两配对检验
        for self.__r1 in range(0,len(self.__testlist)-1):
            for self.__r2 in range(self.__r1+1, len(self.__testlist)):
                self.__y_pred_1, self.__name_1 = self.__testlist[self.__r1]
                self.__y_pred_2, self.__name_2 = self.__testlist[self.__r2]
                self.__pvalue = self.delong_roc_test(self.__y_true,self.__y_pred_1,self.__y_pred_2)
                self.__auc_1, self.__ci_1 = self.delong_roc_ci(self.__y_true, self.__y_pred_1)
                self.__auc_2, self.__ci_2 = self.delong_roc_ci(self.__y_true, self.__y_pred_2)
                self.__namelist = [self.__name_1, self.__name_2]
                self.__auroclist = [self.__auc_1, self.__auc_2]
                self.__cilist = [self.__ci_1, self.__ci_2]
                self.__plist = [self.__pvalue[0][0]]
                self.__List = [self.__namelist, self.__auroclist, self.__cilist, self.__plist]
                self.__outputlist.append(self.__List)
                self.__namelist = None
                self.__auroclist = None
                self.__cilist = None
                self.__plist = None
                self.__List = None
        
        # 输出结果
        print("The Result of Delong Test：")
        for self.__op in range(0, len(self.__outputlist)):
            print(" ")
            print("============================")
            print("【%s & %s】" % (self.__outputlist[self.__op][0][0], self.__outputlist[self.__op][0][1]))
            print(" AUROC of %s：" % self.__outputlist[self.__op][0][0], 
                  "%.3f" % self.__outputlist[self.__op][1][0], 
                  "(%.3f - %.3f)" % (self.__outputlist[self.__op][2][0][0], self.__outputlist[self.__op][2][0][1]))
            print(" AUROC of %s：" % self.__outputlist[self.__op][0][1], 
                  "%.3f" % self.__outputlist[self.__op][1][1], 
                  "(%.3f - %.3f)" % (self.__outputlist[self.__op][2][1][0], self.__outputlist[self.__op][2][1][1]))
            print(" P-Value：%.4f" % self.__outputlist[self.__op][3][0])
            print("============================")
        
        return None

"""========================================================="""

"""================== PlotLearningCurve ===================="""
class PlotLearningCurve:
    
    """
    PlotLearningCurve()方法，用于绘制学习曲线，以判断模型是否存在过拟合
    
    参数介绍
    ========
    model：基于sklearn，当 dnn_model=True 时，该参数失效，且必须传入参数 dnn_path
    
    x：训练集数据,类型为Dataframe，默认为None
    
    y：训练集标签,类型为trainDataFrame或Series，默认为None
    
    method：训练集的更新方式，有'n'、'c'和'v'三种，默认为'n'模式
    
            'n'——常规模式，即按比例随机划分出训练集和验证集各一个，需要设定参数 s
            'c'——交叉验证，通过交叉验证，以平均准确度为评价指标，需要设定参数 k
            'v'——验证集，用传入的验证集进行学习曲线的绘制，需要参数 x_v, y_v
             
    s：'n'（常规模式）模式下，训练集的size，类型为float，范围为(0, 1)，默认为0.8
    
    k：'c'（交叉验证）模式下，交叉验证的折数，类型为int，范围为[2, +∞)，默认为3
    
    x_v：'v'（验证集）模式下，传入的不带标签的验证集，类型为Dataframe，默认为None
    
    y_v：'v'（验证集）模式下，传入的验证集标签，类型为trainDataFrame或Series，默认为None
    
    step_size：每次增加的训练样本比例，类型为float，取值为（0, 1），默认为0.01
    
    p：训练集中，训练样本的初始比例，类型为float，取值为（min_p, 1）,默认为None；★会自动计算最小比例，可根据计算结果更改
    
    train_proportion：固定训练集比例，类型为list，例如[0.25, 0.5, 0.75]默认为None；★当设置该参数时，step_size和p参数失效
    
    r：全局随机种子，类型为None或int，默认为None
    
    dnn_model：是否为keras的DNN模型，类型为bool，默认为False
    
    -----以下参数只有 dnn_model=True 时有效-----
    
    path：所要传入的，未fit的，待绘制学习曲线的DNN模型，类型为str，默认为None（★必须传入！）
    
    epochs：基于Keras的DNN模型训练时的迭代次数,类型为int，默认为50
    
    batch_size：基于Keras的DNN模型训练时,每次用于迭代的样本数，类型为int，默认为64
    
    shuffle：基于Keras的DNN模型训练时的是否打乱，默认为"True"
    
    verbose：基于Keras的DNN模型训练时的参数，默认为0
    
    weight：阳性样本的权重，类型为int或float，默认为1（★注意！是阳性样本的权重）
    
    ========
    """
    
    def __init__(self, model=None, x=None, y=None, method='n', s=0.8, k=3, x_v=None, y_v=None, step_size=0.01, 
                 p=None, train_proportion=None, r=None, 
                 dnn_model=False, path=None, epochs=50, batch_size=64, shuffle='True', verbose=0, weight=1):
        
        # 参数检查
        
        # 模型
        if dnn_model:
            self.__dnn = True
            if type(path) == type('DNN'):
                self.__model = copy.deepcopy(path)
            else:
                raise ValueError("path参数异常，请检查（path的类型需要为str，传入的类型为：%s）" % type(path))
        else:
            self.__dnn = False
            if type(model) == type(None):
                raise ValueError("缺少model参数，请传入model参数")
            else:
                self.__model = copy.deepcopy(model)
        
        # x
        if type(x) == type(None):
            raise ValueError("缺少x参数，请传入x参数")
        else: 
            if type(x) != type(pd.DataFrame()):
                raise ValueError("x的类型必须为DataFrame，而传入的类型为%s" % type(x))
            else:
                self.__x = x.copy()
        
        # y
        if type(y) == type(None):
            raise ValueError("缺少y参数，请传入y参数")
        else: 
            if type(y) != type(pd.DataFrame()):
                if type(y) != type(pd.Series([1])):
                    raise ValueError("y的类型必须为DataFrame或Series，而传入的类型为%s" % type(y))
                else:
                    self.__y = y.copy()
            else:
                self.__y = y.copy()
                
        # method
        if type(method) == type('c'):
            if method == 'n':
                self.__method = 'n'
                if type(s) == type(0.1):
                    if s > 0 and s < 1:
                        self.__s = s
                    else:
                        raise ValueError("参数 s 的范围为(0,1)")
                else:
                    raise ValueError("s的类型必须为flaot，而传入的类型为%s" % type(s))
            elif method == 'c':
                self.__method = 'c'
                if type(k) == type(int(3)):
                    if k >= 2:
                        self.__k = k
                    else:
                        raise ValueError("参数 k 的范围为[2, +∞)")
                else:
                    raise ValueError("k的类型必须为int，而传入的类型为%s" % type(k))
            elif method == 'v':
                self.__method = 'v'
                if type(x_v) == type(None):
                    raise ValueError("缺少x_v参数，请传入x_v参数")
                else: 
                    if type(x_v) != type(pd.DataFrame()):
                        raise ValueError("x_v的类型必须为DataFrame，而传入的类型为%s" % type(x_v))
                    else:
                        self.__xv = x_v.copy()
                if type(y_v) == type(None):
                    raise ValueError("缺少y_v参数，请传入y_v参数")
                else: 
                    if type(y_v) != type(pd.DataFrame()):
                        if type(y_v) != type(pd.Series([1])):
                            raise ValueError("y_v的类型必须为DataFrame或Series，而传入的类型为%s" % type(y_v))
                        else:
                            self.__yv = y_v.copy()
                    else:
                        self.__yv = y_v.copy()
            else:
                raise ValueError("method参数异常，请检查")
        else:
            raise ValueError("method的类型必须为str，且内容为n'、'c'或'v'，传入的类型为%s" % type(method))
                
        # step_size
        if type(step_size) == type(0.01):
            self.__ss = step_size
        else:
            raise ValueError("step_size的类型必须为float，而传入的类型为%s" % type(step_size))
            
        # p
        assert type(p) == type(0.01), "p参数异常，请检查，p参数的类型为float，范围为(0,1)"
        self.__p = p
        
        # train_proportion
        if type(train_proportion) != type(None):
            self.__tpState = True
            assert type(train_proportion) == type([]), "train_proportion参数类型为list，请检查！"
            self.__train_proportion = train_proportion
            self.__p = self.__train_proportion[0]
        else:
            self.__tpState = False
        
        # r
        if type(r) == type(int(42)):
            self.__r = r
        else:
            if type(r) == type(None):
                self.__r = r
            else:
                raise ValueError(" r 的类型必须为int，而传入的类型为%s" % type(r))
                
        # Keras相关参数检查
        if self.__dnn:
            
            if type(epochs) == type(50):
                self.__epochs = epochs
            else:
                raise ValueError("epochs的类型必须为int，而传入的类型为%s" % type(epochs))
                
            if type(batch_size) == type(64):
                self.__batch_size = batch_size
            else:
                raise ValueError("batch_size的类型必须为int，而传入的类型为%s" % type(batch_size))
            
            if  type(shuffle) == type("False"):
                if shuffle == "False":
                    self.__shuffle = shuffle
                else:
                    if shuffle == "True":
                        self.__shuffle = shuffle
                    else:
                        raise ValueError("shuffle的类容为“True”或“False”")
            else:
                raise ValueError("shuffle的类型必须为str")
            
            self.__verbose = verbose
            
            if type(weight) == type(1):
                self.__weight = weight
            else:
                if type(weight) == type(0.1):
                    self.__weight = weight
                else:
                    raise ValueError("weight的类型必须为int或float")
                    
        # 数据集的确定
        if self.__method == 'n':
            self.__t_x, self.__v_x, self.__t_y, self.__v_y = train_test_split(self.__x, self.__y, train_size=self.__s, 
                                                                              random_state=self.__r, stratify=self.__y)
        elif self.__method == 'c':
            self.__x = self.__x.values
            self.__y = self.__y.values
            self.__data_k = StratifiedKFold(n_splits=self.__k, shuffle=True, 
                                            random_state=self.__r).split(self.__x, self.__y)
            self.__t_x = []
            self.__v_x = []
            self.__t_y = []
            self.__v_y = []
            self.__number = 0
            for self.__t, self.__v in self.__data_k:
                self.__number = self.__number + 1
                self.__t_x.append(self.__x[self.__t])
                self.__v_x.append(self.__x[self.__v])
                self.__t_y.append(self.__y[self.__t])
                self.__v_y.append(self.__y[self.__v])
        elif self.__method == 'v':
            self.__t_x = self.__x.copy()
            self.__v_x = self.__xv.copy()
            self.__t_y = self.__y.copy()
            self.__v_y = self.__yv.copy()
        else:
            assert False, "代码有Bug，请联系打包者"
        
        # 最小比例计算和步长合理性检查
        if self.__method == 'c':
            self.__zsamples = len(self.__t_y[0])
            self.__p_rate = self.__t_y[0].sum() / self.__zsamples
            self.__minsample = int(int(1 / self.__p_rate) + 1)
            assert self.__minsample < self.__zsamples, "数据异常请检查"
            self.__p_min = round(self.__minsample/self.__zsamples, 2)
            assert self.__p_min <= 0.99, "数据异常请检查"
            if self.__p < self.__p_min:
                raise ValueError("训练集中训练样本的最小初始比例为：%s，请重新设定" % self.__p_min)
            self.__pb = 0.99 - self.__p
            
            if self.__pb == 0:
                self.__times = 1
            else:
                self.__times = int(self.__pb // self.__ss)
            
            self.__remainder = self.__pb % self.__ss
            if self.__remainder != 0:
                self.__cycle = True
                self.__times = self.__times + 1
            else:
                self.__cycle = False
            
            self.__op = self.__p
            
        else:
            self.__zsamples = len(self.__t_y)
            self.__p_rate = self.__t_y.sum() / self.__zsamples
            self.__minsample = int(int(1 / self.__p_rate) + 1)
            assert self.__minsample < self.__zsamples, "数据异常请检查"
            self.__p_min = round(self.__minsample/self.__zsamples, 2)
            assert self.__p_min <= 0.99, "数据异常请检查"
            if self.__p < self.__p_min:
                raise ValueError("训练集中训练样本的最小初始比例为：%s，请重新设定" % self.__p_min)
            self.__pb = 0.99 - self.__p
            
            if self.__pb == 0:
                elf.__times = 1
            else:
                self.__times = int(self.__pb // self.__ss)
            
            self.__remainder = self.__pb % self.__ss
            if self.__remainder != 0:
                self.__cycle = True
                self.__times = self.__times + 1
            else:
                self.__cycle = False
                
            self.__op = self.__p
            
        # 输出最小起始步长
        if self.__method == 'n':
            print("学习曲线的绘制方法为：常规方法")
            print("用于训练的数据集占传入数据集的%s" % self.__s)
            if self.__tpState:
                print ("每次用于模型训练的训练集样本比例为：", self.__train_proportion)
            else:
                print("最小初始比列为：%s，设定的比例：%s" % (self.__p_min, self.__op))
                print("每次增加样本比例数为：%s" % self.__ss)
        elif self.__method == 'c':
            print("学习曲线的绘制方法为：%s折交叉验证" % self.__k)
            if self.__tpState:
                print ("每次用于模型训练的训练集样本比例为：", self.__train_proportion)
            else:
                print("最小初始比列为：%s，设定的比例：%s" % (self.__p_min, self.__op))
                print("每次增加样本比例数为：%s" % self.__ss)
        else:
            print("学习曲线的绘制方法为：传入验证集方法")
            if self.__tpState:
                print ("每次用于模型训练的训练集样本比例为：", self.__train_proportion)
            else:
                print("最小初始比列为：%s，设定的比例：%s" % (self.__p_min, self.__op))
                print("每次增加样本比例数为：%s" % self.__ss)
            
        # 绘图的相关设置
        self.__title = "Learning Curve of Model"
        self.__xlabel = "Numbers of Traning Sample"
        self.__ylabel = "Accuracy"
        self.__color = ['#ff6d6d', '#79a4ea']
        self.__marker = None
        self.__linestyle = '-'
        
        self.__mState = False
        self.__state = False
        
        return None
    
    def lc_set(self, title=None, xlabel=None, ylabel=None, colorlist=None, marker=None, linestyle=None):
        """
        lc_set()方法用于学习曲线绘制的相关设置
        
        参数介绍
        ========
        title：标题，类型为str，默认为None，即"Learning Curve of Model"
        
        xlabel：X轴标题，类型为str，默认为None，即"Numbers of Sample"
        
        ylabel：Y轴标题，类型为str，默认为None，即"Accuracy"
        
        colorlist：颜色列表，类型为list，默认为None，即['#ff6d6d', '#79a4ea']
        
        marker：点型，类型为str，默认为None，即不显示点
        
        linestyle：线型，类型为str，默认为None，即“实线”
        
        ========
        """
        
        if type(title) != type(None):
            if type(title) == type("Learning Curve of Model"):
                self.__title = title
            else:
                raise ValueError("title的类型必须为str，而传入的类型为%s" % type(title))
                
        if type(xlabel) != type(None):
            if type(xlabel) == type("Numbers of Sample"):
                self.__xlabel = xlabel
            else:
                raise ValueError("xlabel的类型必须为str，而传入的类型为%s" % type(xlabel))
                
        if type(ylabel) != type(None):
            if type(ylabel) == type("Accuracy"):
                self.__ylabel = ylabel
            else:
                raise ValueError("ylabel的类型必须为str，而传入的类型为%s" % type(ylabel))
                
        if type(colorlist) != type(None):
            if type(colorlist) == type([]):
                if len(colorlist) >= 2:
                    self.__color = colorlist
                else:
                    raise ValueError("color的数量至少等于2，而传入的数量为：%s" % len(colorlist))
            else:
                raise ValueError("colorlist的类型必须为list，而传入的类型为%s" % type(colorlist))
        
        if type(marker) !=  type(None):
            if type(marker) == type('.'):
                self.__mState = True
                self.__marker = marker
            else:
                raise ValueError("marker的类型必须为str，而传入的类型为%s" % type(marker)) 
        
        if type(linestyle) != type(None):
            if type(linestyle) == type('-'):
                self.__linestyle = linestyle
            else:
                raise ValueError("linestyle的类型必须为str，而传入的类型为%s" % type(linestyle)) 
        
        return None
    
    def plot(self):
        """plot()方法用于绘制学习曲线"""
        
        # 学习曲线计算
        self.__train_accuracy = []
        self.__test_accuracy = []
        self.__sample_list = []
        self.__mmodel = None
        self.__tx = None
        self.__ty = None
        self.__nx = None
        self.__ny = None
        self.__train_pred = None
        self.__test_pred = None
        
        if self.__method == 'c':
            for self.__l1 in range(0, self.__times+1):
                self.__train_accuracy.append([])
                self.__test_accuracy.append([])
                
            for self.__l2 in range(0, self.__times):
                for self.__l3 in range(0, self.__k):
                    self.__tx, self.__nx, self.__ty, self.__ny = train_test_split(self.__t_x[self.__l3], 
                                                                                  self.__t_y[self.__l3], 
                                                                                  train_size=self.__p, 
                                                                                  stratify=self.__t_y[self.__l3])
                    if self.__l3 == 0:
                        self.__sample_list.append(len(self.__ty))
                    
                    
                    if self.__dnn:
                        self.__mmodel = load_model(self.__model)
                        self.__mmodel.fit(self.__tx, self.__ty, epochs=self.__epochs, shuffle=self.__shuffle, 
                                          batch_size=self.__batch_size, verbose=self.__verbose, 
                                          class_weight={0:1,1:self.__weight})
                        self.__train_pred = self.__mmodel.predict(self.__tx)
                        self.__train_pred[self.__train_pred >= 0.5] = 1
                        self.__train_pred[self.__train_pred < 0.5] = 0
                        self.__test_pred = self.__mmodel.predict(self.__v_x[self.__l3])
                        self.__test_pred[self.__test_pred >= 0.5] = 1
                        self.__test_pred[self.__test_pred < 0.5] = 0
                        self.__train_accuracy[self.__l2].append(accuracy_score(self.__ty, self.__train_pred))
                        self.__test_accuracy[self.__l2].append(accuracy_score(self.__v_y[self.__l3], self.__test_pred))
                        self.__mmodel = None
                    else:
                        self.__mmodel = copy.deepcopy(self.__model)
                        self.__mmodel.fit(self.__tx, self.__ty)
                        self.__train_pred = self.__mmodel.predict(self.__tx)
                        self.__test_pred = self.__mmodel.predict(self.__v_x[self.__l3])
                        self.__train_accuracy[self.__l2].append(accuracy_score(self.__ty, self.__train_pred))
                        self.__test_accuracy[self.__l2].append(accuracy_score(self.__v_y[self.__l3], self.__test_pred))
                        self.__mmodel = None
                
                if self.__cycle:
                    if self.__l2 == self.__times-2:
                        self.__p = self.__p + self.__remainder
                    else:
                        self.__p = self.__p + self.__ss
                else:
                    self.__p = self.__p + self.__ss
            
            for self.__l4 in range(0, self.__k):
                
                if self.__l4 == 0:
                    self.__sample_list.append(len(self.__t_y[self.__l4]))
                
                if self.__dnn:
                    self.__mmodel = load_model(self.__model)
                    self.__mmodel.fit(self.__t_x[self.__l4], self.__t_y[self.__l4], epochs=self.__epochs, 
                                      batch_size=self.__batch_size, shuffle=self.__shuffle, 
                                      verbose=self.__verbose, class_weight={0:1,1:self.__weight})
                    self.__train_pred = self.__mmodel.predict(self.__t_x[self.__l4])
                    self.__train_pred[self.__train_pred >= 0.5] = 1
                    self.__train_pred[self.__train_pred < 0.5] = 0
                    self.__test_pred = self.__mmodel.predict(self.__v_x[self.__l4])
                    self.__test_pred[self.__test_pred >= 0.5] = 1
                    self.__test_pred[self.__test_pred < 0.5] = 0
                    self.__train_accuracy[self.__times].append(accuracy_score(self.__t_y[self.__l4], self.__train_pred))
                    self.__test_accuracy[self.__times].append(accuracy_score(self.__v_y[self.__l4], self.__test_pred))
                    self.__mmodel = None
                else:
                    self.__mmodel = copy.deepcopy(self.__model)
                    self.__mmodel.fit(self.__t_x[self.__l4], self.__t_y[self.__l4])
                    self.__train_pred = self.__mmodel.predict(self.__t_x[self.__l4])
                    self.__test_pred = self.__mmodel.predict(self.__v_x[self.__l4])
                    self.__train_accuracy[self.__times].append(accuracy_score(self.__t_y[self.__l4], self.__train_pred))
                    self.__test_accuracy[self.__times].append(accuracy_score(self.__v_y[self.__l4], self.__test_pred))
                    self.__mmodel = None
            
            assert len(self.__sample_list) == len(self.__train_accuracy), "代码BUG请检查！"
            assert len(self.__sample_list) == len(self.__test_accuracy), "代码BUG请检查！"
            assert len(self.__train_accuracy[0]) == self.__k, "代码BUG请检查！"
            assert len(self.__test_accuracy[0]) == self.__k, "代码BUG请检查！"
            
            self.__ta = []
            self.__va = []
            for self.__l5 in range(0, len(self.__sample_list)):
                self.__th = 0
                self.__vh = 0
                for self.__l6 in range(0, self.__k):
                    self.__th = self.__th + self.__train_accuracy[self.__l5][self.__l6]
                    self.__vh = self.__vh + self.__test_accuracy[self.__l5][self.__l6]
                self.__th = self.__th / self.__k
                self.__vh = self.__vh / self.__k
                self.__ta.append(self.__th)
                self.__va.append(self.__vh)
                
            self.__train_accuracy = self.__ta
            self.__test_accuracy = self.__va
        
        else:
            for self.__l1 in range(0, self.__times):
                self.__tx, self.__nx, self.__ty, self.__ny = train_test_split(self.__t_x, self.__t_y, 
                                                                              train_size=self.__p, 
                                                                              stratify=self.__t_y)
                
                if self.__dnn:
                    self.__mmodel = load_model(self.__model)
                    self.__mmodel.fit(self.__tx, self.__ty, epochs=self.__epochs, shuffle=self.__shuffle, 
                                      batch_size=self.__batch_size, verbose=self.__verbose, 
                                      class_weight={0:1,1:self.__weight})
                    self.__train_pred = self.__mmodel.predict(self.__tx)
                    self.__train_pred[self.__train_pred >= 0.5] = 1
                    self.__train_pred[self.__train_pred < 0.5] = 0
                    self.__test_pred = self.__mmodel.predict(self.__v_x)
                    self.__test_pred[self.__test_pred >= 0.5] = 1
                    self.__test_pred[self.__test_pred < 0.5] = 0
                    self.__train_accuracy.append(accuracy_score(self.__ty, self.__train_pred))
                    self.__test_accuracy.append(accuracy_score(self.__v_y, self.__test_pred))
                    self.__sample_list.append(len(self.__ty))
                    self.__mmodel = None
                else:
                    self.__mmodel = copy.deepcopy(self.__model)
                    self.__mmodel.fit(self.__tx, self.__ty)
                    self.__train_pred = self.__mmodel.predict(self.__tx)
                    self.__test_pred = self.__mmodel.predict(self.__v_x)
                    self.__train_accuracy.append(accuracy_score(self.__ty, self.__train_pred))
                    self.__test_accuracy.append(accuracy_score(self.__v_y, self.__test_pred))
                    self.__sample_list.append(len(self.__ty))
                    self.__mmodel = None
                    
                if self.__cycle:
                    if self.__l1 == self.__times-2:
                        self.__p = self.__p + self.__remainder
                    else:
                        self.__p = self.__p + self.__ss
                else:
                    self.__p = self.__p + self.__ss
                    
            if self.__dnn:
                self.__mmodel = load_model(self.__model)
                self.__mmodel.fit(self.__t_x, self.__t_y, epochs=self.__epochs, shuffle=self.__shuffle, 
                                  batch_size=self.__batch_size, verbose=self.__verbose, 
                                  class_weight={0:1,1:self.__weight})
                self.__train_pred = self.__mmodel.predict(self.__t_x)
                self.__train_pred[self.__train_pred >= 0.5] = 1
                self.__train_pred[self.__train_pred < 0.5] = 0
                self.__test_pred = self.__mmodel.predict(self.__v_x)
                self.__test_pred[self.__test_pred >= 0.5] = 1
                self.__test_pred[self.__test_pred < 0.5] = 0
                self.__train_accuracy.append(accuracy_score(self.__t_y, self.__train_pred))
                self.__test_accuracy.append(accuracy_score(self.__v_y, self.__test_pred))
                self.__sample_list.append(len(self.__t_y))
                self.__mmodel = None
            else:
                self.__mmodel = copy.deepcopy(self.__model)
                self.__mmodel.fit(self.__t_x, self.__t_y)
                self.__train_pred = self.__mmodel.predict(self.__t_x)
                self.__test_pred = self.__mmodel.predict(self.__v_x)
                self.__train_accuracy.append(accuracy_score(self.__t_y, self.__train_pred))
                self.__test_accuracy.append(accuracy_score(self.__v_y, self.__test_pred))
                self.__sample_list.append(len(self.__t_y))
                self.__mmodel = None
        
        plt.close()
        plt.figure(figsize=(15,15), dpi=300, facecolor='w')
        plt.subplot(224)
        
        plt.plot(self.__sample_list, self.__train_accuracy, lw=1, color=self.__color[0], label="Training Set")
        plt.plot(self.__sample_list, self.__test_accuracy, lw=1, color=self.__color[1], label="Validation Set")
        
        if self.__method == 'c':
            plt.xlim([10.0, len(self.__t_y[0])])
        else:
            plt.xlim([10.0, len(self.__t_y)])
        
        plt.ylim([0.0, 1.05])
        plt.xlabel( self.__xlabel)
        plt.ylabel( self.__ylabel)
        plt.title(self.__title)
        plt.legend(loc="lower right")
        self.__fig_cl = plt.gcf()
        plt.show()
        
        self.__state = True
        
        return  None
    
    def save(self, path="LC.png", dpi=300):
        """
        save()方法用于保存绘制的学习曲线
        
        参数介绍
        ========
        
        path：保存路径，类型为str，格式为“/file1/file2/image_name.png”；默认保存在当前路径，名字为"LC.png"
        
        dpi：每英寸上，所能印刷的网点数，类型为int，默认为300
        
        ========
        """
        
        if self.__state:
            self.__path = path
            self.__dpi = dpi
            
            self.__fig_cl.savefig(self.__path, dpi=self.__dpi, bbox_inches = 'tight')
            print("LC已保存，路径为：%s" % self.__path)
        else:
            raise RuntimeError("尚未绘制LC，请先调用plot()方法")
            
        return None
        
"""========================================================="""

"""===================== CompareModel ======================"""
class CompareModel:
    """
    CompareModel 3.2 用于其他模型的比较（包括评分量表类模型），并于混淆矩阵（ConfusionMatrix）同接口，可以与PlotROC、PlotPRC以及DelongTest联用
    
    【仅支持二分类预测模型，评分量表仅支持得分类量表】
    
    参数介绍
    ========
    model_type：被比较的模型的类型，类型为str，默认为："probability"
                ① "probability"，预测值为概率的一类的模型
                ② "score"，预测值为评分的一类模型
    
    probability：当model_type为“probability”时有效，表示被比较模型的预测概率，传入前请确保传入类型正确，并与真实标签对齐，默认为None
    
    score：当model_type为“score”时有效，表示被比较模型的预测得分，传入前请确保传入类型正确，并与真实标签对齐，默认为None
    
    score_pattern：当model_type为“score”时有效，表示评分的模式，类型为str，默认为“asc”，即升序
                   ① “asc”：升序，表示评分大于并等于某个数值时，预测为阳性，如STOP-BANG评分
                   ② “desc”：降序，表示评分小于并等于某个数值时，预测为阳性，如GCS评分
    
    full_score：当model_type为“score”时有效，被比较的评分量表的最高分值，类型为int，默认为None
    
    true_label：对应的真实标签，传入前请确保传入类型正确，并与传入的预测概率对齐
    
    threshold：被比较模型阈值的确定方法，类型为str、float或int；默认为以约登指数寻找最佳阈值
               ① "Youden"：以约登指数寻找最佳阈值
               ② "F1"：以F1值寻找最佳阈值
               ③ float：手动设置最佳阈值，[0,1)
               ④ int：当模型的比较类型为“score”的时候，手动设置评分的阈值（大于等于该阈值时被标记为阳性）
               
    AUPRC_mod：表示计算AUPRC的方法，类型为str，默认未“trapezoidal”，即梯形规则法
                ①“trapezoidal”：梯形规则法
                ②“interpolate”：非线性插值法
                
    ========
    
    
    方法介绍
    ========
    comparing()：启动模型比较，并输出被比较模型的相应指标，为其他接口的前提
    ========
    """
    
    def __init__(self, model_type="probability", probability=None, score=None, score_pattern='asc', full_score=None, 
                 true_label=None, threshold='Youden', AUPRC_mod='trapezoidal'):
        
        # 参数检查
        if model_type == "probability":
            self.__type = 1
        else:
            if model_type == "score":
                self.__type = 2
            else:
                raise  ValueError("model_type应为“probability”或“score”类型为str，传入参数为：%s[%s]" % (model_type, type(model_type)))
        
        if self.__type == 1:
            if type(probability) == type(None):
                raise ValueError("请传入probability参数，并确保其类型正确，同时与真实标签对齐")
            else:
                self.__yscore = copy.deepcopy(probability)
                self.__ypred = copy.deepcopy(probability)
        elif self.__type == 2:
            if type(score) == type(None):
                raise ValueError("请传入score参数，并确保其类型正确，同时与真实标签对齐")
            else:
                if type(full_score) == type(None):
                    raise ValueError("请传入full_score参数,并确保其正确")
                else:
                    if score_pattern == 'asc':
                        self.__probability = score/full_score
                    
                    elif score_pattern == 'desc':
                        self.__probability = 1 - score/full_score
                        
                    else:
                        raise ValueError("score_pattern参数只能为“asc”或“desc”，错误参数%s[%s]" % (score_pattern,type(score_pattern)))
                        
                    self.__yscore = copy.deepcopy(self.__probability)
                    self.__ypred = copy.deepcopy(self.__probability)
                    # print(self.__probability)
        else:
            raise ValueError("代码有BUG，请告知包装人员修复")
            
            
        if type(true_label) == type(None):
            raise ValueError("请传入true_label参数，并确保其类型正确，同时与传入概率对齐")
        else:
            self.__y = copy.deepcopy(true_label)
            
        if type(threshold) == type("Youden"):
            if threshold == "Youden":
                self.__switch = 1
            elif threshold == "F1":
                self.__switch = 2
            else:
                raise ValueError("threshold参数错误，请检查") 
        else:
            if type(threshold) == type(0.12):
                if threshold > 1:
                    raise ValueError("threshold参数错误，请检查")
                else:
                    self.__switch = 3
                    self.__optimal_threshold = threshold
            elif  type(threshold) == type(1):
                self.__switch = 3
                if score_pattern == 'asc':
                    self.__optimal_threshold = threshold/full_score
                else:
                    self.__optimal_threshold = 1 - threshold/full_score
                # print(self.__optimal_threshold)
            else:
                raise ValueError("threshold参数错误，请检查")
        
        if AUPRC_mod == 'trapezoidal':
            self.__modstate = 1
        elif AUPRC_mod == 'interpolate':
            self.__modstate = 2
        else:
            raise ValueError("AUPRC_mod参数错误，其值只能为“trapezoidal”或“interpolate”")
        
        # 设置接口开关，以确保调用接口前，先调用comparing()方法
        self.__comparing_states = False
        
        return None
    
    def comparing(self):
        """comparing()方法，用于启动模型比较，并输出被比较模型的相应指标，为其他接口的前提"""
        
        # 计算不同阈值下的FPR、TPR、Recall、Precision
        self.__FPR, self.__TPR, self.__thresholds = roc_curve(self.__y,self.__yscore)
        self.__precision, self.__recall, self.__thresholds2 = precision_recall_curve(self.__y,self.__yscore)
        
        # 设定特定阈值
        if self.__switch == 1:
            self.__Y = self.__TPR - self.__FPR
            self.__youdenindex = np.argmax(self.__Y)
            self.__optimal_threshold = self.__thresholds[self.__youdenindex]
            self.__point = [self.__FPR[self.__youdenindex], self.__TPR[self.__youdenindex]]
            
        if self.__switch == 2:
            self.__Y = (2 * self.__precision * self.__recall) / (self.__precision + self.__recall)
            self.__f1_index = np.argmax(self.__Y)
            self.__optimal_threshold = self.__thresholds2[self.__f1_index]
            self.__point = [self.__recall[self.__f1_index], self.__precision[self.__f1_index]]
        
        # 最佳阈值下的混淆矩阵
        self.__ypred[self.__ypred >= self.__optimal_threshold] = 1
        self.__ypred[self.__ypred < self.__optimal_threshold] = 0
        self.__roc = auc(self.__FPR,self.__TPR)
        if self.__modstate ==1:
            self.__prc = auc(self.__recall, self.__precision)
        elif self.__modstate ==2:
            self.__iprauc = AUC(curve='PR')
            self.__iprauc.update_state(self.__y,self.__yscore)
            self.__prc =  np.float64(self.__iprauc.interpolate_pr_auc().numpy())
        else:
            assert False, "代码有Bug，请联系包装者检查"        
        self.__cm = confusion_matrix(self.__y,self.__ypred)
        
        # DeLong 计算AUROC置信区间
        self.__alpha = 0.95
        self.__y_true = self.__y
        assert np.array_equal(np.unique(self.__y_true), [0, 1])
        self.__order = (-self.__y_true).argsort()
        self.__label_1_count = int(self.__y_true.sum())
        self.__ordered_sample_weight = None
        self.__predictions_sorted_transposed = self.__yscore[np.newaxis, self.__order]
        
        self.__m = self.__label_1_count
        self.__n = self.__predictions_sorted_transposed.shape[1] - self.__m
        self.__positive_examples = self.__predictions_sorted_transposed[:, :self.__m]
        self.__negative_examples = self.__predictions_sorted_transposed[:, self.__m:]
        self.__k = self.__predictions_sorted_transposed.shape[0]
        
        self.__tx = np.empty([self.__k, self.__m], dtype=float)
        self.__ty = np.empty([self.__k, self.__n], dtype=float)
        self.__tz = np.empty([self.__k, self.__m + self.__n], dtype=float)
        
        for self.__r in range(self.__k):
            self.__tx[self.__r, :] = self.compute_midrank(self.__positive_examples[self.__r, :])
            self.__ty[self.__r, :] = self.compute_midrank(self.__negative_examples[self.__r, :])
            self.__tz[self.__r, :] = self.compute_midrank(self.__predictions_sorted_transposed[self.__r, :])
        
        self.__delong_aucs = self.__tz[:, :self.__m].sum(axis=1) / self.__m / self.__n - float(self.__m + 1.0) / 2.0 / self.__n
        self.__v01 = (self.__tz[:, :self.__m] - self.__tx[:, :]) / self.__n
        self.__v10 = 1.0 - (self.__tz[:, self.__m:] - self.__ty[:, :]) / self.__m
        
        self.__sx = np.cov(self.__v01)
        self.__sy = np.cov(self.__v10)
        
        self.__auroc_cov = self.__sx / self.__m + self.__sy / self.__n
        assert len(self.__delong_aucs) == 1, "代码有Bug，请告知包装者"
        self.__delong_auc = self.__delong_aucs[0]
        
        self.__auc_std = np.sqrt(self.__auroc_cov)
        self.__lower_upper_q = np.abs(np.array([0, 1])-(1-self.__alpha)/2)
        
        self.__roc_ci = stats.norm.ppf(self.__lower_upper_q, loc=self.__delong_auc, scale=self.__auc_std)
        self.__roc_ci[self.__roc_ci > 1] = 1
        
        # Logit Method 计算AUPRC置信区间
        self.__postivesample = self.__y.sum()
        self.__exponent1 = math.log(self.__prc/(1-self.__prc), math.e) - 1.96*math.pow(self.__postivesample*self.__prc*(1-self.__prc), -1/2)
        self.__exponent2 = math.log(self.__prc/(1-self.__prc), math.e) + 1.96*math.pow(self.__postivesample*self.__prc*(1-self.__prc), -1/2)
        self.__prc_ci = [math.exp(self.__exponent1) / (1 + math.exp(self.__exponent1)), 
                         math.exp(self.__exponent2) / (1 + math.exp(self.__exponent2))]
        
        # 输出评估结果
        print('The Optimal Threshold：%.3f' % self.__optimal_threshold)
        print('')
        print('AUROC：%.3f(%.3f-%.3f)'% (self.__roc, self.__roc_ci[0], self.__roc_ci[1]))
        print('---------')
        print('NOTE: The Confidence Interval for AUROC was Calculated by Delong Method (Delong et al, 1988)')
        print('---------')
        print('')
        print('AUPRC：%.3f(%.3f-%.3f)'% (self.__prc, self.__prc_ci[0], self.__prc_ci[1]))
        if self.__modstate == 1:
            print("Method: Trapezoidal Rule")
        elif self.__modstate == 2:
            print("Method: Interpolation (Davis & Goadrich 2006)")
        else:
            assert False, "Error!"
        print('---------')
        print('NOTE: The Confidence Interval for AUPRC was Calculated by Logit Method (Boyd et al, 2013)')
        print('---------')
        print('')
        print('Sensitivity：%.3f'%(self.__cm[1,1]/(self.__cm[1,1]+self.__cm[1,0])))
        print('Specifity：%.3f'%(self.__cm[0,0]/(self.__cm[0,0]+self.__cm[0,1])))
        print('PPV：%.3f'%(self.__cm[1,1]/(self.__cm[1,1]+self.__cm[0,1])))
        print('NPV：%.3f'%(self.__cm[0,0]/(self.__cm[0,0]+self.__cm[1,0])))
        print('Accuracy：%.3f'%accuracy_score(self.__y,self.__ypred)) 
        print('Precision：%.3f'%precision_score(self.__y,self.__ypred))
        print('Recall：%.3f'%recall_score(self.__y,self.__ypred))
        print('F1-Value：%.3f'%f1_score(self.__y,self.__ypred))
        
        # 更改状态
        self.__comparing_states = True
        
        return None
    
    def compute_midrank(self, x):
        """
        compute_midrank()方法，用于DeLong置信区间中间值计算，仅供内部调用，外部调用无意义
        
        """
        
        self.__J = np.argsort(x)
        self.__Z = x[self.__J]
        self.__N = len(x)
        self.__T = np.zeros(self.__N, dtype=float)
        self.__i = 0
        while self.__i < self.__N:
            self.__j = self.__i
            while self.__j < self.__N and self.__Z[self.__j] == self.__Z[self.__i]:
                self.__j += 1
            self.__T[self.__i:self.__j] = 0.5*(self.__i + self.__j - 1)
            self.__i = self.__j
        self.__T2 = np.empty(self.__N, dtype=float)
        self.__T2[self.__J] = self.__T + 1
        
        return self.__T2
    
    def roc(self):
        """★此方法为PlotROC的接口，仅在ConfusionMatrix中调用无意义★"""
        
        if self.__comparing_states:
            return self.__FPR, self.__TPR, self.__roc
        else:
            raise RuntimeError("请先调用comparing()方法")
    
    def prc(self):
        """★此方法为PlotPRC的接口，仅在ConfusionMatrix中调用无意义★"""
        
        if self.__comparing_states:
            return self.__precision, self.__recall, self.__prc
        else:
            raise RuntimeError("请先调用comparing()方法")
            
    def cc(self):
        """★此方法为PlotCC的接口，仅在ConfusionMatrix中调用无意义★"""
        
        if self.__comparing_states:
            return self.__y, self.__yscore
        else:
            raise RuntimeError("请先调用comparing()方法")
    
    def delong_test(self):
        """★此方法为DelongTest的接口，仅在ConfusionMatrix中调用无意义★"""
        
        if self.__comparing_states:
            return self.__y, self.__yscore
        else:
            raise RuntimeError("请先调用comparing()方法")
            
    def bootstrap_prc(self):
        """★此方法为Bootstrap_PRC的接口，仅在ConfusionMatrix中调用无意义★"""
        
        if self.__comparing_states:
            return self.__y, self.__yscore, self.__prc, self.__modstate, self.__prc_ci
        else:
            raise RuntimeError("请先调用comparing()方法")
            
    def dca(self):
        """★此方法为PlotDCA的接口，仅在ConfusionMatrix中调用无意义★"""
        if self.__comparing_states:
            return self.__yscore
        else:
            raise RuntimeError("请先调用comparing()方法")

"""========================================================="""

"""======================== PlotCC ========================="""
class PlotCC:
    """
    PlotCC用于决策曲线(Calibration Curve, CC)的绘制，需要前置【ConfusionMatrix】使用
    
    ☆注意：PlotCC暂时没有模型概率校准功能，模型概率校准可使用
    
    参数介绍
    ========
    CM_Name_List：是待作CC曲线模型的ConfusionMatrix和模型命名字符串的组合列表,类型为list，默认为None，格式为↓
                  
                  CM_Name_List = [(cm_lr, 'LR'),(cm_dnn,'DNN')]
                  
    ========
    
    
    方法介绍
    ========
    
    cc_set()方法：用于设置CC曲线的相关属性
    
    plot()方法：用于绘制CC曲线

    save()方法：用于保存绘制的曲线
    
    ========
    """
    
    def __init__(self, CM_Name_List=None):
        
        #参数检查
        
        if type(CM_Name_List) == type(None):
            raise ValueError("缺少CM_Name_List参数")
        elif type(CM_Name_List) != type([]):
            raise ValueError("CM_Name_List参数的类型应该为list，而传入的类型为%s" % type(CM_Name_List))
        else:
            self.__list = CM_Name_List
            
        #CC曲线相关参数预设
        
        self.__title = "Calibration Curve"
        self.__xlabel = "Predicted Probability"
        self.__ylabel_c = "Observed Probability"
        self.__ylabel_d = "Count"
        
        self.__bins = []
        for self.__b in range(0, len(self.__list)):
            self.__bins.append(10)
        
        self.__colorlist = ['#ff6d6d', '#79a4ea', '#14f474', '#f5ff63', '#b651f0', '#e4b628', '#1de497', '#e41d6c']
        self.__colorswitch = False
        self.__systemcolor = False
        
        self.__D = False
        self.__plotstate = False
        
        return None
    
    def cc_set(self, title=None, xlabel=None, ylabel_c=None, bins=10, 
               colorlist=None, distribution=False, ylabel_d=None):
        """
        cc_set()方法用于设置ROC曲线的相关属性
        
        参数介绍
        ========
        title：曲线的标题，类型为str，默认为None，即“Calibration Curve”
        
        xlabel：X轴 标题，类型为str，默认为None，即“Predicted Probability”
        
        ylabel_c：校准曲线的 Y轴 标题，类型为str，默认为None，即“Observed Probability”
        
        bins：用于拟合校准曲线的，点的个数，类型为int或者list，默认为int，且为10；若为list数量对应
        
        colorlist：曲线的颜色列表，类型为list，颜色的数目需大于等于曲线的数目，默认为None，即采用默认配色
        
        distribution：是否绘制概率分布图，类型为bool，默认为False
        
        ylabel_d：概率分布图的 Y轴 标题，类型为str，默认为None，即“Count” 
        
        ========
        """
        if type(title) != type(None):
            if type(title) == type("String"):
                self.__title = title
            else:
                raise ValueError("title的类型应为str，而传入的类型为%s" % type(title))
        
        if type(xlabel) != type(None):
            if type(xlabel) == type("String"):
                self.__xlabel = xlabel
            else:
                raise ValueError("xlabel的类型应为str，而传入的类型为%s" % type(xlabel))
        
        if type(ylabel_c) != type(None):
            if type(ylabel_c) == type("String"):
                self.__ylabel_c = ylabel_c
            else:
                raise ValueError("ylabel_c的类型应为str，而传入的类型为%s" % type(ylabel_c))
                
        if type(bins) != type(None):
            if type(bins) == type(int(10)):
                self.__bins = []
                for self.__b in range(0, len(self.__list)):
                    self.__bins.append(bins)
            else:
                if type(bins) == type([]):
                    if len(bins) == len(self.__list):
                        self.__bins = bins
                    else:
                        raise ValueError("bins（%s）与CM_Name_List（%s）数目不匹配" % (len(bins), len(self.__list))) 
                
                else:
                    raise ValueError("bins的类型应为int或者list，而传入的类型为%s" % type(bins))
                
        if type(colorlist) != type(None):
            if type(colorlist) == type([]):
                if len(self.__list) > len(colorlist):
                    raise ValueError("待绘制的ROC曲线数目为%s，colorlist的数目也至少为%s，而传入颜色数为%s" % (len(self.__list), len(self.__list), len(colorlist)))
                else:
                    self.__colorswitch = True
                    self.__colorlist = colorlist
            else:
                raise ValueError("colorlist的类型应为list，而传入的类型为%s" % type(colorlist))
        else:
            if len(self.__list) > 8:
                self.__systemcolor = True
                
        if distribution == True:
            self.__D = True
            if type(ylabel_d) != type(None):
                if type(ylabel_d) == type("Count"):
                    self.__ylabel_d = ylabel_d
                else:
                    raise ValueError("ylabel_d的类型应为str，而传入的类型为%s" % type(ylabel_d)) 
        
        return None
    
    def plot(self):
        """plot方法用于绘制决策曲线图"""
        
        self.__column_1 = []
        self.__column_2 = []
        self.__column_3 = []
        self.__column_4 = []
        
        plt.close()
        plt.figure(figsize=(10,10), dpi=300, facecolor='w')
        
        # 校准曲线图
        self.__ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        
        # 概率分布图
        if self.__D:
            self.__ax2 = plt.subplot2grid((3, 1), (2, 0))
        
        self.__colorindex = 0
        self.__p = 0
        
        # 校准基线
        self.__ax1.plot([0, 1], [0, 1], color='#696e77', label="Perfectly calibrated", lw=1.25, linestyle='--')
        
        # 校准曲线的绘制
        for self.__cm, self.__name in self.__list:
            
            self.__column_1.append(self.__name)
            
            self.__liner = LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
            self.__y, self.__yscore = self.__cm.cc()
            
            # Brier Score
            self.__brierscore = brier_score_loss(self.__y, self.__yscore, pos_label=self.__y.max())
            self.__brier = "%1.3f" % self.__brierscore
            self.__column_4.append(self.__brier)
            
            # Calibration Curve
            self.__observed, self.__predicted = calibration_curve(self.__y, self.__yscore, n_bins=self.__bins[self.__p])
            self.__liner.fit(self.__predicted.reshape(-1,1), self.__observed.reshape(-1,1))
            self.__intercept = "%1.2f" % self.__liner.intercept_
            self.__slope = "%1.2f" % self.__liner.coef_
            self.__column_2.append(self.__intercept)
            self.__column_3.append(self.__slope)
            self.__liner = None
            
            self.__ax1.plot(self.__predicted, self.__observed, "s-", color=self.__colorlist[self.__colorindex], 
                            label="%s (Brier Score: %1.3f)" % (self.__name, self.__brierscore))
            
            if self.__D:
                self.__ax2.hist(self.__yscore, range=(0, 1), bins=self.__bins[self.__p], 
                                color=self.__colorlist[self.__colorindex], label=self.__name, histtype="step", lw=2)
                
            self.__colorindex = self.__colorindex + 1
            self.__p = self.__p + 1
                
        self.__colorindex = None
        
        self.__ax1.set_ylabel(self.__ylabel_c)
        self.__ax1.set_ylim([-0.05, 1.05])
        self.__ax1.legend(loc="lower right")
        self.__ax1.set_title(self.__title)
        
        if self.__D:
            self.__ax2.set_xlabel(self.__xlabel)
            self.__ax2.set_ylabel(self.__ylabel_d)
            self.__ax2.legend(loc="upper center", ncol=2)
        else:
            self.__ax1.set_xlabel(self.__xlabel)
            
        plt.tight_layout()
        self.__CCPlot = plt.gcf()
        plt.show()
        
        self.__dictionary = {'Model Name':self.__column_1, 'Intercept':self.__column_2, 
                             'Slope':self.__column_3, 'Brier Score':self.__column_4}
        self.__frame = pd.DataFrame(self.__dictionary)
        display(self.__frame)
        print("★注意：bins参数会影响斜率和截距的值")
        
        self.__column_1 = None
        self.__column_2 = None
        self.__column_3 = None
        self.__column_4 = None
        
        self.__plotstate = True
        
        return None
    
    def save(self, path="CC.png", dpi=300):
        """
        save()方法用于保存绘制的CC曲线
        
        参数介绍
        ========
        
        path：保存路径，类型为str，格式为“/file1/file2/image_name.png”；默认保存在当前路径，名字为"CC.png"
        
        dpi：每英寸上，所能印刷的网点数，类型为int，默认为300
        
        ========
        """
        
        if self.__plotstate:
            self.__path = path
            self.__dpi = dpi
            
            self.__CCPlot.savefig(self.__path, dpi=self.__dpi, bbox_inches = 'tight')
            print("Calibration Curve已保存，路径为：%s" % self.__path)
        else:
            raise RuntimeError("尚未绘制Calibration Curve，请先调用plot()方法")
            
        return None
"""========================================================="""

"""================== NNCalibrationModel ==================="""
class NNCalibrationModel:
    """
    NNCalibrationModel用于保存概率校准后的神经网络模型的预测概率，可作为伪模型,并配合ConfusionMatrix使用
    
    ========
    参数说明
    ========
    
    pattern：模式，类型为str，只能为'save'或'load'，默认为'save'，即保存模式，用于第一次创建，第二次用load模式即可
    
    model：用CalibratedClassifierCV进行概率校准后的KerasClassifier模型，即要保存概率的模型
    
    x_data：预测的数据集，类型为DataFrame，默认为None
    
    file_path：保存概率的文件夹名，save模式必须设置，多个概率可设置同样的名字，类型为str，默认为“NetworkCalibration”
    
    path：概率的储存地址，类型为str，注意最后格式一定以“.csv”结尾，并注意与file_path文件名一致
    
    ★：模式为“save”时，model、x_data、file_path、以及path为必须参数；
    ★：模式为“load”时，path为必须参数，其余参数无用；
    
    ========
    """
    
    def __init__(self, pattern='save', model=None, x_data=None, file_path='NetworkCalibration', path='NetworkCalibration/Model.csv'):
        
        "参数检查"
        # pattern
        if type(pattern) == type('save'):
            if pattern == 'save':
                self.__pattern = pattern
            else:
                if pattern == 'load':
                    self.__pattern = pattern
                else:
                    raise ValueError("pattern参数只能为'save'或'load'")
        else:
            raise ValueError("pattern的类型必须为str，而传入的类型为%s" % type(pattern))
        
        # model
        if pattern == 'save':
            if type(model) == type(None):
                raise ValueError("缺少model参数，请传入model参数")
            else:
                self.__model = model
            
        # x_data
        if pattern == 'save':
            if type(x_data) == type(None):
                raise ValueError("缺少x_data参数，请传入x_data参数")
            else: 
                if type(x_data) != type(pd.DataFrame()):
                    raise ValueError("x_data的类型必须为DataFrame，而传入的类型为%s" % type(x_data))
                else:
                    self.__x = x_data
        
        # file_path
        if self.__pattern == 'save':
            if type(file_path) == type("NetworkCalibration"):
                self.__file_path = file_path
            else:
                raise ValueError("file_path的类型必须为str，而传入的类型为%s" % type(file_path))
                
        # path
        if type(path) == type("NetworkCalibration/Model.csv"):
            self.__path = path
        else:
            raise ValueError("path的类型必须为str，而传入的类型为%s" % type(path))
            
        "功能实现"
        # 概率的保存
        if self.__pattern == 'save':
            
            # 创建文件夹
            self.__folder = os.path.exists(self.__file_path)
            if not self.__folder:
                os.makedirs(self.__file_path)
                
            # 保存概率，以Keras模型的形式
            self.__pred_prob = self.__model.predict_proba(self.__x)[:, 1].reshape(-1,1)
            self.__probability = self.__model.predict_proba(self.__x)[:, 1].reshape(-1,1) # ★核心变量
            self.__data = pd.DataFrame(self.__pred_prob)
            self.__data.to_csv(self.__path, encoding='utf-8', index = False)
            print("概率已保存，并成功创建NNCalibrationModel类的伪模型")
        
        elif self.__pattern == 'load':
            
            # 读取概率
            self.__pred_prob = pd.read_csv(self.__path)
            self.__probability = self.__pred_prob.to_numpy() # ★核心变量
            print("概率读取成功，并成功创建NNCalibrationModel类的伪模型")
            
        else:
            raise RuntimeError("运行错误请检查！")
        
        return None
    
    def predict(self, x_data=None):
        """predict()方法，为NNCalibrationModel类的伪预测方法，直接调用无意义"""
        
        # x_data
        if type(x_data) == type(None):
            self.__predictstates = False
        else: 
            if type(x_data) != type(pd.DataFrame()):
                self.__predictstates = False
            else:
                self.__predictstates = True
        
        assert self.__predictstates, "代码有Bug，请联系包装者检查"
        self.__p = None
        self.__p = copy.deepcopy(self.__probability)
        return self.__p
"""========================================================="""

"""================== Plot_Learning_Curve =================="""
def Plot_Learning_Curve(model, x, y, cv=None, ylim=None, n_jobs=-1, train_sizes=np.linspace(0.1,1.0,5), title='Learning Curve', xlabel='Training examples', ylabel='Score', tlabel='Training', vlabel='Cross Validation'):
    
    plt.figure(dpi=300)
    
    plt.title(title) # 图像标题
    if ylim is not None: # y轴限制不为空时
        plt.ylim(*ylim)
    plt.xlabel(xlabel) # 两个标题
    plt.ylabel(ylabel)
    
    train_sizes,train_scores,test_scores=learning_curve(model, x, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes) # 获取训练集大小，训练得分集合，测试得分集合
    train_scores_mean=np.mean(train_scores, axis=1)#将训练得分集合按行的到平均值
    train_scores_std=np.std(train_scores, axis=1)#计算训练矩阵的标准方差
    test_scores_mean=np.mean(test_scores, axis=1)
    test_scores_std=np.std(test_scores, axis=1)
    plt.grid()#背景设置为网格线
    
    plt.fill_between(train_sizes,train_scores_mean-train_scores_std,train_scores_mean+train_scores_std,alpha=0.1,color='#ff6d6d')
    # plt.fill_between()函数会把模型准确性的平均值的上下方差的空间里用颜色填充。
    plt.fill_between(train_sizes,test_scores_mean-test_scores_std,test_scores_mean+test_scores_std,alpha=0.1,color='#79a4ea')
    plt.plot(train_sizes,train_scores_mean,'o-',color='#ff6d6d',label=tlabel)
    # 然后用plt.plot()函数画出模型准确性的平均值
    plt.plot(train_sizes,test_scores_mean,'o-',color='#79a4ea',label=vlabel)
    plt.legend(loc='best')#显示图例
    
    return None
"""========================================================="""

"""===================== Bootstrap_PRC ====================="""
class Bootstrap_PRC:
    """
     Bootstrap_PRC用于AUPRC显著性检验，需要配合ConfusionMatrix使用（7.1及其以上）
    
    参数介绍
    ========
    CM_Name_List：是待检验模型的ConfusionMatrix和模型命名字符串的组合列表,类型为list，默认为None，格式为↓
                  
                  CM_Name_List = [(cm_lr, 'LR'),(cm_svm, 'SVM'), (cm_dnn,'DNN')]
                  
    times：自助抽样的次数，类型为int，默认为3000
    
    random_seed：自助抽样的随机数种子，类型为int，默认为None，即完全随机,建议设置
    ========
    
    方法介绍
    ========
    
    test()方法：执行自助抽样，并输出结果
    
    ========
    
    """
    
    def __init__(self, CM_Name_List=None, times=3000, random_seed=None):
        
        # 参数检查
        # CM_Name_List
        if type(CM_Name_List) == type(None):
            raise ValueError("缺少CM_Name_List参数")
        elif type(CM_Name_List) != type([]):
            raise ValueError("CM_Name_List参数的类型应该为list，而传入的类型为%s" % type(CM_Name_List))
        else:
            self.__list = CM_Name_List
            
        # times
        if type(times) == type(3000):
            self.__times = times
            if self.__times <1000:
                warnings.warn("注意：自助抽样的次数小于1000，为：%s" & slef.__times)
        else:
            raise ValueError("times参数的类型应该为int，而传入的类型为%s" % type(times))
            
        # random_seed
        if type(random_seed) == type(None):
            self.__seed = random_seed
        elif type(random_seed) == type(43):
            self.__seed = random_seed
        else:
            raise ValueError("random_seed参数的类型应该为int或None，而传入的类型为%s" % type(random_seed))
            
        # 参数转换
        # 测试列表
        self.__testlist = []
        for self.__cm, self.__name in self.__list:
            self.__y_true, self.__y_pred, self.__auprc, self.__prcmod, self.__prcci = self.__cm.bootstrap_prc()
            self.__newtuple = (self.__y_true, self.__y_pred, self.__auprc, self.__prcmod, self.__name, self.__prcci)
            self.__testlist.append(self.__newtuple)
        self.__y_true = None
        self.__y_pred = None
        self.__auprc = None
        self.__prcmod = None
        
        # 随机种子numpy arry
        np.random.seed(self.__seed)
        self.__seedarry = np.random.randint(low=0,high=999,size=self.__times)
        
        return None
    
    def test(self, alpha=0.95):
        """
        test()方法用于启动Bootstrap法
        
        参数介绍
        ========
        alpha：置信水平，类型为float，默认为0.95，即95%的置信区间
        
        """
        
        assert type(alpha) == type(0.95),"alpha参数错误，请检查！"
        assert alpha < 1,"alpha参数错误，请检查！"
        self.__alpha = alpha
        self.__k1 = int(self.__times * (1-self.__alpha) /2)
        self.__k2 = int(self.__times *(1- (1-self.__alpha) /2))
        self.__outputlist = []
        
        #两两配对检验
        for self.__r1 in range(0,len(self.__testlist)-1):
            for self.__r2 in range(self.__r1+1, len(self.__testlist)):
                # AUPRC大小确定
                self.__name_a = self.__testlist[self.__r1][4]
                self.__name_b = self.__testlist[self.__r2][4]
                self.__Name = (self.__name_a, self.__name_b)
                self.__value_a = self.__testlist[self.__r1][2]
                self.__value_b = self.__testlist[self.__r2][2]
                self.__AUPRC = (self.__value_a, self.__value_b)
                self.__ci_a = self.__testlist[self.__r1][5]
                self.__ci_b = self.__testlist[self.__r2][5]
                self.__PRCCI = (self.__ci_a, self.__ci_b)
                if self.__value_a > self.__value_b:
                    self.__y_true_1, self.__y_pred_1, self.__auprc_1, self.__prcmod_1, self.__name_1, self.__prcci_1 = self.__testlist[self.__r1]
                    self.__y_true_2, self.__y_pred_2, self.__auprc_2, self.__prcmod_2, self.__name_2, self.__prcci_2 = self.__testlist[self.__r2]
                else:
                    self.__y_true_1, self.__y_pred_1, self.__auprc_1, self.__prcmod_1, self.__name_1, self.__prcci_1 = self.__testlist[self.__r2]
                    self.__y_true_2, self.__y_pred_2, self.__auprc_2, self.__prcmod_2, self.__name_2, self.__prcci_2 = self.__testlist[self.__r1]
                
                # 参数检验
                if self.__prcmod_1 != self.__prcmod_2:
                    warnings.warn("警告：%s与%s两模型计算AUPRC的方式不一样" % (self.__name_1, self.__name_2))
                assert len(self.__y_true_1) == len(self.__y_true_2),"真实标签错误，请检查！"
                
                # 参数转化/声明
                self.__difference = self.__auprc_1 - self.__auprc_2
                self.__sample = len(self.__y_true_1)
                self.__result = []
                
                # 自助抽样
                for self.__i in  range(self.__times):
                    np.random.seed(self.__seedarry[self.__i])
                    self.__sampleindex = np.random.randint(low=0,high=self.__sample,size=self.__sample)
                    self.__y_true_a, self.__y_pred_a = self.__y_true_1[self.__sampleindex], self.__y_pred_1[self.__sampleindex]
                    self.__y_true_b, self.__y_pred_b = self.__y_true_2[self.__sampleindex], self.__y_pred_2[self.__sampleindex]
                    if self.__prcmod_1 ==1:
                        self.__precision_a, self.__recall_a, self.__thresholds_a = precision_recall_curve(self.__y_true_a,self.__y_pred_a)
                        self.__precision_b, self.__recall_b, self.__thresholds_b = precision_recall_curve(self.__y_true_b,self.__y_pred_b)
                        self.__prc_a = auc(self.__recall_a, self.__precision_a)
                        self.__prc_b = auc(self.__recall_b, self.__precision_b)
                    elif self.__prcmod_1 ==2:
                        self.__iprauc_a = AUC(curve='PR')
                        self.__iprauc_a.update_state(self.__y_true_a,self.__y_pred_a)
                        self.__prc_a =  np.float64(self.__iprauc_a.interpolate_pr_auc().numpy())
                        self.__iprauc_b = AUC(curve='PR')
                        self.__iprauc_b.update_state(self.__y_true_b,self.__y_pred_b)
                        self.__prc_b =  np.float64(self.__iprauc_b.interpolate_pr_auc().numpy())
                    else:
                        assert False, "代码有Bug，请联系包装者检查"
                    self.__result.append(self.__prc_a - self.__prc_b)
                self.__result_sorted = sorted(self.__result)
                self.__lower = self.__result_sorted[self.__k1]
                self.__higher = self.__result_sorted[self.__k2]
                self.__CI = (self.__lower, self.__higher)
                self.__outputlist.append((self.__Name,self.__prcmod_1,self.__AUPRC,self.__PRCCI,self.__difference,self.__CI))
                
        # 输出结果
        print("The Result of Bootstrap：")
        for self.__op in range(0, len(self.__outputlist)):
            print(" ")
            print("============================")
            print("【%s & %s】" % (self.__outputlist[self.__op][0][0], self.__outputlist[self.__op][0][1]))
            if self.__outputlist[self.__op][1] == 1:
                print("AUPRC Claculated by Trapezoidal Method")
            else:
                print("AUPRC Claculated by Interpolate Method")
            print(" AUPRC of %s：" % self.__outputlist[self.__op][0][0], 
                  "%.3f" % self.__outputlist[self.__op][2][0], 
                  "(%.3f-%.3f)" % (self.__outputlist[self.__op][3][0][0], self.__outputlist[self.__op][3][0][1]))
            print(" AUPRC of %s：" % self.__outputlist[self.__op][0][1], 
                  "%.3f" % self.__outputlist[self.__op][2][1], 
                  "(%.3f-%.3f)" % (self.__outputlist[self.__op][3][1][0], self.__outputlist[self.__op][3][1][1]))
            print(" Difference (Bootstrap CI)：%.3f (%.3f-%.3f)" % (self.__outputlist[self.__op][4], self.__outputlist[self.__op][5][0], self.__outputlist[self.__op][5][1]))
            if self.__outputlist[self.__op][5][0]<0:
                print(" P-Value>0.05")
            else:
                print(" P-Value<0.05")
            print("============================")
            
        return None
"""========================================================="""