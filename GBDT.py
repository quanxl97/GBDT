"""
GBDT
regression, binary classify, multi calssify
quan xueliang
2021.5
"""

import abc
import math
import logging  # 记录日志
import pandas as pd
from decisionTree import Tree
from lossFunction import SequaresError, BinomialDeviance, MultinomialDeviance
from treePlot import plot_multi, plot_tree, plot_all_trees
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



class AbstractBaseGradientBoosting(metaclass=abc.ABCMeta):
    " 定义抽象类 "
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


class BaseGradientBoosting(AbstractBaseGradientBoosting):
    " GBDT 的基本类 "
    def __init__(self, loss, learning_rate, n_trees, max_depth, min_samples_split=2
                 , is_log=False, is_plot=False):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.features = None
        self.is_log = is_log
        self.is_plot = is_plot
        self.trees = {}  # 存放GBDT模型的所有树
        self.f_0 = {}  # 存放初始化的基模型 f_0

    def fit(self, data):
        """

        :param data: pandas.DataFrame, the features data of training
        :return:
        """
        # 删除id和label，得到特征名称
        self.features = list(data.columns)[1:-1]
        # 初始化 f_0
        # 对于平方损失来说，初始化 f_0(x) 就是 y 的均值
        self.f_0 = self.loss.initialize_f_0(data)
        # 对于 m = 1, 2, ..., M, 得到的基模型为：
        for iter in range(1, self.n_trees+1):
            # 保存模型日志
            if len(logger.handlers) > 1:
                logger.removeHandler(logger.handlers[-1])
            fh = logging.FileHandler('results/N0.{}_tree.log'.format(iter), mode='w', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
            # 计算负梯度信息--对于平方误差来说就是残差
            logger.info(('-------------------构建第%d棵树---------------------' % iter))
            self.loss.calculate_residual(data, iter)
            target_name = 'res_' + str(iter)
            # 生成GBDT模型的第iter棵树
            self.trees[iter] = Tree(data, self.max_depth, self.min_samples_split,
                                    self.features, self.loss, target_name, logger)
            # 更新集成树模型的损失函数,这个learning_rate 是直接给出的
            self.loss.update_f_m(data, self.trees, iter, self.learning_rate, logger)
            # 根据需要，画出当前迭代得到的基模型
            if self.is_plot:
                plot_tree(self.trees[iter], max_depth=self.max_depth, iter=iter)
        # print(self.trees)
        # 打印所有的树模型
        if self.is_plot:
            plot_all_trees(self.n_trees)


class GradientBoostingRegressor(BaseGradientBoosting):
    """ 回归问题的 GBDT 模型 """
    def __init__(self, learning_rate, n_trees, max_depth,
                 min_samples_split=2, is_log=False, is_plot=False):
        super().__init__(
            SequaresError(), learning_rate, n_trees, max_depth,
            min_samples_split, is_log, is_plot
        )

    def predict(self, data):
        data['f_0'] = self.f_0
        for iter in range(1, self.n_trees+1):
            f_prev_name = 'f_' + str(iter-1)
            f_m_name = 'f_' + str(iter)
            data[f_m_name] = data[f_prev_name] + \
                             self.learning_rate * \
                             data.apply(lambda x: self.trees[iter].root_node.get_predict_value(x), axis=1)
        # 回归问题，获取预测值
        data['predict_value'] = data[f_m_name]


class GradientBoostingBinaryClassifier(BaseGradientBoosting):
    def __init__(self, learning_rate, n_trees, max_depth,
                 min_samples_split=2, is_log=False, is_plot=False):
        super(GradientBoostingBinaryClassifier, self).__init__(
            BinomialDeviance(), learning_rate, n_trees, max_depth,
            min_samples_split, is_log, is_plot
        )

    def predict(self, data):
        " 预测 "
        data['f_0'] = self.f_0
        for iter in range(1, self.n_trees +1):
            f_m_name = 'f_' + str(iter)
            f_prev_name = 'f_' + str(iter-1)
            data[f_m_name] = data[f_prev_name] + \
                             self.learning_rate * \
                             data.apply(lambda x: self.trees[iter].root_node.get_predict_value(x), axis=1)
        # 输出预测的概率，引入指数函数
        data['predict_proba'] = data[f_m_name].apply(lambda x: 1 / (1 + math.exp(-x)))
        # 将上面的概率问题转化成分类问题
        data['predict_label'] = data['predict_proba'].apply(lambda x: 1 if x >= 0.5 else 0)



class GradientBoostingMultiClassifier(BaseGradientBoosting):
    """
    构建多分类问题的GBDT模型，每一轮迭代中，每个类别构建一个决策树
    """
    def __init__(self, learning_rate, n_trees, max_depth,
                 min_samples_split=2, is_log=False, is_plot=False):
        super(GradientBoostingMultiClassifier, self).__init__(
            MultinomialDeviance(), learning_rate, n_trees, max_depth,
            min_samples_split, is_log, is_plot
        )

    def fit(self, data):
        # 删去id和label，获取特征名称
        self.features = list(data.columns)[1:-1]
        # 统计所有类别
        self.classes = data['label'].unique().astype(str)
        # 初始化多分类问题的的损失函数 K
        self.loss.init_classes(self.classes)
        # 根据 label 做 one-hot 处理
        for class_name in self.classes:
            label_name = 'label_' + class_name
            data[label_name] = data['label'].apply(lambda x: 1 if str(x) == class_name else 0)
            # 初始化 f_0(x)
            self.f_0[class_name] = self.loss.initialize_f_0(data, class_name)
        # print(data)
        # 对 m=1, 2, 3, ..., M
        logger.handlers[0].setLevel(logging.INFO if self.is_log else logging.CRITICAL)
        for iter in range(1, self.n_trees + 1):
            if len(logger.handlers) > 1:
                logger.removeHandler(logger.handlers[-1])
            fh = logging.FileHandler('results/N0.{}_tree.log'.format(iter), mode='w', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            logger.addHandler(fh)
            logger.info(('---------------------构建第%d棵树--------------------' % iter))
            # 这里计算负梯度是为了计算p_sum的一致性
            self.loss.calculate_residual(data, iter)
            self.trees[iter] = {}
            for class_name in self.classes:
                # 每次迭代每棵树要拟合的残差值
                target_name = 'res_' + class_name + '_' + str(iter)
                # 训练该轮次中该类别对应的决策树
                self.trees[iter][class_name] = Tree(data, self.max_depth, self.min_samples_split,
                                                    self.features, self.loss, target_name, logger)
                self.loss.update_f_m(data, self.trees, iter, class_name, self.learning_rate, logger)
            if self.is_plot:
                plot_multi(self.trees[iter], max_depth=self.max_depth, iter=iter)
        if self.is_plot:
            plot_all_trees(self.n_trees)

    def predict(self, data):
        """
        次数的预测方式和生成树的方式不同。
        生成树是需要每个类别的树的每次迭代一起进行，外层循环是iter, 内层循环是class。
        而在预测时，树已经生成，可以将class这层作为外循环，进而节省运算成本。
        :param data:
        :return:
        """
        for class_name in self.classes:
            f_0_name = 'f_' + class_name + '_0'
            data[f_0_name] = self.f_0[class_name]
            for iter in range(1, self.n_trees+1):
                f_m_name = 'f_' + class_name + '_' + str(iter)
                f_prev_name = 'f_' + class_name + '_' + str(iter-1)
                data[f_m_name] = data[f_prev_name] + \
                                    self.learning_rate * data.apply(lambda x:
                                    self.trees[iter][class_name].root_node.get_predict_value(x), axis=1)
        data['sum_exp'] = data.apply(lambda x:
                          sum([math.exp(x['f_'+ i +'_'+str(iter)]) for i in self.classes]), axis=1)

        for class_name in self.classes:
            proba_name = 'predict_proba_' + class_name
            f_m_name = 'f_' + class_name + '_' + str(iter)
            data[proba_name] = data.apply(lambda x: math.exp(x[f_m_name]) / x['sum_exp'], axis=1)
        # TODO: log, 每一类的概率
        data['predict_label'] = data.apply(lambda x: self._get_multi_label(x), axis=1)

    def _get_multi_label(self, x):
        label = None
        max_proba = -1
        for class_name in self.classes:
            if x['predict_proba_' + class_name] > max_proba:
                max_proba = x['predict_proba_' + class_name]
                label = class_name
        return label


