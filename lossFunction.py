"""
计算损失函数
loss for regression, binary classify, multi calssify
quan xueliang
2021.5
"""

import math
import abc
# abc 是一个抽象基类模块，主要用于实现 （1）某种情况下，判定某个对象的类型， （2）强制子类必须实现某些方法，即ABC的派生类


class LossFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def initialize_f_0(self, data):
        """ 初始化 F_0 """

    @abc.abstractmethod
    def calculate_residual(self, data, iter):
        """ 计算负梯度 """

    @abc.abstractmethod
    def update_f_m(self, data, trees, iter, learning_rate, logger):
        """ 计算 F_m """

    @abc.abstractmethod
    def update_leaf_values(self, targets, y):
        """ 更新叶子节点的预测值 """

    @abc.abstractmethod
    def get_train_loss(self, y, f, iter, logger):
        """ 计算训练损失 """

# 上面损失函数的继承类
class SequaresError(LossFunction):
    " 平方损失函数，用于回归问题 "

    def initialize_f_0(self, data):
        """ 初始化 F_0 """
        # 计算标签的均值，用于学习
        data['f_0'] = data['label'].mean()
        return data['label'].mean()


    def calculate_residual(self, data, iter):
        """
        计算负梯度
        iter: 迭代次数，树的个数
        """
        res_name = 'res_' + str(iter)
        f_prev_name = 'f_' + str(iter - 1)
        data[res_name] = data['label'] - data[f_prev_name]


    def update_f_m(self, data, trees, iter, learning_rate, logger):
        """ 计算 F_m """
        f_prev_name = 'f_' + str(iter-1)  # 上一棵树
        f_m_name = 'f_' + str(iter)  # 当前树
        data[f_m_name] = data[f_prev_name]
        # 遍历前一棵树的叶子节点
        for leaf_node in trees[iter].leaf_nodes:
            data.loc[leaf_node.data_index, f_m_name] += learning_rate * leaf_node.predict_value
        # 打印每一棵树的 train loss
        self.get_train_loss(data['label'], data[f_m_name], iter, logger)


    def update_leaf_values(self, targets, y):
        """ 更新叶子节点的预测值 """
        return targets.mean()


    def get_train_loss(self, y, f, iter, logger):
        """
        计算训练损失
        :param y: 真实标签
        :param f: 预测标签
        :param iter: 迭代次数
        :param logger: 模型日志
        :return: None
        """
        loss = ((y-f)**2).mean()  # 损失是平方误差的均值
        logger.info(('第%d棵树： mse_loss: %.4f' % (iter, loss)))




class BinomialDeviance(LossFunction):
    " 二分类场景下损失计算及模型更新 "
    def initialize_f_0(self, data):
        pos = data['label'].sum()  # 正类样本的数量
        neg = data.shape[0] - pos  # 负类样本的数量
        # 这里计算正负样本数量用了一个小技巧

        # 这里的log是以e为底，即 ln
        f_0 = math.log(pos/neg)
        data['f_0'] = f_0
        return f_0

    def calculate_residual(self, data, iter):
        # 计算残差，也就是负梯度信息
        res_name = 'res_' + str(iter)  # 第m轮迭代产生的残差进行命名
        f_prev_name = 'f_' + str(iter-1)
        # 利用m轮的预测值和m-1轮的预测输出计算当前迭代轮次的负梯度信息，用于下一步优化m轮次的基模型
        data[res_name] = data['label'] - 1 / (1+data[f_prev_name].apply(lambda x: math.exp(-x)))


    def update_f_m(self, data, trees, iter, learning_rate, logger):
        # 优化更新得到m轮次的基模型
        f_m_name = 'f_' + str(iter)
        f_prev_name = 'f_' + str(iter-1)
        data[f_m_name] = data[f_prev_name]
        # 遍历当前轮次的决策树的所有叶子节点
        for leaf_node in trees[iter].leaf_nodes:
            data.loc[leaf_node.data_index, f_m_name] += learning_rate * leaf_node.predict_value
        # 打印每棵树的训练损失,存储在训练日志中
        self.get_train_loss(data['label'], data[f_m_name], iter, logger)


    def update_leaf_values(self, targets, y):

        # 更新叶子节点的预测值，注意此处是二分类问题
        numerator = targets.sum()
        if numerator == 0:
            return 0.0
        denominator = ((y-targets)*(1-y+targets)).sum()
        if abs(denominator) < 1e-150:
            return 0.0
        else:
            return numerator / denominator


    def get_train_loss(self, y, f, iter, logger):
        """
        计算训练损失
        :param y: 真实标签
        :param f: 预测标签
        :param iter: 当前迭代轮次
        :param logger: 模型训练日志
        :return:
        """
        # 指数损失函数
        loss = -2.0*((y*f)-f.apply(lambda x: math.exp(1+x))).mean()
        logger.info(('第%d棵树： log-likelihood:%.4f' % (iter, loss)))



class MultinomialDeviance():
    """  注意，多分类的损失计算 不是直接继承的 LossFunction 类。
    GBDT在进行多分类任务时，在每一个轮次的迭代训练时，每一轮要训练c棵决策树，
    是一对多的模型结构。"""
    def init_classes(self, classes):
        self.classes = classes

    @abc.abstractmethod
    # 以下函数是从LossFunction类继承过来的
    def initialize_f_0(self, data, class_name):
        " 初始化基模型 f_0, f_0 有 C 棵决策树 "
        label_name = 'label_' + class_name
        f_name = 'f_' + class_name + '_0'
        class_counts = data[label_name].sum()  # 统计类别数量 C
        f_0 = class_counts / len(data)
        data[f_name] = f_0
        return f_0


    def calculate_residual(self, data, iter):
        " 计算负梯度信息 "
        data['sum_exp'] = data.apply(lambda x:
                                     sum([math.exp(x['f_' + i + '_' + str(iter-1)]) for i in self.classes]),
                                     axis=1)
        # 这里计算的是每个类别的负梯度信息
        for class_name in self.classes:
            label_name = 'label_' + class_name
            res_name = 'res_' + class_name + '_' + str(iter)
            f_prev_name = 'f_' + class_name + '_' + str(iter-1)
            data[res_name] = data[label_name] - math.e ** data[f_prev_name] / data['sum_exp']



    def update_f_m(self, data, trees, iter, class_name, learning_rate, logger):
        f_m_name = 'f_' + class_name + '_' + str(iter)
        f_prev_name = 'f_' + class_name + '_' + str(iter-1)
        data[f_m_name] = data[f_prev_name]
        # 遍历每一个叶子节点，这里和论文中是一致的，即模型的更新是对每一个叶子节点的更新
        for leaf_node in trees[iter][class_name].leaf_nodes:
            data.loc[leaf_node.data_index, f_m_name] += learning_rate * leaf_node.predict_value
        # 将每一棵树的训练损失存储到训练日志中
        self.get_train_loss(data['label'], data[f_m_name], iter, logger)


    def update_leaf_values(self, targets, y):
        # 这里没有完全明白
        numerator = targets.sum()
        if numerator == 0:
            return 0.0
        numerator *= (self.classes.size-1) / self.classes.size
        denominator = ((y-targets)*(1-y+targets)).sum()
        if abs(denominator) < 1e-150:
            return 0.0
        else:
            return numerator / denominator


    def get_train_loss(self, y, f, iter, logger):
        " 计算训练损失，并保存到模型的训练日志中 "
        loss = -2.0*((y*f) - f.apply(lambda x: math.exp(1+x))).mean()
        logger.info(('第%d棵树：log-likelihood:%.4f' % (iter, loss)))



