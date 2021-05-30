"""
构建一个决策树类别
quan xueliang
2021.5
"""

class Node:
    """
    目标是构建CART树，每个非叶子节点只有两个子节点，构建节点类
    """
    def __init__(self, data_index, logger=None, split_feature=None, split_value=None, is_leaf=False, loss=None, deep=None):
        self.loss = loss  # 节点分裂带来的损失，理解为信息增益，实际节点划分时，使loss最小
        self.split_feature = split_feature  # split_feature是特征维度的索引
        self.split_value = split_value  # 节点依据某个特征进行分裂时，需要一个分裂的取值
        self.data_index = data_index  # 节点样本子集的索引
        self.is_leaf = is_leaf  # 该节点是否为叶子节点
        self.predict_value = None  # 如果是叶子节点，对样本的预测值
        self.left_child = None  # 如果是非叶子节点，有左右两个子树
        self.right_child = None
        self.logger = logger  # logger是存储的信息文件
        self.deep = deep  # 该节点的深度

    def update_predict_value(self, targets, y):
        self.predict_value = self.loss.update_leaf_values(targets, y)
        self.logger.info(('叶子节点预测值', self.predict_value))


    def get_predict_value(self, instance):
        """ 在对样本进行预测时，需要递归的调用函数本身，树模型正常都会使用到递归"""

        # 递归终止条件
        if self.is_leaf:  # 如果是叶子节点，返回该节点的预测值
            self.logger.info(('predict:', self.predict_value))
            return self.predict_value

        # 非递归终止条件下，递归的调用函数本身，进行样本的预测
        if instance[self.split_feature] < self.split_value:
            # 小于分裂值，划分到左子树
            return self.left_child.get_predict_value(instance)
        else:
            return self.right_child.get_predict_value(instance)


class Tree:
    """构建树模型"""
    def __init__(self, data, max_depth, min_samples_split, features, loss, target_name, logger):
        self.loss = loss  # 这个loss应该就是根据数据集建立的树模型的信息熵或者信息增益
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.features = features  # 当前节点数据中包含的特征
        self.logger = logger
        self.target_name = target_name
        self.remain_index = [True] * len(data)
        self.leaf_nodes = []
        self.root_node = self.build_tree(data, self.remain_index, depth=0)


    def build_tree(self, data, remain_index, depth=0):
        """
        树继续生长的条件（节点继续分裂的条件）：
        1· 树的深度没有达到最大，树的深度介入是3，这里的depth只能是0，1，
            所以判断条件是 depth < self.max_depth - 1
        2· 节点的样本数 >= min_samples_split
        3· 该节点是的样本的 target_name 值不一样（如果值一样，说明已经划分的很好了，不需要再分）
        周志华 西瓜书 节点停止分裂的条件：
        1· 当前节点包含的样本都是一个类别的，不用在继续划分，沿着当前路径即可做出决策
        2· 当前节点的属性集为空，或者节点内样本在所有属性上的取值全部相同，这种条件下无法继续划分
        3· 当前节点内包含的样本集合为空，不能再继续划分
        """
        now_data = data[remain_index]  # 剩余的样本

        """ 树的生长条件1，2，3, 节点分裂条件需要3条都满足，但是节点停止分裂的条件满足其中的一条就可以了"""
        if depth < self.max_depth - 1 \
                and len(now_data) >= self.min_samples_split \
                and len(now_data[self.target_name].unique()) > 1:   # 样本标签不统一
            # unique() 统计所有出现的元素
            se = None
            split_feature = None
            split_value = None
            left_index_of_now_data = None
            right_index_of_now_data = None
            self.logger.info(('--树的深度： %d' % depth))
            # 选择用于分裂的特征
            for feature in self.features:
                self.logger.info(('----划分特征:', feature))
                feature_values = now_data[feature].unique()
                # 选择特征用于分裂的取值
                for fea_value in feature_values:
                    # 尝试划分
                    # 左子树的样本集在数据集中的索引
                    left_index = list(now_data[feature] < fea_value)
                    right_index = list(now_data[feature] >= fea_value)
                    # 基于左子树的样本集的标签计算label的方差
                    left_se = calculate_se(now_data[left_index][self.target_name])
                    right_se = calculate_se(now_data[right_index][self.target_name])
                    # 该节点左子树和右子树的label的方差的和，依据最小的方差选择节点特征的分裂值
                    # 这里并没有使用到信息熵或者基尼指数计算数据集的信息增益
                    sum_se = left_se + right_se
                    self.logger.info(('----划分值：%.3f, 左子树损失：%.3f, 右子树损失：%.3f, 总损失：%.3f'%
                                      (fea_value, left_se, right_se, sum_se)))
                    # 选择最小的损失对应的取值为特征的分裂值
                    if se is None or sum_se < se:
                        split_feature = feature
                        split_value = fea_value
                        se = sum_se
                        left_index_of_now_data = left_index
                        right_index_of_now_data = right_index
            self.logger.info(('--最佳划分特征：', split_feature))
            self.logger.info(('--最佳划分值：', split_value))

            """ 选择好划分属性和划分值之后，就可以建立树的当前节点，并对子节点继续进行树的生成 """
            node = Node(remain_index, self.logger, split_feature, split_value, deep=depth)
            """
            trick for dataForm, index revert
            下面这部分代码是为了记录划分后样本在原始数据中的索引
            dataForm的数据索引可以使用True和False
            所以下面得到的是一个bool型的元组组成的数组
            利用这个数组进行索引获得划分后的数据
            """
            left_index_of_all_data = []
            for i in remain_index:
                if i:
                    if left_index_of_now_data[0]:
                        left_index_of_all_data.append(True)
                        del left_index_of_now_data[0]
                    else:
                        left_index_of_all_data.append(False)
                        del left_index_of_now_data[0]
                else:
                    left_index_of_all_data.append(False)

            right_index_of_all_data = []
            for i in remain_index:
                if i:
                    if right_index_of_now_data[0]:
                        right_index_of_all_data.append(True)
                        del right_index_of_now_data[0]
                    else:
                        right_index_of_all_data.append(False)
                        del right_index_of_now_data[0]
                else:
                    right_index_of_all_data.append(False)

            """ 递归调用生成树模型本身，进行树的生成， 树模型一定是需要使用递归的 """
            # 生成左子树
            node.left_child = self.build_tree(data, left_index_of_all_data, depth+1)
            node.right_child = self.build_tree(data, right_index_of_all_data, depth+1)
            return node

        # 满足节点停止分裂条件的任意一条，即停止分裂, 生成叶子节点
        else:
            node = Node(remain_index, self.logger, is_leaf=True, loss=self.loss, deep=depth)
            if len(self.target_name.split('_')) == 3:
                label_name = 'label_' + self.target_name.split('_')[1]
            else:
                label_name = 'label'
            node.update_predict_value(now_data[self.target_name], now_data[label_name])
            self.leaf_nodes.append(node)
            return node


def calculate_se(label):
    """ 计算标签的方差"""
    mean = label.mean()  # 标签的均值
    se = 0
    for y in label:  # 计算label 的方差
        se += (y - mean) * (y - mean)
    return se


