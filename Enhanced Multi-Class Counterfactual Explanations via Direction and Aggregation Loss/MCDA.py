import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import shap
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import datetime
from typing import List, Optional
import torch.nn.functional as F
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
import time
from carla import log
from carla.recourse_methods.processing import reconstruct_encoding_constraints
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# 人造数据集
# np.random.seed(42)
#
# # 样本数量
# num_samples = 10000
#
# # 生成特征1和特征2的数据，值在0到100之间
# feature_1 = np.random.uniform(0, 100, num_samples)
# feature_2 = np.random.uniform(0, 100, num_samples)
#
# # 生成分类标签，值在1到10之间
# labels = np.random.randint(1, 9, num_samples)
#
# # 创建DataFrame
# data = pd.DataFrame({
#     'Feature_1': feature_1,
#     'Feature_2': feature_2,
#     'Label': labels
# })
#定义多层感知机模型MLP
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPModel, self).__init__()
        # 定义两层网络：隐藏层和输出层
        self.hidden1 = nn.Linear(input_size, hidden_size)

        # 定义第二层隐藏层
        self.hidden2 = nn.Linear(hidden_size, hidden_size // 2)  # 可选择缩小神经元数量
        self.hidden3 = nn.Linear(hidden_size//2, hidden_size // 4)
        # 定义输出层
        self.output = nn.Linear(hidden_size // 4, num_classes)



    def forward(self, x):
        # 使用第一隐藏层
        x = torch.relu(self.hidden1(x))
        # 使用第二隐藏层
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        # 输出层返回 logits
        # return self.output(x)
        #输出归一化分数
        return F.softmax(self.output(x), dim=1)

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)



# 定义逻辑回归模型
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size,num_calsses): #第一个参数特征数量，第二个参数几分类问题
        super(LogisticRegressionModel, self).__init__()
        # 线性层（input_size -> 1，输出是概率）
        self.linear = nn.Linear(input_size, num_calsses)

    def forward(self, x):
        # 使用 Sigmoid 函数将线性输出转换为概率
        return torch.softmax(self.linear(x),dim=1)

    def forward1(self, x):
        return self.linear(x)

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            predicted = torch.argmax(logits, dim=1)
        return predicted

# 求类别中心
def heart(data :pd.DataFrame):
    data = data.values
    center = np.mean(data,axis=0)
    # print(f"质心坐标:{center}")
    return center

# 数据集的加载及处理过程
data = pd.read_csv('path')
data = data.fillna(data.mean())  #处理缺失值
label_encoder = LabelEncoder()
data['class'] = label_encoder.fit_transform(data['class'])     #对分类特征进行排序，从0开始
scaler = MinMaxScaler()
x = data.iloc[:]  #选取数据集中的样本
x = scaler.fit_transform(x)  #对样本进行标准化
y = data.iloc[:]  #目标类别
#训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,random_state=42)


def calculate_centroid(data, label_column, label_value, feature_columns):
    """
    计算特定类别的数据的质心

    Parameters:
    - data: 数据框 (DataFrame)
    - label_column: 样本类别列的索引 (整数)
    - label_value: 需要选择的类别的值
    - feature_columns: 用于计算质心的样本特征列范围 (列表)

    Returns:
    - 该类别数据的质心
    """
    # 选择特定类别的数据
    data_x = data[data.iloc[:, label_column] == label_value]

    # 计算该类别数据的质心
    return scaler.transform(heart(data_x.iloc[:, feature_columns]))
class1 = calculate_centroid(data, label_column=81, label_value=0, feature_columns=range(1, 81))
class2 = calculate_centroid(data, label_column=81, label_value=1, feature_columns=range(1, 81))
class3 = calculate_centroid(data, label_column=81, label_value=2, feature_columns=range(1, 81))
class4 = calculate_centroid(data, label_column=81, label_value=3, feature_columns=range(1, 81))
class5 = calculate_centroid(data, label_column=81, label_value=4, feature_columns=range(1, 81))
class6 = calculate_centroid(data, label_column=81, label_value=5, feature_columns=range(1, 81))
class7 = calculate_centroid(data, label_column=81, label_value=6, feature_columns=range(1, 81))
class8 = calculate_centroid(data, label_column=81, label_value=7, feature_columns=range(1, 81))

pointToBeTested = x[410]
# 评估模型的函数
def evaluate_model(model, x_train, y_train, x_test, y_test, num_epochs=260, learning_rate=0.01):
    # 定义损失函数（交叉熵）和优化器（Adam）
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    y_train = y_train.long()
    y_test = y_test.long()

    # 训练模型
    for epoch in range(num_epochs):

        y_pred = model(x_train)


        loss = criterion(y_pred, y_train)


        optimizer.zero_grad()


        loss.backward()


        optimizer.step()

    # 计算训练集和测试集的精度
    with torch.no_grad():

        train_pred = model.predict(x_train)
        train_accuracy = (train_pred == y_train).float().mean().item()


        test_pred = model.predict(x_test)
        test_accuracy = (test_pred == y_test).float().mean().item()

    return train_accuracy, test_accuracy

# 定义一个选择最优模型的函数
def select_best_model(models, x_train, y_train, x_test, y_test, num_epochs=260, learning_rate=0.01):
    best_train_accuracy = 0
    best_test_accuracy = 0
    best_model = None

    for model in models:
        print(f"Training model: {model.__class__.__name__}")
        train_accuracy, test_accuracy = evaluate_model(model, x_train, y_train, x_test, y_test, num_epochs, learning_rate)
        print(f"Train Accuracy: {train_accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%")

        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_train_accuracy = train_accuracy
            best_model = model

    print(f"\nBest Model: {best_model.__class__.__name__}")
    print(f"Best Train Accuracy: {best_train_accuracy * 100:.2f}%, Best Test Accuracy: {best_test_accuracy * 100:.2f}%")
    return best_model

model1 = LogisticRegressionModel(input_size=80,num_calsses=8)

model2 = MLPModel(input_size=80, hidden_size=256, num_classes=8)
models = [model1, model2]
# 使用选择最优模型的函数
model = select_best_model(models, x_train, y_train, x_test, y_test)


def find_boundary_points(model, x, y, probability_range=(0.5, 0.95)):
    """
    计算边界点：模型预测正确且预测概率在指定范围内的样本

    参数:
    - model: 已训练的模型
    - x: 输入数据
    - y: 真实标签
    - probability_range: 预测概率的有效范围（默认为 [0.5, 0.95]）

    返回:
    - boundary_points: 一个包含边界点索引的数组
    """
    boundary_points = np.zeros(len(x), dtype=int)
    for i in range(len(x)):
        factual = torch.tensor([x[i]], dtype=torch.float32)
        out = model(factual)
        maxindex = out.argmax().item()
        if y[i] == maxindex and probability_range[0] <= out[0][maxindex] <= probability_range[1]:
            boundary_points[i] = i
    boundary_points = boundary_points[boundary_points != 0]
    return boundary_points


def calculate_shap_importance(model, x_train, x_test,target_category):
    """
    计算 SHAP 特征重要性并进行归一化

    参数:
    - model: 已训练的模型
    - x_train: 训练数据
    - x_test: 测试数据
     -target_category: 目标类别

    返回:
    - shap_nor: 归一化后的特征重要性
    """
    explainer = shap.DeepExplainer(model, x_train)
    shap_values = explainer.shap_values(x_test)
    shap1 = np.sum(shap_values[0], axis=target_category)
    shap_sum = np.sum(np.abs(shap1))
    shap_nor = shap1 / shap_sum
    print("Normalized SHAP Importance: ")
    print(shap_nor)

    return shap_nor


input_data = torch.tensor([pointToBeTested], dtype=torch.float32)

output = model(input_data)
# 打印模型输出和形状
print(f"Model Output: {output}")
print(f"Output Shape: {output.shape}")
start_time = time.time()  # 记录开始时间


def y_loss(a,model1):
    loss_fn = torch.nn.CrossEntropyLoss()
    logits = model1.forward1(a)
    y_loss = loss_fn(logits, torch.tensor([0]))
    return y_loss


def distance_loss(a):
    b = torch.tensor([pointToBeTested], dtype=torch.float32)
    l1_norm = torch.abs(a - b).sum()
    return l1_norm


def direct_loss(b):
    a = torch.tensor(class1, dtype=torch.float32)
    a = a.squeeze()
    b = b.squeeze()
    dot_product = torch.dot(a, b)
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    cos_value = dot_product / (norm_a * norm_b)
    return 1 - cos_value


def agger_loss(b):
    a1 = torch.tensor(class1, dtype=torch.float32)
    a2 = torch.tensor(class2, dtype=torch.float32)
    a3 = torch.tensor(class3, dtype=torch.float32)
    a4 = torch.tensor(class4, dtype=torch.float32)
    a5 = torch.tensor(class5, dtype=torch.float32)
    a6 = torch.tensor(class6, dtype=torch.float32)
    a7 = torch.tensor(class7, dtype=torch.float32)
    a8 = torch.tensor(class8, dtype=torch.float32)
    l1 = torch.abs(a1 - b).sum()
    l2 = torch.abs(a2 - b).sum()
    l3 = torch.abs(a3 - b).sum()
    l4 = torch.abs(a4 - b).sum()
    l5 = torch.abs(a5 - b).sum()
    l6 = torch.abs(a6 - b).sum()
    l7 = torch.abs(a7 - b).sum()
    l8 = torch.abs(a8 - b).sum()
    return l1/(l2+l6+l3+l4+l7+l8+l5)



def loss1(a,model1):
    f_loss = y_loss(a,model1)
    dis_loss = distance_loss(a)
    cost_loss = direct_loss(a)
    kaojin_loss =agger_loss(a)
    return 1*f_loss+0.5*dis_loss+0.5*cost_loss+1*kaojin_loss


discreteFeatures = []   #离散特征位置
immutableFeatures = [] #不可变特征
x_new = Variable(input_data.clone(), requires_grad = True)
optimizer2 = optim.Adam([x_new],0.001,amsgrad=True)
addFeatures = []
reduceFeatures = []
it = 0
it_max = 500

#自定义学习率更新类
class CustomLRScheduler:
    def __init__(self, optimizer, initial_lr=0.01, increase_factor=1.20,decrease_factor = 0.9):
        self.optimizer = optimizer
        self.lr = initial_lr
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor

    def step(self):
        # 每次更新后按 increase_factor 增大学习率
        self.lr *= self.increase_factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
    def stop(self):
        self.lr *= self.decrease_factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr


scheduler = CustomLRScheduler(optimizer2, initial_lr=0.01, increase_factor=1.10,decrease_factor = 0.9)


#预测目标是否满足类别
def f_y(model,a):
    b = a.clone()
    out  = model(b)
    maxindex = out.argmax().item()
    if maxindex == 0:
        return False
    else:
        return True


P_pre = model(x_new)[:, 0]
P_now = model(x_new)[:, 0]

while f_y(model,x_new)and it < it_max:
    while f_y(model,x_new) and it < it_max:
        optimizer2.zero_grad()
        loss = loss1(x_new,model)
        loss.backward()
        min_abs_grad, min_abs_index = x_new.grad[0].abs().min(dim=0)
        x_new.grad[0][min_abs_index] = 0.0
        #不可变约束
        with torch.no_grad():
            for i in immutableFeatures:
                x_new[:, i] = 0.0
        #只增特征
        with torch.no_grad():
            for i in addFeatures:
                if x_new[:, i] >=0:
                    x_new[:, i] = 0.0
       #只减特征
        with torch.no_grad():
            for i in addFeatures:
                if x_new[:, i]<=0:
                    x_new[:, i] = 0.0
        optimizer2.step()

    # 投影处理，确保所有值在 [0, 1] 范围内
        torch.clamp(x_new,0,1)
        it += 1
        #学习率更新判断
        if (it % 5) == 0:
            P_pre = P_now
            P_now = model(x_new)[:, 0]
            if (P_now - P_pre) < 0.005:
                scheduler.step()
        # print(f"模型预测{model(x_new)}")
    #离散约束
    with torch.no_grad():
        for i in discreteFeatures:
            x_new[:, i] = torch.round(x_new[:, i])
end_time = time.time()  # 记录结束时间
print("运行时间:", end_time - start_time, "秒")
array2 = x_new.detach().numpy()
row_50_true = scaler.inverse_transform(pointToBeTested.reshape(1, -1))
x_new_true = scaler.inverse_transform(array2)
#对离散特征进行处理以计算距离
for i in discreteFeatures:
    if row_50_true[0][i]>x_new_true[0][i]:
        row_50_true[0][i] = 1/80
    elif row_50_true[0][i]<x_new_true[0][i]:
        x_new_true[0][i] = 1/80
L1 = np.sum(np.abs(row_50_true - x_new_true))
print(L1)
print(f"迭代次数{it}")
distance2 = np.linalg.norm(row_50_true - x_new_true)
print(f"L2距离为{distance2}")


def loss2(a, model1, f_loss_weight=1.0, dis_loss_weight=0.5, cost_loss_weight=0.5, kaojin_loss_weight=1.0):
    f_loss = y_loss(a, model1)
    dis_loss = distance_loss(a)
    cost_loss = direct_loss(a)
    kaojin_loss = agger_loss(a)
    total_loss = f_loss_weight * f_loss + dis_loss_weight * dis_loss + cost_loss_weight * cost_loss + kaojin_loss_weight * kaojin_loss
    return total_loss

#参数分析
def parameter_analysis(model, input_data, lisanfactual, max_iterations=500, initial_lr=0.01):
    t1 = time.time()
    result = []
    resultL2 = []
    resultIT = []
    resultTime = []
    dis_loss_weights = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for dis_loss_weight in dis_loss_weights:
        x_new = Variable(input_data.clone(), requires_grad=True)
        optimizer = optim.Adam([x_new], lr=initial_lr, amsgrad=True)
        it = 0
        scheduler = CustomLRScheduler(optimizer, initial_lr=initial_lr, increase_factor=1.1, decrease_factor=0.9)

        f_y1_pre = model(x_new)[:, 0]
        f_y1_now = model(x_new)[:, 0]
        while f_y(model, x_new) and it < max_iterations:
            while f_y(model, x_new) and it < it_max:
                optimizer.zero_grad()
                loss = loss2(x_new, model, 1.0, 0.5,0.5,dis_loss_weight)
                loss.backward()

                min_abs_grad, min_abs_index = x_new.grad[0].abs().min(dim=0)
                x_new.grad[0][min_abs_index] = 0.0
                optimizer.step()

                torch.clamp(x_new, 0, 1)  # 确保所有值在[0, 1]范围内

                it += 1
                if (it % 5) == 0:
                    f_y1_pre = f_y1_now
                    f_y1_now = model(x_new)[:, 0]
                    if (f_y1_now - f_y1_pre) < 0.005:
                        scheduler.step()  # 学习率衰减

            with torch.no_grad():
                for i in lisanfactual:
                    x_new[:, i] = torch.round(x_new[:, i])
        array2 = x_new.detach().numpy()
        t2 = time.time()
        row_50_true = scaler.inverse_transform(pointToBeTested.reshape(1, -1))
        x_new_true = scaler.inverse_transform(array2)
        L1 = np.sum(np.abs(row_50_true - x_new_true))
        L2 = np.linalg.norm(row_50_true - x_new_true)
        result.append(L1)
        resultL2.append(L2)
        resultIT.append(it)
        resultTime.append(t2-t1)