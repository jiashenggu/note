import numpy as np

# 激活函数基类
class Activation:
    def forward(self, x):
        pass
    
    def backward(self, grad_output):
        pass

# ReLU激活函数
class ReLU(Activation):
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        return grad_output * (self.input > 0)

# Linear激活函数
class Linear(Activation):
    def forward(self, x):
        return x
    
    def backward(self, grad_output):
        return grad_output

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01):
        # 初始化权重和偏置
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros(output_dim)
        self.lr = lr
        
        # 初始化激活函数
        self.activation1 = ReLU()
        self.activation2 = Linear()
        
    def forward(self, X):
        self.X = X
        # 第一层
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation1.forward(self.z1)
        # 第二层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.activation2.forward(self.z2)
        # softmax
        self.output = self.softmax(self.a2)
        return self.output
    
    def backward(self, y):
        m = y.shape[0]
        
        # 输出层梯度
        dz2 = self.output - y  # softmax的导数
        da2 = self.activation2.backward(dz2)
        dW2 = np.dot(self.a1.T, da2) / m
        db2 = np.sum(da2, axis=0) / m
        
        # 隐藏层梯度
        da1 = np.dot(da2, self.W2.T)
        dz1 = self.activation1.backward(da1)
        dW1 = np.dot(self.X.T, dz1) / m
        db1 = np.sum(dz1, axis=0) / m
        
        self.gradients = {
            'W1': dW1, 'b1': db1,
            'W2': dW2, 'b2': db2
        }
        
    def update(self):
        self.W1 -= self.lr * self.gradients['W1']
        self.b1 -= self.lr * self.gradients['b1']
        self.W2 -= self.lr * self.gradients['W2']
        self.b2 -= self.lr * self.gradients['b2']
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# 测试
if __name__ == "__main__":
    # 生成随机数据
    np.random.seed(0)
    X = np.random.randn(100, 2)
    y = np.array([1 if x1 + x2 > 0 else 0 for x1, x2 in X])
    y_onehot = np.zeros((y.shape[0], 2))
    y_onehot[np.arange(y.shape[0]), y] = 1
    
    # 初始化并训练MLP
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=2)
    
    # 训练1000轮
    for i in range(1000):
        pred = mlp.forward(X)
        mlp.backward(y_onehot)
        mlp.update()
        
        if i % 100 == 0:
            loss = -np.mean(np.sum(y_onehot * np.log(pred + 1e-8), axis=1))
            print(f"Iteration {i}, Loss: {loss:.4f}")
    
    # 预测并计算准确率
    pred = np.argmax(mlp.forward(X), axis=1)
    accuracy = np.mean(pred == y)
    print(f"\nAccuracy: {accuracy:.4f}")
