import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split


class Tensor:
    def __init__(self, value, parents=(), params=[], name="simple", need_grad=False):
        self.value = np.array(value)
        self.parents = parents
        self.params = params
        self.grad = 0.0
        self.backprop_fn = lambda: None
        self.name = name
        self.need_grad = need_grad

    def __add__(self, other):
        value = self.value + other.value
        parents = (self, other)
        z = Tensor(value, parents, name="fromAddition")

        def backward():
            self.grad += 1.0 * z.grad
            other.grad += 1.0 * z.grad

        z.backprop_fn = backward
        return z

    def __mul__(self, other):
        value = self.value * other.value
        parents = (self, other)
        z = Tensor(value, parents, name="fromMultiplication")

        def backward():
            self.grad += other.value * z.grad
            other.grad += self.value * z.grad

        z.backprop_fn = backward
        return z

    def __sub__(self, other):
        return self + Tensor(-1) * other

    def __pow__(self, power):
        value = self.value ** power
        parents = (self,)
        z = Tensor(value, parents, name="fromPower")

        def backward():
            self.grad += power * self.value ** (power - 1) * z.grad

        z.backprop_fn = backward
        return z

    def __matmul__(self, other):
        value = np.matmul(self.value, other.value)
        parents = (self, other)
        z = Tensor(value, parents, name="fromMatMul")

        def backward():
            if isinstance(z.grad, float):
                z.grad = np.array([[z.grad]])
            self.grad += np.matmul(z.grad, other.value.T)
            other.grad += np.matmul(self.value.T, z.grad)

        z.backprop_fn = backward
        return z

    def transpose(self):
        value = np.transpose(self.value)
        parents = (self,)
        z = Tensor(value, parents, name="fromTranspose")

        def backward():
            self.grad += np.transpose(z.grad)

        z.backprop_fn = backward
        return z

    def trace(self):
        value = np.trace(self.value)
        parents = (self,)
        z = Tensor(value, parents, name="fromTrace")

        def backward():
            self.grad += np.identity(self.value.shape[0]) * z.grad

        z.backprop_fn = backward
        return z

    @staticmethod
    def log(self):
        value = np.log(self.value)
        parents = (self,)
        z = Tensor(value, parents, name="fromLog")

        def backward():
            self.grad += 1 / self.value * z.grad

        z.backprop_fn = backward
        return z

    @staticmethod
    def exp(self):
        value = np.exp(self.value)
        parents = (self,)
        z = Tensor(value, parents, name="fromExp")

        def backward():
            self.grad += np.exp(self) * z.grad

        z.backprop_fn = backward
        return z

    @staticmethod
    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.value))
        parents = (self,)
        z = Tensor(sig, parents, name="fromSigmoid")

        def backward():
            self.grad += sig * (1 - sig) * z.grad

        z.backprop_fn = backward
        return z

    @staticmethod
    def tanh(self):
        tanh = np.tanh(self.value)
        parents = (self,)
        z = Tensor(tanh, parents, name="fromTanh")

        def backward():
            self.grad += (1 - tanh ** 2) * z.grad

        z.backprop_fn = backward
        return z

    @staticmethod
    def relu(self):
        value = np.maximum(0, self.value)
        parents = (self,)
        z = Tensor(value, parents, name="fromReLU")

        def backward():
            self.grad += np.where(self.value > 0, 1, 0) * z.grad

        z.backprop_fn = backward
        return z

    def backprop(self):
        visited = set()
        sorted = []

        def sort(tensor):
            if tensor not in visited:
                visited.add(tensor)
                sorted.append(tensor)
                for ten in tensor.parents:
                    sort(ten)

        sort(self)
        self.grad = 1.0
        for tensor in sorted:
            if tensor.need_grad:
                self.params.append(tensor)
            tensor.backprop_fn()

    def update(self, lr=0.001):
        for tensor in self.params:
            tensor.value -= lr * tensor.grad

    def zero_init(self):
        for tensor in self.params:
            tensor.grad = 0

    def __str__(self):
        return f"name: {self.name}, value shape: {self.value.shape}, grad: {self.grad.shape}, parents: {self.parents}, params: {self.params}"


class Linear:
    def __init__(self, input_size, output_size, once=True):
        self.weights = Tensor(np.random.randn(input_size, output_size), need_grad=True)
        self.bias = Tensor(np.random.randn(output_size), need_grad=True)
        self.once = once

    def __call__(self, x):
        if self.once:
            self.bias.value = np.tile(self.bias.value, (x.value.shape[0], 1))
            self.once = False
        return x @ self.weights + self.bias


class LinearRegression:
    def __init__(self, act_func, input_size, hidden_size, output_size=1):
        layers = []
        hidden_size = [input_size] + hidden_size + [output_size]
        for i in range(len(hidden_size) - 1):
            layers.append(Linear(hidden_size[i], hidden_size[i + 1]))
            if i >= len(hidden_size) - 2:
                continue
            layers.append(act_func[i])
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def mse(predictions, targets):
    return (predictions - targets).transpose() @ (predictions - targets)


def bce(predictions, targets):
    one = Tensor(np.ones(targets.value.shape))
    return (Tensor(-1) * (targets @ Tensor.log(predictions).transpose() + (one - targets) @ Tensor.log(
        one - predictions).transpose())).trace()


def train(model, loss_func, X_train, y_train, learning_rate=0.01, epochs=100):
    for epoch in range(epochs):
        pred = model.forward(X_train)
        loss = loss_func(pred, y_train)
        loss.backprop()
        loss.update(learning_rate)
        loss.zero_init()

        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.value}")


diabetes_data = load_diabetes()
X_diabetes_train, X_diabetes_test, y_diabetes_train, y_diabetes_test = train_test_split(diabetes_data.data,
                                                                                        diabetes_data.target,
                                                                                        test_size=0.2, random_state=42)
X_diabetes_train = Tensor(X_diabetes_train)
X_diabetes_test = Tensor(X_diabetes_test)
y_min = y_diabetes_train.min()
y_max = y_diabetes_train.max()
y_diabetes_train = Tensor((((y_diabetes_train - y_min) / (y_max - y_min))).reshape(-1, 1))
y_diabetes_test = Tensor(y_diabetes_test.reshape(-1, 1))

cancer_data = load_breast_cancer()
X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(cancer_data.data, cancer_data.target,
                                                                                test_size=0.2, random_state=42)
X_cancer_train = Tensor(X_cancer_train)
X_cancer_test = Tensor(X_cancer_test)
y_cancer_train = Tensor(y_cancer_train.reshape(-1, 1))
y_cancer_test = Tensor(y_cancer_test.reshape(-1, 1))

np.random.seed(42)
act_func_linear = [Tensor.sigmoid]
hidden_size = [16]
model_linear = LinearRegression(act_func_linear, X_diabetes_train.value.shape[1], hidden_size)
train(model_linear, mse, X_diabetes_train, y_diabetes_train, learning_rate=0.0001, epochs=500)
"""
For logistic regression same but with the loss that is defined in "bce" function
"""
