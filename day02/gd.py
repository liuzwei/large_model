import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数
def f(x):
    return (x-3)**2

# 定义梯度下降函数
def df(x):
    return 2*(x-3)

# 梯度下降算法
def gradient_descent(start_point, learning_rate, num_iterations):
    x = start_point
    # 记录每一步的x值
    history = [x]
    for _ in range(num_iterations):
        #  计算当前的梯度
        grad = df(x)
        #  更新x值
        x = x - learning_rate * grad
        #  记录每一步的x值
        history.append(x)
    return x, history
# 初始设置
start_point = 0.0
learning_rate = 0.2
num_iterations = 30

# 执行梯度下降算法
result, history = gradient_descent(start_point, learning_rate, num_iterations)
print(f"最小值点: {result}")

# 绘制结果
x_vals = np.linspace(-1, 7, 60000)
y_vals = f(x_vals)

plt.plot(x_vals, y_vals, label=r'$f(x) = (x-3)^2$', color='blue')
plt.scatter(history, f(np.array(history)), color='red', label='gradient descent', zorder=5)
#  给点标上序号
for i, x in enumerate(history):
    plt.annotate(str(i), (x, f(x)), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')
# plt.plot(history, f(np.array(history)), color='green', label='gradient descent', zorder=5)
plt.title('Gradient Descent 2D')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()