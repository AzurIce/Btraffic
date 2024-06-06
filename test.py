import numpy as np

# 示例数据
x = np.arange(24000)  # 位移，假设单位是千米
y = np.array([1 / 3.6 for i in range(24000)])
# y = np.random.random(24000) * 100 + 1  # 速度，假设单位是千米每小时 (km/h)，并确保速度不为零

# 确保速度不为零，可以将速度为0的值替换为一个很小的值，例如1e-6
y[y == 0] = 1e-6

def calculate_time(x, y, x1, x2):
    # 确保 x1 和 x2 在 x 的范围内
    if x1 < x[0] or x2 > x[-1]:
        raise ValueError("x1 和 x2 应在 x 的范围内")

    # 确定 x1 和 x2 对应的索引
    idx1 = np.searchsorted(x, x1)
    idx2 = np.searchsorted(x, x2)

    # 位移和速度的截取区间
    x_segment = x[idx1:idx2+1]
    y_segment = y[idx1:idx2+1]

    # 使用梯形积分法计算时间
    time = np.trapz(1 / y_segment, x_segment)

    return time

# 示例计算
x1 = 1000
x2 = 10000
time = calculate_time(x, y, x1, x2)
print(x1, x2, time, time/3600)
