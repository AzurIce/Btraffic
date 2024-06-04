import pandas as pd
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from scipy.interpolate import interp1d

from typing import List, Tuple

启动加速度 = 0.8 # m/s^2
牵引加速度 = 0.5 # m/s^2
紧急制动率 = 1.2 # m/s^2
常用制动率 = 0.5 # m/s^2
车长 = 80 # m
制动建立时延 = 0.7 # s
牵引切断延时 = 0.7 # s
旋转质量系数 = 1.08
ATP余量 = 3 # km/h
ATO余量 = 5 # km/h
max_distance = 24000 # m
max_speed = 87 # km/h
g = 9.81
m = 800 * 10 ** 3 # kg

x_step = 1
X = np.linspace(0, max_distance, int(max_distance / x_step)).tolist()

# Load the Excel file using pandas
FILE_NAME = '线路条件数据.xlsx'
station_df = pd.read_excel(FILE_NAME, sheet_name='station')
curve_df = pd.read_excel(FILE_NAME, sheet_name='curve')
grad_df = pd.read_excel(FILE_NAME, sheet_name='grad')

def grad_a(grad, k):
    # grad_degree = np.arctan(grad / 1000)
    # return g * np.sin(grad_degree) / k
    return grad / 1000 / k

def sign(x):
    return 1 if x == 1 else -1

def calc_grad_a() -> List[float]:
    grad_a_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for i in range(len(grad_df.index)):
        x1, y1 = grad_df.iloc[i, 0], grad_df.iloc[i, 2] * grad_a(sign(grad_df.iloc[i, 3]), 旋转质量系数)
        x2, y2 = grad_df.iloc[i, 1], grad_df.iloc[i, 2] * grad_a(sign(grad_df.iloc[i, 3]), 旋转质量系数)
        grad_a_segments.append(((x1, y1), (x2, y2)))
    grad_a_segments = sorted(grad_a_segments, key=lambda x: x[0][0])

    points: List[Tuple[float, float]] = [(0, 0)]
    x = 0
    for segment in grad_a_segments:
        p1, p2 = segment
        if x < p1[0]:
            points.append((x, 0))
            points.append((p1[0], 0))
        points.append(p1)
        points.append(p2)
        x = p2[0]
    if x < max_distance:
        points.append((x, 0))
        points.append((max_distance, 0))

    points_x = [point[0] for point in points]
    points_y = [point[1] for point in points]
    interp_func = interp1d(points_x, points_y, kind='linear')

    y = interp_func(X)
    return y

def calc_static_limit() -> List[float]:
    static_limit_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for i in range(len(station_df.index)):
        x1, y1 = station_df.iloc[i, 0], station_df.iloc[i, 2]
        x2, y2 = station_df.iloc[i, 1], station_df.iloc[i, 2]
        static_limit_segments.append(((x1, y1), (x2, y2)))
    for i in range(len(curve_df.index)):
        x1, y1 = curve_df.iloc[i, 0], curve_df.iloc[i, 7]
        x2, y2 = curve_df.iloc[i, 1], curve_df.iloc[i, 7]
        static_limit_segments.append(((x1, y1), (x2, y2)))
    static_limit_segments = sorted(static_limit_segments, key=lambda x: x[0][0])

    points: List[Tuple[float, float]] = [(0, max_speed)]
    x = 0
    for segment in static_limit_segments:
        p1, p2 = segment
        if x < p1[0]:
            points.append((x, max_speed))
            points.append((p1[0], max_speed))
        points.append(p1)
        points.append(p2)
        x = p2[0]
    if x < max_distance:
        points.append((x, max_speed))
        points.append((max_distance, max_speed))

    points_x = [point[0] for point in points]
    points_y = [point[1] for point in points]
    interp_func = interp1d(points_x, points_y, kind='linear')

    y = interp_func(X)
    return y

def calc_safev(static_limit_y, a, redundant=0) -> List[float]:
    y = [y - redundant for y in static_limit_y]
    for i in range(2, len(y)+1):
        safev = np.sqrt(2 * a * x_step + y[-i + 1] ** 2)
        y[-i] = min(y[-i], safev)

    return y

def calc_safev_with_extra_a(static_limit_y, extra_a, a, redundant=0) -> List[float]:
    y = [y - redundant for y in static_limit_y]
    for i in range(2, len(y)+1):
        safev = np.sqrt(max(2 * (a + extra_a[-i]) * x_step + y[-i + 1] ** 2, 0))
        y[-i] = min(y[-i], safev)

    return y

def calc_v(sbi_y, extra_a) -> List[float]:
    v: List[float] = [0]
    for i in range(len(X) - 1):
        a = 启动加速度 if v[i] < 40 * 10 ** 3 else 牵引加速度
        newv = np.sqrt(2 * (a + extra_a[i]) * x_step + v[i] ** 2)
        if newv > sbi_y[i]:
            newv = sbi_y[i]
        v.append(newv)
    return v

if __name__ == '__main__':
    x = X
    static_limit_y = calc_static_limit()

    extra_a = calc_grad_a()
    # plt.plot(x, extra_a)
    # print(extra_a)
    # ebi_x, ebi_y = calc_safev(x, static_limit_y, 紧急制动率)
    # sbi_x, sbi_y = calc_safev(x, static_limit_y, 常用制动率)
    # plt.plot(x, static_limit_y, 'b-')
    # plt.plot(ebi_x, ebi_y, 'r-')
    # plt.plot(sbi_x, sbi_y, 'g-')
    # plt.show()

    ebi_y = calc_safev_with_extra_a(static_limit_y, extra_a, 紧急制动率)
    sbi_y = calc_safev_with_extra_a(static_limit_y, extra_a, 常用制动率, ATO余量)
    v_y = calc_v(sbi_y, extra_a)
    # plt.plot(x, static_limit_y, 'b-')
    # plt.plot(ebi_x, ebi_y, 'r-')
    # plt.plot(sbi_x, sbi_y, 'g-')
    # plt.show()
    fig = go.Figure(data=[
        go.Scatter(x=x, y=static_limit_y, name='静态限速'),
        go.Scatter(x=x, y=ebi_y, name='EBI'),
        go.Scatter(x=x, y=sbi_y, name='SBI'),
        go.Scatter(x=x, y=v_y, name='列车运行速度')
    ])
    fig.update_layout(
        title='作业罢了',
        xaxis_title='Distance (m)',
        yaxis_title='Speed (km/h)',
        xaxis=dict(
            rangeslider=dict(
                visible=True
            )
        )
    )
    fig.show()
