import pandas as pd
import matplotlib.pyplot as plt
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

# Load the Excel file using pandas
FILE_NAME = '线路条件数据.xlsx'
station_df = pd.read_excel(FILE_NAME, sheet_name='station')
curve_df = pd.read_excel(FILE_NAME, sheet_name='curve')
grad_df = pd.read_excel(FILE_NAME, sheet_name='grad')

def grad_a(grad, rmc):
    grad_degree = np.arctan(grad / 1000)
    return g * np.sin(grad_degree) - rmc * g * np.cos(grad_degree) / m

def sign(x):
    return 1 if x == 1 else -1

def calc_grad_a() -> Tuple[List[float], List[float]]:
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

    x = np.linspace(0, max_distance, max_distance * 100).tolist()
    y = interp_func(x)
    return x, y

def calc_static_limit() -> Tuple[List[float], List[float]]:
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

    x = np.linspace(0, max_distance, max_distance * 100).tolist()
    y = interp_func(x)
    return x, y

def calc_safev(static_limit_x, static_limit_y, a) -> Tuple[List[float], List[float]]:
    x, y = static_limit_x.copy(), static_limit_y.copy()
    delta_x = x[1] - x[0]
    for i in range(2, len(y)+1):
        safev = np.sqrt(2 * a * delta_x + y[-i + 1] ** 2)
        y[-i] = min(y[-i], safev)

    return x, y

def calc_safev_with_extra_a(static_limit_x, static_limit_y, extra_a, a) -> Tuple[List[float], List[float]]:
    x, y = static_limit_x.copy(), static_limit_y.copy()
    delta_x = x[1] - x[0]
    for i in range(2, len(y)+1):
        safev = np.sqrt(max(2 * (a + extra_a[-i]) * delta_x + y[-i + 1] ** 2, 0))
        y[-i] = min(y[-i], safev)

    return x, y

if __name__ == '__main__':
    x, static_limit_y = calc_static_limit()

    _, extra_a = calc_grad_a()
    # plt.plot(x, extra_a)
    # print(extra_a)
    # ebi_x, ebi_y = calc_safev(x, static_limit_y, 紧急制动率)
    # sbi_x, sbi_y = calc_safev(x, static_limit_y, 常用制动率)
    # plt.plot(x, static_limit_y, 'b-')
    # plt.plot(ebi_x, ebi_y, 'r-')
    # plt.plot(sbi_x, sbi_y, 'g-')
    # plt.show()

    ebi_x, ebi_y = calc_safev_with_extra_a(x, static_limit_y, extra_a, 紧急制动率)
    sbi_x, sbi_y = calc_safev_with_extra_a(x, static_limit_y, extra_a, 常用制动率)
    plt.plot(x, static_limit_y, 'b-')
    plt.plot(ebi_x, ebi_y, 'r-')
    plt.plot(sbi_x, sbi_y, 'g-')
    plt.show()
