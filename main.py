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

# Load the Excel file using pandas
station_df = pd.read_excel('线路条件数据.xlsx', sheet_name='station')
curve_df = pd.read_excel('线路条件数据.xlsx', sheet_name='curve')

static_limit_x = []
static_limit_y = []
def plot_static_limit_curve():
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

    static_limit_points: List[Tuple[float, float]] = [(0, max_speed)]
    x = 0
    for segment in static_limit_segments:
        p1, p2 = segment
        if x < p1[0]:
            static_limit_points.append((x, max_speed))
            static_limit_points.append((p1[0], max_speed))
        static_limit_points.append(p1)
        static_limit_points.append(p2)
        x = p2[0]
    if x < max_distance:
        static_limit_points.append((x, max_speed))
        static_limit_points.append((max_distance, max_speed))

    static_limit_points_x = [point[0] for point in static_limit_points]
    static_limit_points_y = [point[1] for point in static_limit_points]
    interp_func = interp1d(static_limit_points_x, static_limit_points_y, kind='linear')
    global static_limit_x, static_limit_y
    static_limit_x = np.linspace(0, max_distance, max_distance * 100)
    static_limit_y = interp_func(static_limit_x)

ebi_x, ebi_y = [], []
def plot_ebi():
    global static_limit_x, static_limit_y
    global ebi_x, ebi_y
    x, y = static_limit_x.copy(), static_limit_y.copy()
    delta_x = x[1] - x[0]
    for i in range(2, len(y)+1):
        safev = np.sqrt(2 * 紧急制动率 * delta_x + y[-i + 1] ** 2)
        y[-i] = min(y[-i], safev)

    ebi_x, ebi_y = x, y


if __name__ == '__main__':
    plot_static_limit_curve()
    plt.plot(static_limit_x, static_limit_y, 'b-')

    plot_ebi()
    plt.plot(ebi_x, ebi_y, 'r-')
    plt.show()
