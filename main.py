import pandas as pd
# import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d

from typing import Callable, List, Sequence, Tuple

def grad_a(grad, k):
    # grad_degree = np.arctan(grad / 1000)
    # return g * np.sin(grad_degree) / k
    return grad / 100 / k

def sign(x):
    return 1 if x == 1 else -1

def read_segments(
    df: pd.DataFrame,
    get_x1: Callable[[pd.DataFrame, int], float],
    get_x2: Callable[[pd.DataFrame, int], float],
    get_y: Callable[[pd.DataFrame, int], float],
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    '''read segments from df'''
    segments = []
    for i in range(len(df.index)):
        x1, x2 = get_x1(df, i), get_x2(df, i)
        y = get_y(df, i)
        segments.append(((x1, y), (x2, y)))
    segments = sorted(segments, key=lambda x: x[0][0])
    return segments


def segments_to_points(
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    default_y: float=0
) -> List[float]:
    '''convert segments to points'''
    points: List[Tuple[float, float]] = [(0, default_y)]
    x = 0
    for segment in segments:
        p1, p2 = segment
        if x < p1[0]:
            points.append((x, default_y))
            points.append((p1[0], default_y))
        points.append(p1)
        points.append(p2)
        x = p2[0]
    if x < max_distance:
        points.append((x, default_y))
        points.append((max_distance, default_y))

    points_x = [point[0] for point in points]
    points_y = [point[1] for point in points]
    interp_func = interp1d(points_x, points_y, kind='linear')

    return interp_func(X)

# stops = [20]
stops = [1, 6, 10, 20]

# 启动加速度 = 0.8 # m/s^2
# 牵引加速度 = 0.5 # m/s^2
# 紧急制动率 = 1.2 # m/s^2
# 常用制动率 = 0.5 # m/s^2
# 车长 = 80 # m
# 制动建立时延 = 0.7 # s
# 牵引切断延时 = 0.7 # s
# 旋转质量系数 = 1.08
# ATP余量 = 3 # km/h
# ATO余量 = 5 # km/h
# max_distance: float = 24000 # m
# max_speed = 87 # km/h
g = 9.81
# m = 800 * 10 ** 3 # kg

# Load the Excel file using pandas
FILE_NAME = '线路条件数据.xlsx'
basic_info_df = pd.read_excel(FILE_NAME, sheet_name='BasicInfo')
station_df = pd.read_excel(FILE_NAME, sheet_name='station')
station_segments = read_segments(
    station_df,
    lambda df, i: df.iloc[i, 0],
    lambda df, i: df.iloc[i, 1],
    lambda df, i: df.iloc[i, 2]
)

curve_df = pd.read_excel(FILE_NAME, sheet_name='curve')
grad_df = pd.read_excel(FILE_NAME, sheet_name='grad')

启动加速度 = basic_info_df.iloc[0, 0] # m/s^2
牵引加速度 = basic_info_df.iloc[0, 1] # m/s^2
紧急制动率 = basic_info_df.iloc[0, 2] # m/s^2
常用制动率 = basic_info_df.iloc[0, 3] # m/s^2
车长 = basic_info_df.iloc[0, 4] # m
制动建立时延 = basic_info_df.iloc[0, 5] # s
牵引切断延时 = basic_info_df.iloc[0, 6] # s
旋转质量系数 = basic_info_df.iloc[0, 7]
ATP余量 = basic_info_df.iloc[0, 8] # km/h
ATO余量 = basic_info_df.iloc[0, 9] # km/h
max_distance = basic_info_df.iloc[0, 10] # m
max_speed = basic_info_df.iloc[0, 11] # km/h

# length: x_scale * max_distance
# index: [0, xscale * max_distance)
# value: [0, max_distance]
x_scale = 1
X = np.arange(0, max_distance, 1 / x_scale).tolist()

def calc_grad_a() -> List[float]:
    grad_a_segments = read_segments(
        grad_df,
        lambda df, i: df.iloc[i, 0],
        lambda df, i: df.iloc[i, 1],
        lambda df, i: df.iloc[i, 2] * grad_a(sign(df.iloc[i, 3]), 旋转质量系数)
    )
    return segments_to_points(grad_a_segments)

def calc_static_limit() -> List[float]:
    static_limit_segments = []
    static_limit_segments += read_segments(
        station_df,
        lambda df, i: df.iloc[i, 0],
        lambda df, i: df.iloc[i, 1],
        lambda df, i: df.iloc[i, 2]
    )
    static_limit_segments += read_segments(
        curve_df,
        lambda df, i: df.iloc[i, 0],
        lambda df, i: df.iloc[i, 1],
        lambda df, i: df.iloc[i, 7]
    )
    static_limit_segments = sorted(static_limit_segments, key=lambda x: x[0][0])
    return segments_to_points(static_limit_segments, default_y=max_speed) # km/h

def calc_v(sbi_y, extra_a) -> List[float]:
    input = [x / 3.6 for x in sbi_y]
    start = int(station_segments[0][1][0] * x_scale)
    end = int(station_segments[-1][1][0] * x_scale)

    v: List[float] = [0 for i in range(start + 1)]
    for i in range(start, end + 1):
        a = 启动加速度 if v[i] < 40 * 10 ** 3 else 牵引加速度
        newv = np.sqrt(2 * (a + extra_a[i]) * (1 / x_scale) + v[i] ** 2)
        if newv > input[i]:
            newv = input[i]
        v.append(newv)

    v = [x * 3.6 for x in v]
    return v

def calc_safev(static_limit_ms, a, extra_a=[0 for i in X]) -> List[float]:
    y = static_limit_ms.copy()
    for i in range(2, len(y)+1):
        safev = np.sqrt(max(2 * (a + extra_a[-i]) * (1 / x_scale) + y[-i + 1] ** 2, 0))
        y[-i] = min(y[-i], safev)

    return y

def calc_bi(static_limit_kmh, break_a, redundant, extra_a=[0 for i in X]) -> List[float]:
    accel_a = 牵引加速度
    # break_a = 紧急制动率
    t0 = 牵引切断延时
    t1 = 制动建立时延
    # redundant = ATP余量
    x_step = 1 / x_scale

    y = [(x - redundant) / 3.6 for x in static_limit_kmh] # add ato offset and convert from km/h to m/s

    # 考虑制动
    safev = calc_safev(y, break_a, extra_a)

    # 考虑制动建立时延
    input = safev.copy()
    res = safev.copy()
    for i in range(1, len(X) + 1):
        v1 = input[-i]

        max_delta_x = v1 * t1
        for step in range(1, int(max_delta_x / x_step) + 1):
            if i + step > len(X):
                break
            res[-i - step] = min(res[-i - step], v1)

    # 考虑牵引切断延时
    input = res.copy()
    for i in range(1, len(X) + 1):
        v1 = input[-i]

        max_delta_x = v1 * t0 - 0.5 * accel_a * t0 * t0
        for step in range(1, int(max_delta_x / x_step) + 1):
            if i + step > len(X):
                break
            delta_x = step * x_step
            v = np.sqrt(-2 * accel_a * delta_x + v1 ** 2)
            res[-i - step] = min(res[-i - step], v)

    return [x * 3.6 for x in res]

def calc_t(v_y, x1, x2):
    x_segment = np.array(X[x1:x2+1])
    y_segment = np.array([x / 3.6 for x in v_y[x1:x2+1]]) # km/h
    # print(y_segment.mean())
    # print(x_segment)
    # print(y_segment)
    y_segment[y_segment==0] = 999999999
    # print(1/y_segment)

    time = np.trapz(1 / y_segment)
    return time

if __name__ == '__main__':
    static_limit_kmh = calc_static_limit()

    extra_a = calc_grad_a()
    ebi_y = calc_bi(static_limit_kmh, 紧急制动率, ATP余量, extra_a)

    sbi_y = calc_bi(static_limit_kmh, 常用制动率, ATO余量, extra_a)
    sbi_y = [x / 3.6 for x in sbi_y] # m/s
    for stop in stops:
        station_start, station_end = station_segments[stop]

        x = int(station_end[0])
        sbi_y[x] = 0
        x -= 1
        while x >= 0:
            v = np.sqrt(2 * (常用制动率 + extra_a[x]) * (1 / x_scale) + sbi_y[x+1] ** 2)
            if sbi_y[x] < v:
                break
            sbi_y[x] = v
            x -= 1
    sbi_y = [x * 3.6 for x in sbi_y] # km/h

    v_y = calc_v(sbi_y, extra_a) # km/h

    fig = go.Figure(data=[
        go.Scatter(x=X, y=static_limit_kmh, name='静态限速'),
        go.Scatter(x=X, y=ebi_y, name='EBI'),
        go.Scatter(x=X, y=sbi_y, name='SBI'),
        go.Scatter(x=X, y=v_y, name='列车运行速度'),
        go.Scatter(x=X, y=extra_a, name='坡度加速度'),
    ])

    end, start = int(station_segments[stops[-1]][1][0] * x_scale), int(station_segments[0][1][0] * x_scale)
    dis = end - start # m
    time_stop = np.sum([station_df.iloc[x, 3] for x in stops] + [station_df.iloc[0, 3]])
    time = calc_t(v_y, start, end)  # s
    v = dis / (time + time_stop)
    fig.update_layout(
        title=f'作业罢了 航行速度 = {dis}/({time} + {time_stop}) = {v} m/s = {v * 3.6} km/h',
        xaxis_title='Distance (m)',
        yaxis_title='Speed (km/h)',
        xaxis=dict(
            range=[station_segments[0][0][0], station_segments[-1][1][0]],
            rangeslider=dict(
                range=[station_segments[0][0][0], station_segments[-1][1][0]],
                visible=True
            )
        ),
        shapes=[
            dict(type="rect", x0=s[0][0], x1=s[1][0], y0=0, y1=90, layer="below", fillcolor="lightblue", opacity=0.5)
            for s in station_segments
        ]
    )
    fig.show()
