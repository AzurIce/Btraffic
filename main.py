import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Tuple

# Load the Excel file using pandas
station_df = pd.read_excel('线路条件数据.xlsx', sheet_name='station')
curve_df = pd.read_excel('线路条件数据.xlsx', sheet_name='curve')

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

    static_limit_points: List[Tuple[float, float]] = [(0, 87)]
    x = 0
    for segment in static_limit_segments:
        p1, p2 = segment
        if x < p1[0]:
            static_limit_points.append((x, 87))
            static_limit_points.append((p1[0], 87))
        static_limit_points.append(p1)
        static_limit_points.append(p2)
        x = p2[0]
    if x < 24000:
        static_limit_points.append((x, 87))
        static_limit_points.append((24000, 87))

    static_limit_points_x = [point[0] for point in static_limit_points]
    static_limit_points_y = [point[1] for point in static_limit_points]
    plt.plot(static_limit_points_x, static_limit_points_y, 'b-')

    plt.xlabel('Distance (m)')
    plt.ylabel('Speed Limit (km/h)')
    plt.title('Speed Limit Curve')
    plt.show()
