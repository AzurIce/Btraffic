依赖安装：

```
pip install pandas matplotlib openpyxl numpy scipy plotly
```



进行紧急制动的过程中由于「牵引切断时延」和「制动建立时延」会导致存在三个阶段：

1. 牵引：加速度为 **加速度**，持续 $t_0$
2. 切断牵引，仍未建立制动：加速度为 $0$，持续 $t_1$
3. 制动：加速度为 **制动率**



**先推出一条只考虑 3 的曲线**

在任意一个点 $x_1$，我们可以逆推绘制出以该点为终点的 **制动** 速度曲线：
$$
v(x_1-\delta_x) = \sqrt{-2a\delta_x + v(x_1)^2}
$$
将这条曲线与限速曲线取最小即可得到满足该点安全的制动安全曲线。从后往前对所有点进行重复，即可得到整体的制动安全曲线。

然而在实际实现上，数据是离散的采样点，而非连续直线，计算机也没有两条曲线取最小的直觉，只能比较同一 $x$ 处的 $y$。

**然后由这条曲线作为输入，考虑 1 和 2，再推出最终的曲线**

假设对于 $x > x_0$ 的速度均已得到，那么 $x_0$ 处的速度 $v(x_0)$ 受到其后 $\delta_t \in [0, t_0 + t_1]$ 时间内的速度的限制：
$$
v(x_0) = \min \begin{cases}
    v(x_0 + \delta_x) - a\delta_t, &0 \leq \delta_t < t_0\\
    v(x_0 + \delta_x) - at_0, &t_0 \leq \delta_t \leq t_0 + t_1\\
\end{cases}
$$
或者说如果我们已知 $x_1$ 处的速度 $v(x_1)$，那么它会限制其前 $\delta_t \in [0, t_0 + t_1]$ 内的速度：
$$
v(x_1 - \delta_x) \leq \begin{cases}
	v(x_1), & 0 \leq \delta_t < t_0\\
	v(x_1) - a(\delta_t - t_0), & t_0 \leq \delta_t \leq t_0 + t_1
\end{cases}
$$
现在我们希望能够用 $\delta_x$ 来表示 $\delta_t$，可以转化为：
$$
v(x_1 - \delta_x) \leq \begin{cases}
	v(x_1), & 0 \leq \delta_x < v(x_1)t_1\\
	\sqrt{-2a(\delta_x - v(x_1)t_1) + v(x_1)^2}, & v(x_1)t_1 \leq \delta_x \leq t_1 + v(x_1)t_0 - \frac{1}{2}at_0^2
\end{cases}
$$

用这个式子逆着再递推一边就好了。



下面这个不对


在这个基础之上，可以再加入 1、2 的考虑，逆推 $\Delta t = t_0 + t_1$ 时间前，$x_0$ 处的速度 $v(x_0)$：
$$
\begin{align}
x_0
&= x_1 - [v_1t_1] - [(v_1 - at_0)t_0 + \frac{1}{2}at_0^2]\\
&= x_1 - [v_1t_1] - [v_1t_0 - \frac{1}{2}at_0^2]\\

v(x_0) &= v_0 = v(x_1) - at_0 = v_1 - at_0
\end{align}
$$
即 **整个制动过程** 的底线。

然而使用 $\Delta t$ 会导致逆推过程中 $x$ 落点不重叠的问题。










![image-20240604164314449](./assets/image-20240604164314449.png)
