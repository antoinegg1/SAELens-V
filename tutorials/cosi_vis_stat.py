# main.py
import os
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

from scipy.optimize import nnls
from scipy.interpolate import BSpline
# 从 constants.py 中引入常量
from constant import TXT_FILES, SCORE_MAP, LOWEST_SCORE, HIGHEST_SCORE,Chameleon_FILES

def read_decimal_data(filename):
    """
    从文件中读取逗号后的小数部分，忽略逗号前的整数。
    返回一个 numpy 数组，包含所有小数。
    """
    decimals = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释行（如以 '#' 开头）
            if not line or line.startswith('#'):
                continue
            # 假设格式固定为 "整数,小数"
            parts = line.split(',')
            if len(parts) == 2:
                try:
                    val = float(parts[1])
                    decimals.append(val)
                except ValueError:
                    pass
    return np.array(decimals)

def create_hist_data(decimals, n_bins=20):
    """
    对 decimals (小数) 做直方图统计。
    返回: bin_centers, frequencies
    """
    frequencies, bin_edges = np.histogram(decimals, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, frequencies

def generate_b_spline_basis(x, knots, k):
    """
    对给定的 x 数组，生成所有 B 样条基函数在各点的取值。
    - knots: 完整的 B 样条节点向量 (包含边界节点的重复)
    - k: B 样条阶次 (3 表示三次 B 样条)
    
    返回: shape = (len(x), num_basis)
      每一列是一条 B 样条基函数在所有 x 上的取值
    """
    # B 样条的系数向量要和基函数个数对应，比如如果 knots 有 (M) 个节点，
    # 且是 k 次 B 样条，则基函数个数通常 = M - k - 1
    num_basis = len(knots) - k - 1
    
    basis_matrix = []
    
    # 对每一个基函数 j = 0,1,...,num_basis-1
    # 构造一个仅在第 j 个基函数对应位置为 1，其他为 0 的系数数组 c
    # 用 BSpline(...).__call__(x) 求出其在所有 x 上的值
    for j in range(num_basis):
        c = np.zeros(num_basis)
        c[j] = 1.0
        # BSpline(t, c, k) 返回一个可调用对象
        spl = BSpline(knots, c, k, extrapolate=False)
        basis_matrix.append(spl(x))
    
    # basis_matrix 的 shape 是 (num_basis, len(x))
    # 转置为 (len(x), num_basis)
    return np.array(basis_matrix).T

def nonnegative_bspline_fit(bin_centers, frequencies, k=3, num_internal_knots=6):
    """
    使用 B 样条 + 非负最小二乘拟合 (bin_centers, frequencies)。
    - k: B 样条阶数, k=3 代表三次 B 样条
    - num_internal_knots: 内部节点个数 (可根据需求调大/调小)
    
    返回: (x_fit, y_fit, knots, c_coefs)
    其中 x_fit, y_fit 是拟合曲线； knots, c_coefs 可用于后续生成 BSpline 对象或可视化。
    """
    x_data = bin_centers
    y_data = frequencies
    
    # 1) 构造 B 样条节点向量
    #    我们先在 [x_data.min(), x_data.max()] 间均匀布置一些“内部”节点，
    #    再在两端各重复 k 个节点(即边界节点)用于 B 样条构造。
    x_min, x_max = x_data.min(), x_data.max()
    
     # 2) 根据分位数设置内部节点 (自适应)
    #    例如均匀取 0% ~ 100% 的 num_internal_knots 个点
    #    注意最好保证至少有 2 个以上节点，否则就太少了
    percentiles = np.linspace(0, 100, num_internal_knots)
    internal_knots = np.percentile(x_data, percentiles)
    
    # 3) 在两端各重复 k 次边界节点
    knots = np.concatenate((
        np.repeat(internal_knots[0], k),
        internal_knots,
        np.repeat(internal_knots[-1], k)
    ))
    
    # 4) 构造基函数矩阵
    A_data = generate_b_spline_basis(x_data, knots, k=k)
    
    # 5) 非负最小二乘
    c_coefs, _ = nnls(A_data, y_data)
    
    # 6) 生成平滑曲线
    x_fit = np.linspace(x_min, x_max, 300)
    A_fit = generate_b_spline_basis(x_fit, knots, k=k)
    y_fit = A_fit @ c_coefs
    
    return x_fit, y_fit, knots, c_coefs


def get_second_last_folder_name(filepath):
    """
    获取文件路径中的倒数第二级目录名称。
    例如："/path/to/folder/filename.txt" -> "folder"
    """
    parent_dir = os.path.dirname(filepath)  # 获取文件的父目录
    second_last_dir = os.path.basename(parent_dir)  # 获取倒数第二级目录名
    return second_last_dir


def score_to_color(score, lowest_score, highest_score):
    """
    将分数映射到蓝-红区间，如果 score=None 则返回黑色。
    0 -> 蓝色 (0, 0, 255)
    1 -> 红色 (255, 0, 0)
    """
    if score is None:
        return "rgb(0, 0, 0)"  # 黑色
    # 计算归一化的 score 值 [0, 1]
    ratio = (score - lowest_score) / (highest_score - lowest_score)
    r = int(255 * ratio)
    g = 0
    b = int(255 * (1 - ratio))
    return f"rgb({r},{g},{b})"

def create_gradient_heatmap_trace(all_traces):
    """
    从 all_traces 中筛出非黑色曲线，构建二维 (x,y) 网格，并返回一个 go.Heatmap，
    使得在最低曲线到最高曲线之间做蓝->红的连续颜色渐变。
    
    注意：此函数默认你的 trace.line.color 是类似 'rgb(r,g,b)' 的字符串，
         且 'rgb(0, 0, 0)' 表示黑色。如果有别的格式，需要自行适配。
    """
    import numpy as np
    import plotly.graph_objects as go

    # 1) 收集所有非黑色曲线的数据
    #    每个 trace 的 x、y 都是数组，我们把它们取出来
    #    形如: non_black_xys = [ (x_arr, y_arr), (x_arr, y_arr), ... ]
    non_black_xys = []
    for tr in all_traces:
        # 判断是否为黑色（简单判断 '0, 0, 0' 字样）
        if tr.line.color and ('0, 0, 0' not in tr.line.color):
            x_arr = np.array(tr.x, dtype=float)
            y_arr = np.array(tr.y, dtype=float)
            non_black_xys.append((x_arr, y_arr))
    
    if not non_black_xys:
        # 如果全是黑色，或者啥都没有，那就不返回 Heatmap
        return None

    # 2) 拼合所有 x 取最小值和最大值，用于构造统一的 x 轴网格
    all_x = np.concatenate([xy[0] for xy in non_black_xys])
    x_min, x_max = all_x.min(), all_x.max()
    
    # 在 [x_min, x_max] 上均匀取 300 个点
    x_lin = np.linspace(x_min, x_max, 1000)
    
    # 3) 将各条曲线在 x_lin 上插值，得到 shape=(n_lines, 300) 的 y 矩阵
    y_matrix = []
    for (x_arr, y_arr) in non_black_xys:
        # 用 np.interp 做一维线性插值
        y_interp = np.interp(x_lin, x_arr, y_arr)
        y_matrix.append(y_interp)
    y_matrix = np.array(y_matrix)  # (n_lines, 300)
    
    # 4) 再找出所有插值曲线的全局 y_min, y_max
    global_y_min = y_matrix.min()
    global_y_max = y_matrix.max()
    
    # 若差值过小可能会导致除 0
    if np.isclose(global_y_min, global_y_max, rtol=1e-10):
        return None
    
    # 在 [global_y_min, global_y_max] 上均匀取 200 个点
    y_lin = np.linspace(global_y_min, global_y_max, 800)
    
    # 5) 构造一个 shape=(len(y_lin), len(x_lin)) 的 Z 矩阵 (Plotly Heatmap 里 z[i][j] ~ y vs x)
    #    初始全设为 np.nan，用于表示“超出 [min,max] 区域”
    Z = np.full((len(y_lin), len(x_lin)), np.nan, dtype=float)
    
    # 对每个 x_i，都有多条曲线的 y 值 => y_matrix[:, i]
    # 我们取其 min/max
    for i in range(len(x_lin)):
        local_ys = y_matrix[:, i]
        y_min_i = local_ys.min()
        y_max_i = local_ys.max()
        
        # 若这个切片 min==max，也就一条水平线，不填
        if np.isclose(y_min_i, y_max_i, rtol=1e-12):
            continue
        
        # 找到 y_lin 中落在 [y_min_i, y_max_i] 内的部分
        idx = (y_lin >= y_min_i) & (y_lin <= y_max_i)
        # 在这些位置上，用线性归一化计算 fraction
        Z[idx, i] = (y_lin[idx] - y_min_i) / (y_max_i - y_min_i)
    
    # 6) 创建一个 Heatmap trace。Plotly 的 Heatmap 默认 z 的行列含义是 z[j][i] 对应 (x_i, y_j)
    #    colorscale=[(0, 'blue'), (1, 'red')] 表示 0->蓝, 1->红
    #    也可以加 'reversescale=True' 之类看需求
    heatmap_trace = go.Heatmap(
        x=x_lin,
        y=y_lin,
        z=Z,
        colorscale=[ [0, 'blue'], [1, 'red'] ],
        zmin=0,     # 对应 "最底"
        zmax=1,     # 对应 "最顶"
        opacity=1, # 半透明，让曲线也能看见
        showscale=False  # 是否在右侧显示颜色条
    )
    
    return heatmap_trace


def main():
    if not Chameleon_FILES:
        print("没有指定任何 txt 文件！")
        return

    all_traces = []
    mean_score_pairs = []

    for filename in Chameleon_FILES:
        if not os.path.isfile(filename):
            print(f"文件 {filename} 不存在，跳过。")
            continue

        folder_name = get_second_last_folder_name(filename)
        score = SCORE_MAP.get(folder_name, None)
        line_color = score_to_color(score, LOWEST_SCORE, HIGHEST_SCORE)

        decimals = read_decimal_data(filename)
        if decimals.size == 0:
            print(f"文件 {filename} 中没有有效数据，跳过。")
            continue

        bin_centers, frequencies = create_hist_data(decimals, n_bins=20)
        x_fit, y_fit, _, _ = nonnegative_bspline_fit(
            bin_centers, frequencies, k=3, num_internal_knots=6
        )
        if x_fit is None or y_fit is None:
            print(f"文件 {filename} 无法插值或数据不足，跳过。")
            continue

        # 收集曲线 trace
        trace = go.Scatter(
            x=x_fit,
            y=y_fit,
            mode='lines',
            name=f"{folder_name} (score={score})",
            line=dict(color=line_color, width=2),
        )
        all_traces.append(trace)

        # 收集 "平均小数" 与 "score"
        if score is not None:
            mean_val = np.mean(decimals)
            mean_score_pairs.append((mean_val, score, folder_name))

    # --- （1）先做原先的第一张图: 只有曲线，没有渐变 ---
    if not all_traces:
        print("没有可绘制的曲线。")
        return

    fig_original = go.Figure(data=all_traces)
    fig_original.update_layout(
        title={
            "text": "Model Cosimilarity Score Distributions (Colored by Performance)",
            "font": {
                "size": 24,  # 标题字体大小
                "family": "Times New Roman",  # 字体
                "color": "black"  # 字体颜色
            },
        },
        xaxis_title={
            "text": "Decimal Value",
            "font": {
                "size": 20,  # x 轴标题字体大小
                "family": "Times New Roman",  # 字体
                "color": "black"  # 字体颜色
            }
        },
        yaxis_title={
            "text": "Frequency",
            "font": {
                "size": 20,  # y 轴标题字体大小
                "family": "Times New Roman",  # 字体
                "color": "black"  # 字体颜色
            }
        },
        title_font_size=30,
        margin=dict(l=80, t=80, r=50, b=50)
    )

    output_png_original = "static_plot_original.pdf"
    fig_original.write_image(output_png_original, width=1200, height=800)
    print(f"【原先第一张图】已保存: {output_png_original}")

    # --- （2）再做带渐变背景的图 ---
    heatmap_trace = create_gradient_heatmap_trace(all_traces)  # 见下方函数
    if heatmap_trace is not None:
        fig_gradient = go.Figure(data=[heatmap_trace])
        fig_gradient.update_layout(
            title={
                "text": "Heatmap of Model Cosimilarity Score Distributions (Colored by Performance)",
                "font": {
                    "size": 24,  # 标题字体大小
                    "family": "Times New Roman",  # 字体
                    "color": "black"  # 字体颜色
                },
            },
            xaxis_title={
                "text": "Cosimilarity score",
                "font": {
                    "size": 20,  # x 轴标题字体大小
                    "family": "Times New Roman",  # 字体
                    "color": "black"  # 字体颜色
                }
            },
            yaxis_title={
                "text": "Frequency",
                "font": {
                    "size": 20,  # y 轴标题字体大小
                    "family": "Times New Roman",  # 字体
                    "color": "black"  # 字体颜色
                }
            },
            margin=dict(l=80, t=80, r=50, b=50)
        )

        output_png_gradient = "static_plot_gradient.pdf"
        fig_gradient.write_image(output_png_gradient, width=1200, height=800)
        print(f"【带渐变背景的第一张图】已保存: {output_png_gradient}")
    else:
        print("由于全是黑色曲线或其它原因，无法生成渐变背景图。")

    # ===============================
    # 第二张图（散点 + 多项式拟合）逻辑不变
    # ===============================
    if not mean_score_pairs:
        print("无有效 (mean decimal, score) 数据，无法绘制拟合图。")
        return

    mean_score_pairs.sort(key=lambda x: x[0])
    x_vals = [p[0] for p in mean_score_pairs]
    y_vals = [p[1] for p in mean_score_pairs]
    folder_names = [p[2] for p in mean_score_pairs]

    degree = 3
    poly_params = np.polyfit(x_vals, y_vals, deg=degree)
    poly_func = np.poly1d(poly_params)

    x_fit_curve = np.linspace(min(x_vals), max(x_vals), 300)
    y_fit_curve = poly_func(x_fit_curve)

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            text=folder_names,
            hovertemplate=(
                "Mean decimal=%{x:.4f}<br>"
                "Score=%{y:.2f}<br>"
                "Folder=%{text}"
            ),
            name="Data Points"
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=x_fit_curve,
            y=y_fit_curve,
            mode='lines',
            line=dict(color='red', width=2),
            name=f"Poly Fit (deg={degree})"
        )
    )
    fig2.update_layout(
        title={
            "text": "Average Cosimilarity Score-Model Performance Distribution",
            "font": {
                "size": 24,  # 标题字体大小
                "family": "Times New Roman",  # 字体
                "color": "black",  # 字体颜色
            }
        },
        xaxis_title={
            "text": "Average Cosimilarity Score",
            "font": {
                "size": 20,  # x 轴标题字体大小
                "family": "Times New Roman",  # 字体
                "color": "black",  # 字体颜色
            }
        },
        yaxis_title={
            "text":"Model Performance",
            "font": {
                "size": 20,  # 标题字体大小
                "family": "Times New Roman",  # 字体
                "color": "black",  # 字体颜色
            }
        },
        title_font_size=30,
        margin=dict(l=80, t=80, r=50, b=50)
    )
    output_png2 = "score_vs_mean_decimal_fitted.pdf"
    fig2.write_image(output_png2, width=1000, height=600)
    print(f"第二张图(散点 + 多项式拟合)已保存: {output_png2}")


if __name__ == "__main__":
    main()


