import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
import os
from scipy.interpolate import make_interp_spline
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
                # 只取逗号后的小数
                try:
                    val = float(parts[1])
                    decimals.append(val)
                except ValueError:
                    pass  # 如果转浮点失败，跳过
    return np.array(decimals)


def bin_and_spline(decimals, n_bins=10):
    """
    对 decimals (小数) 做分箱统计，并对 (bin_center, frequency) 做三次样条插值。
    返回插值后的 x_fit, y_fit 用于绘图。
    
    参数:
    - decimals: 需要统计的小数数组
    - n_bins: 分箱数量
    
    说明:
    - 这里以 decimals 的最小值、最大值为区间，用 linspace 均匀分成 n_bins 个区间。
    - 若你的数据永远在 [0,1]，也可固定 bin_edges = np.linspace(0,1,n_bins+1)。
    """
    # 生成直方图
    frequencies, bin_edges = np.histogram(decimals, bins=n_bins)

    # 计算每个区间的中心点（用于插值）
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 使用样条插值将直方图转换为平滑曲线
    x_fit = np.linspace(bin_centers[0], bin_centers[-1], 300)  # 插值点
    y_fit = make_interp_spline(bin_centers, frequencies)(x_fit)

    return x_fit, y_fit

def get_second_last_folder_name(filepath):
    """
    获取文件路径中的倒数第二级目录名称。
    例如："/path/to/folder/filename.txt" -> "folder"
    """
    parent_dir = os.path.dirname(filepath)  # 获取文件的父目录
    second_last_dir = os.path.basename(parent_dir)  # 获取倒数第二级目录名
    return second_last_dir
Score_map={
    "Align-Anything-L0-q0_25":89.30,
    "AA_cooccur_0_25":73.40,
    "AA_l0_0_25":73.60,
    "AA_preference_cosi_0_25":99.47,
    "AA_preference_cosi_0_50":94.77,
    "AA_preference_cosi_0_75":101,
    "AA_preference_cosi_weight":94.07,
    "AA_preference_l0_0_50":96.30,
    "AA_preference_l0_0_75":None,
    "RLAIF-V_Coccur-q0_25_preference":89.20,
    "RLAIF-V_Coocur-q0_25":82.63,
    "RLAIF-V_Coocur-q0_50":77.83,
    "RLAIF-V_Coocur-q0_75":72.93,
    "RLAIF-V_Cosi-q0_25":78.20,
    "RLAIF-V_Cosi-q0_50":73.33,
    "RLAIF-V_Cosi-q0_75":72.77,
    "RLAIF-V_L0-q0_50":78.00,
    "RLAIF-V_L0-q0_75":70.83,
    "RLAIF-V-Cosi-q0_25_preference":93.20,
    "RLAIF-V-L0-q0_25_preference":90.20,
    
}
def main():
    txt_files =[
        "/mnt/file2/changye/dataset/Align-Anything-interp/Align-Anything_cosi_weight/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/Align-Anything-interp/Align-Anything-L0-q0_25/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/Align-Anything-preference_interp/AA_cooccur_0_25/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/Align-Anything-preference_interp/AA_l0_0_25/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/Align-Anything-preference_interp/AA_preference_cosi_0_25/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/Align-Anything-preference_interp/AA_preference_cosi_0_50/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/Align-Anything-preference_interp/AA_preference_cosi_0_75/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/Align-Anything-preference_interp/AA_preference_cosi_weight/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/Align-Anything-preference_interp/AA_preference_l0_0_50/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/Align-Anything-preference_interp/AA_preference_l0_0_75/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/RLAIF-V_interp/RLAIF-V_Coccur-q0_25_preference/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/RLAIF-V_interp/RLAIF-V_Coocur-q0_25/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/RLAIF-V_interp/RLAIF-V_Coocur-q0_50/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/RLAIF-V_interp/RLAIF-V_Coocur-q0_75/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/RLAIF-V_interp/RLAIF-V_cosi_weight/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/RLAIF-V_interp/RLAIF-V_Cosi-q0_25/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/RLAIF-V_interp/RLAIF-V_Cosi-q0_50/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/RLAIF-V_interp/RLAIF-V_Cosi-q0_75/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/RLAIF-V_interp/RLAIF-V_L0-q0_50/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/RLAIF-V_interp/RLAIF-V_L0-q0_75/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/RLAIF-V_interp/RLAIF-V-Cosi-q0_25_preference/cosi_feature_list.txt",
        "/mnt/file2/changye/dataset/RLAIF-V_interp/RLAIF-V-L0-q0_25_preference/cosi_feature_list.txt",
        ]
    if not txt_files:
            print("没有指定任何 txt 文件！")
            return

    all_traces = []
    file_labels = []

    for filename in txt_files:
        if not os.path.isfile(filename):
            print(f"文件 {filename} 不存在，跳过。")
            continue

        # 提取倒数第二级目录名称作为名字
        file_name_only = get_second_last_folder_name(filename)

        # 读取数据并处理
        decimals = read_decimal_data(filename)
        x_fit, y_fit = bin_and_spline(decimals, n_bins=10)
        if x_fit is None or y_fit is None:
            print(f"文件 {filename} 无法插值或数据不足，跳过。")
            continue

        # 添加曲线
        trace = go.Scatter(
            x=x_fit,
            y=y_fit,
            mode='lines',
            name=file_name_only,  # 使用倒数第二级目录名
            visible=True  # 默认可见
        )
        all_traces.append(trace)
        file_labels.append(file_name_only)

    if not all_traces:
        print("没有可绘制的曲线。")
        return

    fig = go.Figure(data=all_traces)

    # 交互按钮
    n_files = len(file_labels)
    buttons = []

    # a) Show All 按钮：让所有曲线都可见
    show_all_visibility = [True] * n_files
    buttons.append(dict(
        label="Show All",
        method="update",
        args=[{"visible": show_all_visibility}]
    ))

    # b) Hide All 按钮：让所有曲线都隐藏
    hide_all_visibility = [False] * n_files
    buttons.append(dict(
        label="Hide All",
        method="update",
        args=[{"visible": hide_all_visibility}]
    ))

    # c) 每个曲线的独立按钮：用于切换单条曲线的显示状态
    for i, fname in enumerate(file_labels):
        # 使用 `restyle`，只更新当前曲线的 visible 属性
        buttons.append(dict(
            label=f"Toggle {fname}",
            method="restyle",
            args=[{"visible": [None] * len(file_labels)}, [i]],  # 更新第 i 条曲线的状态
            # args2=[{"visible": [None] * len(file_labels)}, [i]],  # 使其能切换状态
        ))

    fig.update_layout(
        title="Decimal Distribution (Per File) with Multi-Select",
        xaxis_title="Decimal Value",
        yaxis_title="Frequency (Interpolated)",
        updatemenus=[
            dict(
                type="buttons",
                buttons=buttons,
                direction="down",
                x=-0.5,  # 按钮位置靠左
                xanchor="left",
                y=1,  # 按钮顶部对齐图表
                yanchor="top",
                showactive=True
            )
        ],
        margin=dict(l=400, t=50, r=50, b=50)  # 增加左边距避免按钮和坐标轴重叠
    )

    # 保存为 HTML
    output_html = "interactive_plot.html"
    fig.write_html(output_html)
    print(f"交互式图表已保存到 {output_html}")


if __name__ == "__main__":
    main()
