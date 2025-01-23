import plotly.graph_objects as go
import numpy as np

def generate_standalone_colorbar(
    min_score=70.83,
    max_score=108.17,
    color_min="rgb(255,200,200)",  # 浅红
    color_max="rgb(180,0,0)",      # 深红
    out_file="colorbar_only.pdf"
):
    """
    生成一张仅包含单独Color Bar的小图：
    最浅红(低分=70.83) -> 最深红(高分=108.17)，并设置全局字体为Times New Roman。
    """
    # 构造一行数据，从min_score到max_score，用于触发颜色映射
    z_values = np.linspace(min_score, max_score, 50)
    
    heatmap = go.Heatmap(
        # 这里将z设为一行[[...]]，或者一列[[v] for v in ...]都可以
        z=[z_values],
        zmin=min_score,
        zmax=max_score,
        colorscale=[
            [0.0, color_min],
            [1.0, color_max],
        ],
        # 只要showscale=True，就会在右侧显示 colorbar
        showscale=True,
        # 配置 colorbar
        colorbar=dict(
            title="Score",
            # 只在色条上显示首末两个刻度值
            tickvals=[min_score, max_score],
            ticktext=[f"{min_score}", f"{max_score}"],
            thickness=20,   # color bar 宽度
            len=0.8         # color bar 占图高度的比例
        ),
        # 将热图本身透明化，不要显示主色块，只保留右侧color bar
        opacity=0
    )

    fig = go.Figure(data=heatmap)
    
    # 全局字体设为 Times New Roman
    fig.update_layout(
        font=dict(
            family="Times New Roman",
            size=14,       # 自行调节
            color="black"
        ),
        width=200,       # 图像宽度
        height=400,      # 图像高度
        margin=dict(l=0, r=0, t=0, b=0)  # 让色条尽量居中
    )

    # 隐藏坐标轴
    fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False)

    # 保存为图像文件（需要安装 kaleido 或 orca）
    fig.write_image(out_file)
    print(f"已保存 color bar 小图: {out_file}")

# 使用示例
if __name__ == "__main__":
    generate_standalone_colorbar()
