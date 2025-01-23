import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
import numpy as np
import seaborn as sns
my_font_prop = FontProperties(fname="/mnt/file2/changye/Times New Roman Regular.ttf")
# mpl.rcParams['font.family'] = my_font_prop
# # 使用seaborn的主题让图表整体更美观
# sns.set_theme(style="whitegrid", font_scale=1, rc={"font.family": my_font_prop})

# 数据准备
conditions = ["Mistral-7B,\n Text", "LLaVA-NEXT-7B,\n Text", "LLaVA-NEXT-7B,\n Text&Vision"]
scores = {
    "Zero": [10.37, 10.37, 10.37],
    "SAE Mistral": [6.19, 6.78, 6.85],
    "SAE Llava": [6.79, 6.76, 6.90],
    "SAE-V Llava": [2.71, 2.24, 2.64],
    "Original": [1.62, 1.67, 2.50]
}

methods = list(scores.keys())
colors = ["#4b0080", "#1D24CA", "#98ABEE", "#008000","#8bc34a"]

n_methods = len(methods)
n_conditions = len(conditions)

x = np.arange(n_conditions)
width = 0.15

fig, ax = plt.subplots(figsize=(10, 6))

for i, method in enumerate(methods):
    offset = (i - (n_methods - 1) / 2) * width
    ax.bar(x + offset, scores[method], width, label=method, color=colors[i])

ax.set_xticks(x)
ax.set_xticklabels(conditions, rotation=0,fontproperties=my_font_prop, fontsize=24)

ax.set_ylabel("Reconstruction Loss", fontproperties=my_font_prop,fontsize=24)
# ax.set_title("Cross-Modal Transferability of SAE and SAE-V", fontsize=24)
legend=ax.legend()
for text in legend.get_texts():
    text.set_fontproperties(my_font_prop)
    text.set_fontsize(18)

ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

# 保存为PDF格式（可根据需要修改文件名和格式）
fig.savefig("reconstruction_scores.pdf", dpi=300)
plt.show()
