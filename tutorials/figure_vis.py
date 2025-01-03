import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 使用seaborn的主题让图表整体更美观
sns.set_theme(style="whitegrid", font_scale=1)

# 数据准备
conditions = ["Mistral+T", "LLaVA-Mistral+T", "LLaVA-Mistral+T+I"]
scores = {
    "Zero": [10.37, 10.37, 10.37],
    "SAE Mistral+Pile": [6.19, 6.78, 6.85],
    "SAE Llava+Pile": [6.79, 6.76, 6.90],
    "SAE Llava+Obelic": [2.71, 2.24, 2.64],
    "Original": [1.62, 1.67, 2.50]
}

methods = list(scores.keys())


colors = ["#4b0080", "#003399", "#2a52be", "#008000","#8bc34a" ]

n_methods = len(methods)
n_conditions = len(conditions)

x = np.arange(n_conditions)
width = 0.15

fig, ax = plt.subplots(figsize=(10,6))

for i, method in enumerate(methods):
    offset = (i - (n_methods-1)/2)*width
    ax.bar(x + offset, scores[method], width, label=method, color=colors[i])

ax.set_xticks(x)
ax.set_xticklabels(conditions, rotation=0, fontsize=20)

ax.set_ylabel("Reconstruction Loss",fontsize=24)
ax.set_title("Cross-Modal Transferability of SAE and SAE-V",fontsize=24)
ax.legend()

ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

# 保存为PDF格式（可根据需要修改文件名和格式）
fig.savefig("reconstruction_scores.pdf", dpi=300)
plt.show()
