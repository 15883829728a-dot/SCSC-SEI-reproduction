# SCSC-SEI · 复现：对比自监督聚类用于特定辐射源识别

> **Unofficial PyTorch reproduction** of
> X. Hao, Z. Feng, R. Liu, S. Yang, L. Jiao, R. Luo,
> *"Contrastive Self-Supervised Clustering for Specific Emitter Identification,"*
> **IEEE Internet of Things Journal**, vol. 10, no. 23, pp. 20803–20817, 2023.
> [DOI: 10.1109/JIOT.2023.3284428](https://doi.org/10.1109/JIOT.2023.3284428)
>
> ⚠️ 本仓库是个人**学习复现**，非论文作者官方实现，与原作者/单位无隶属关系。
> 原论文的 30 辐射源 USRP 数据集**未公开**，本仓库不包含、也不重分发任何受版权
> 保护的代码或数据；默认用**合成数据**跑通流程，真实数据请自行准备。

---

## 这是什么 / What

SCSC 把 **对比聚类（Contrastive Clustering）** 思想用到一维射频信号上，实现
**无标注**的辐射源聚类：

- **1D-FPFE**（一维指纹金字塔特征提取器）：大卷积核（1×15 / 1×19 / 1×23）+ 多尺度金字塔池化，直接吃 1D 实信号或 2 通道 IQ。
- **BPS 信号增强**：把信号切成 M 段，用随机 M-bit 脉冲码决定哪些段做增强，含 4 种方法——段反转 SS / 幅度抖动 AJ / 时序错位 TS / 随机噪声 RS。
- **双对比头**：实例级头 `G_I` + 簇级头 `G_E`，目标函数 `L = L_I + L_E`（簇级损失含熵正则，避免坍缩到单一簇）。
- **单阶段、端到端**：只需指定簇数 K，无需辅助数据集、无需额外聚类后处理；推理时簇头 `argmax` 即为类别。

## 目录结构

```
SCSC-SEI-reproduction/
├── scsc/
│   ├── augment.py     # BPS + SS/AJ/TS/RS 信号增强
│   ├── datasets.py    # 合成辐射源 / RML2016 适配器 / 自带 .npy 加载器 / 对比对包装
│   ├── model.py       # 1D-FPFE 主干 + 实例头 + 簇头 (SCSCNet)
│   ├── losses.py      # 实例级 + 簇级对比损失 (L = L_I + L_E)
│   ├── metrics.py     # 聚类指标：ACC(匈牙利匹配)/NMI/ARI/F
│   └── utils.py
├── train.py           # 训练
├── evaluate.py        # 评估 + t-SNE 可视化
├── requirements.txt
└── README.md
```

## 安装 / Install

```bash
conda create -n sei python=3.10 -y
conda activate sei
# 本机调代码：CPU 版即可
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
# 云端 GPU 训练：装对应 CUDA 版的 torch（见 pytorch.org）
```

## 快速开始（本机 CPU，合成数据，几分钟）

```bash
# 训练：8 个合成辐射源，tau=0.5，40 epoch，CPU
python train.py --dataset synthetic --num-emitters 8 --per-emitter 120 \
    --length 512 --epochs 40 --batch-size 128 --eval-every 5 \
    --instance-temp 0.5 --device cpu
# 评估 + t-SNE 可视化
python evaluate.py --ckpt checkpoints/scsc.pt --dataset synthetic \
    --num-emitters 8 --per-emitter 120 --length 512 --tsne tsne.png
```

合成数据为 **preamble 式**：所有辐射源发同一已知波形，仅硬件指纹（CFO / IQ 失衡 /
功放非线性 / 载波泄漏）不同，逐样本叠加随机相位、时序抖动与噪声。它用于**验证整条
复现流程是否正确**，聚类指标会随训练明显上升（见下表），不是真实 SEI 数据的替代。

> 提示：论文称 RF 信号无需温度系数（等价 `--instance-temp 1.0`，即本仓库默认）；
> 在该合成玩具上 `--instance-temp 0.5` 收敛更快更好。真实数据建议两者都试。

## 用真实 / 公开数据

- **自带 IQ 数据（推荐复现真实 SEI）**：保存为 `X.npy (N,C,L) float32` 与 `Y.npy (N,)`，
  ```bash
  python train.py --dataset npy --x-path X.npy --y-path Y.npy \
      --num-clusters 30 --length 4096 --epochs 200 --batch-size 128 --device cuda
  ```
- **RML2016.10a（DeepSig 公开）**：用于论文中的增强基准（聚类的是调制类型，非辐射源）。
  自行下载 `RML2016.10a_dict.pkl` 后：
  ```bash
  python train.py --dataset rml --pkl-path RML2016.10a_dict.pkl --snr-min 6 \
      --num-clusters 11 --length 128 --epochs 100 --device cuda
  ```

## 云端训练建议

本机 GPU 较弱时，按「本机 CPU 调通 → 云端 GPU 跑全量」的流程：把整个目录上传到
Colab / AutoDL 等，装 GPU 版 torch，`--device cuda` 即可。`--length 4096 --batch-size 128`
对应论文设置（论文用 RTX 3090 24G）。

## 复现结果

| 数据集 | #辐射源 | 训练设置 | ACC | NMI | ARI |
|---|---|---|---|---|---|
| Synthetic (preamble toy) | 8 | CPU · 40 epoch · τ=0.5 | 0.64 | 0.67 | 0.50 |
| 你的真实数据 | 30 | GPU · 200 epoch | _填_ | _填_ | _填_ |

> 合成玩具仅几分钟 CPU 训练即从随机水平（ACC 0.125）升到 **ACC≈0.64 / NMI≈0.67**，
> 且第 40 epoch 仍在上升；加大 epoch、上 GPU 会更高。这验证了 SCSC 流程能端到端
> 学到“辐射源可分”的表征。真实数据请把数字与 `tsne.png` 补进来。

## 与论文的差异 / Notes

- **数据集**：原 30 辐射源 USRP 数据集未公开，故用合成数据 + 公开 RML + 自带数据替代。
- **温度系数**：论文指出 RF 信号无需温度系数，故代码默认 `temperature=1.0`（忠实于论文）；实践中 `--instance-temp 0.5` 通常收敛更快更好，已在快速开始中采用。
- **1D-FPFE**：论文未给出逐层超参，本实现按“大核 + 金字塔多尺度池化”的描述搭建，可在 `scsc/model.py` 调整 `channels/kernels`。
- **开集未知类拒识（ZSL+语义质心）** 与在线增量聚类为论文展望部分，本仓库暂未实现，欢迎 PR。

## 引用 / Citation

```bibtex
@article{hao2023scsc,
  title   = {Contrastive Self-Supervised Clustering for Specific Emitter Identification},
  author  = {Hao, Xiaoyang and Feng, Zhixi and Liu, Ruoyu and Yang, Shuyuan and Jiao, Licheng and Luo, Rong},
  journal = {IEEE Internet of Things Journal},
  volume  = {10},
  number  = {23},
  pages   = {20803--20817},
  year    = {2023},
  doi     = {10.1109/JIOT.2023.3284428}
}
```

## 许可 / License

本仓库**自有代码**以 [MIT](LICENSE) 开源。论文方法与思想版权归原作者所有；
请勿将任何受版权保护的第三方代码或非公开数据集提交到本仓库。
