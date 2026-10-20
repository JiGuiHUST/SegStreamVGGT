# SegStreamVGGT 技术总结：把 SegVGGT 式 object-query 语义分割搬到 StreamVGGT，并在 7 个航拍数据集上联合训练

> 面向 `/data1/luhangze/AIC/` 的 7 个航拍/无人机语义分割数据集，在 [StreamVGGT](https://github.com/wzzheng/StreamVGGT)
> （ICLR 2026）上实现 [SegVGGT](https://github.com/IDEA-Research/SegVGGT) 风格的 **object query → mask → 语义分割**
> 流程，并针对"多数据集 + 多分辨率 + 部分标签 + 大图"的实际问题做了 8 项工程/算法改进。
> 本文记录**每项改进的动机、实现位置、实测证据**，便于复现与后续迭代。

---

## 目录

- [0. 一句话总览](#0-一句话总览)
- [1. 基线选择与总体架构](#1-基线选择与总体架构)
- [2. 改进 1：query 塌缩的诊断与逐 query 辅助监督](#2-改进-1query-塌缩的诊断与逐-query-辅助监督)
- [3. 改进 2：多帧 tile-grid 训练（用对流式模型）](#3-改进-2多帧-tile-grid-训练用对流式模型)
- [4. 改进 3：部分标签感知的损失屏蔽（7 数据集联合训练的关键）](#4-改进-3部分标签感知的损失屏蔽7-数据集联合训练的关键)
- [5. 改进 4：分阶段课程训练](#5-改进-4分阶段课程训练)
- [6. 改进 5：Mask2Former 式语义组合与归一化](#6-改进-5mask2former-式语义组合与归一化)
- [7. 改进 6：crop 预算式 epoch 与任意分辨率输入](#7-改进-6crop-预算式-epoch-与任意分辨率输入)
- [8. 改进 7：统一缓存数据管线（含 dstl 16-bit TIFF 修正）](#8-改进-7统一缓存数据管线含-dstl-16-bit-tiff-修正)
- [9. 改进 8：可观测性（query 诊断 / 分层验证 / 可视化评测）](#9-改进-8可观测性query-诊断--分层验证--可视化评测)
- [10. 实测结果汇总](#10-实测结果汇总)
- [11. 与 SegVGGT / Mask2Former 的异同](#11-与-segvggt--mask2former-的异同)
- [12. 局限与后续方向](#12-局限与后续方向)

---

## 0. 一句话总览

| 项目 | 做法 |
|---|---|
| 主干 | StreamVGGT **冻结**（24 frame + 24 global 因果注意力块，DINOv2 ViT-L patch embed，~880M） |
| 分割分支 | 100 个可学习 query（256-d）+ 每层 global block 后 cross-attend + DPT 融合 head + 逐 query 分类 → **54.84M 可训练 (5.69%)** |
| 几何重建部分 | depth/point head 与聚合器主干**全部冻结**（`--train-scope seg` 默认；只训"分割"）|
| 数据 | 7 个数据集 → 统一闭集 **AIC-11**（含 `unlabeled=0` 忽略位），4557 train / 1161 val 张 |
| 训练样本 | **一次前向 = 4 个 crop = 同一张大图上连续铺开的 2×2 网格**（1036² 连续区域） |
| 监督 | 组合概率 CE+Dice（按数据集屏蔽缺失类）**+ 逐 query 辅助监督**（防塌缩） |
| 策略 | **分阶段课程**：先 6/7 数据集都有的常见类 → 4/7 → 全部 11 类 |
| 实测 | epoch 1/12 即达 **val mIoU 0.5322**（7 数据集分层验证），query 诊断健康（7.7 类/样本） |

---

## 1. 基线选择与总体架构

### 1.1 为什么用 StreamVGGT 做 2D 语义分割

StreamVGGT 是 VGGT 的**流式（causal）版本**：global attention 只允许看当前帧和过去的帧，
并支持 KV-cache 增量推理。它输出 depth / point map / camera pose 等几何量，
其 token 特征是"几何接地（geometry-grounded）"的——同一场景的多个视角/多块区域会被聚合到一起。
把语义 query 挂在这种特征上，等价于让分割分支"顺带"利用了几何与跨视图一致性。

本项目**不改动主干**：query 分支是纯增量（见 §1.2），因此行为上完全保留了原模型的几何能力，
`--with-geometry 1` 时还可以同时输出 depth/world_points/pose。

### 1.2 结构映射（SegVGGT → 本项目）

| SegVGGT | SegStreamVGGT | 说明 |
|---|---|---|
| `Aggregator(enable_instance_seg=...)` | `Aggregator(enable_query_seg=...)` | query 作为**独立张量**参与，**不写回 token 序列** → 主干权重 100% 兼容（实测：加载 1210 个主干张量，仅 `patch_embed.pos_embed` 因分辨率不同由插值处理） |
| `CrossBlock` / `CrossAttention` | 逐字移植 | query 作 Q，主干 token 作 K/V；保留 `P`（每帧 token 数）参数以便读出**帧级注意力**（FADA 式） |
| `instance_queries_proj` + `SemanticHead` | `query_proj` + `SemanticHead` | query 与 DPT dense feature 点积出 mask；分类 MLP 输出 `num_classes+1`（末位 no-object） |
| `instance_maps` / `instance_labels`（实例分割） | `query_masks` / `query_logits` → `seg_prob`（语义分割） | AIC 只有语义标签，故按 Mask2Former 的**语义模式**组合（§6） |
| LoRA 微调 | 暂未移植（`--freeze-backbone 0` 可全量微调） | |

### 1.3 参数量

| 组件 | 参数量 | 训练 |
|---|---:|---|
| StreamVGGT aggregator | ~880 M | ❄️ 冻结 |
| query 分支（`query_token` 100×256 + 24×CrossBlock + `query_proj`） | 54.8 M | ✅ |
| SemanticHead（DPT 融合 + query 分类） | ~26 M | ✅ |
| **合计** | **963.9 M** | **可训练 54.84 M (5.69 %)** |

> ⚠️ 关键工程决策：SegVGGT 的 CrossBlock 是按 **embed_dim=1024** 建的（24 个块 ≈ 300M 参数）。
> 我们把注意力工作维度降到 **256**（Mask2Former 同款 query 维度），参数量降到 54.8M——
> 在"冻结主干 + 从零训练分支"的设定下更合理（数据量 4.5k 张）。

---

## 2. 改进 1：query 塌缩的诊断与逐 query 辅助监督

### 2.1 现象（首次训练实测）

只用组合后的逐像素概率监督时，模型收敛到一个**退化解**：

```
[query 诊断] 每样本不同类别数 1.00/100 个 query, 平均 mask 面积 1.0000
```

即：100 个 query 全部输出同一个类、mask 全图铺满 → `seg_prob` 在空间上是常数，
模型退化成"每张图一个全局分类器"（val mIoU 看似有 0.33，但**没有任何空间分割能力**）。
可视化也确认预测是整片单色。

### 2.2 根因

Mask2Former / DETR / SegVGGT 之所以不会塌缩，是因为它们用**实例 GT 做 Hungarian 匹配**，
对每个 query 施加"分类 + mask"两项逐 query 监督。AIC 只有**语义标签**，
组合层的 CE+Dice 存在"所有 query 同类别 + mask 恒为 1"这个平凡最优解。

### 2.3 解决方案：固定分配 + 逐 query 辅助监督（`aic_data/losses.py: query_aux_loss`）

1. **类别分配**：query `q → 1 + q % (C-1)`（100 query / 11 类 ≈ 9 个/类；课程阶段只在本阶段开启的类里分配）；
2. **逐 query 分类损失**：`CE(query_logits[q], 分配类)` —— query 学会"我是哪一类"；
3. **逐 query mask 损失**：`BCE + Dice(σ(query_masks[q]), labels == 该类)` —— query 学会"这类在哪里"；
4. 该数据集未标注的类（`valid_class=False`）跳过；
5. 组合层的 CE + Dice **仍然保留**作为主任务损失。

权重：`--query-aux-weight 1.0`（内含 `--query-cls-weight/--query-bce-weight/--query-dice-weight`）。

### 2.4 实测证据

| 配置 | 每样本 query 类别数 | 平均 mask 面积 | val mIoU |
|---|---:|---:|---:|
| 无辅助监督（塌缩） | **1.0** / 100 | **1.000** | 0.33（假） |
| 加辅助监督（400 crop / 2 epoch） | 1 → 2.2 → 4.2 | 1.0 → 0.25 | 0.127 → 0.218 |
| 加辅助监督 + 多帧 tile-grid（完整 1 epoch） | **7.7** / 100 | **0.179** | **0.5322** |

---

## 3. 改进 2：多帧 tile-grid 训练（用对流式模型）

### 3.1 动机

StreamVGGT 支持**任意帧数 × 任意分辨率**输入；而航拍大图（Potsdam 6000²、UAVid 4K、dstl 3.3k²、
HD-Maps 4500²）无法整张送入。若每次只喂 1 块 518²，则(a)浪费了模型的流式能力，
(b)每张图每 epoch 只能采到很局部的信息。

### 3.2 做法（`aic_data/dataset.py`）

一次前向 = `--num-frames S` 个 crop，来源由 `--frames-mode` 控制：

| 模式 | 含义 | 适用 |
|---|---|---|
| **`tile-grid`（默认）** | 同一张大图上**连续铺开的 gh×gw 网格**；S=4 → 2×2，覆盖 1036×1036 连续区域 | 正射/大瓦片；帧间空间相邻，几何注意力有意义 |
| `sequence` | 同一 `group`（UAVid 序列 / HD-Maps 序列 / AeroScapes 片段 / DLRSD 场景）内的**连续帧** | 真正的视频流式训练（StreamVGGT 原始场景） |
| `same-scene` | 同一张图随机位置的 crop | 同场景多视角 |
| `independent` | 同数据集另一张图的 crop | 最弱耦合（对照） |

配套细节：
- **样本级几何增强**：同一帧组的所有帧共享同一套翻转/90°旋转（保证多帧一致）；
- **样本形状**：`images (B,S,3,H,W)`，损失在 S 帧上求平均 → 每次前向多得 S 倍监督。

### 3.3 实测：速度与覆盖面

| 配置 | step 时间 | 每帧吞吐 | 说明 |
|---|---:|---:|---|
| S=1, B=6（6 帧/step） | 0.85 s | 142 ms/帧 | 单块 |
| S=4, B=3（12 帧/step） | 1.83 s | 152 ms/帧 | 2×2 网格，**每步覆盖 4× 面积** |

> 结论：**帧数 S 的收益在"覆盖面 + 多帧上下文"，不在速度** —— 因为 global attention 的代价是
> $O((S\cdot P)^2)$（S=4 时序列长度 4 倍 → 注意力 16 倍）。相同 crop 预算下步数减少 4 倍、单步变慢约 2.2 倍，
> 净吞吐基本持平，但同样多的样本覆盖了连续 4 倍面积。

---

## 4. 改进 3：部分标签感知的损失屏蔽（7 数据集联合训练的关键）

### 4.1 问题

`aic/README.md` §6.2：**某数据集没有标注某个类 ≠ 该类是负样本**。
例如 Potsdam 没有 `person` 类，把人标成 clutter；若把 `person` 当 Potsdam 的负样本训练，
会产生系统性错误监督。

### 4.2 实现（三层防线）

1. **registry 声明**：`aic_data/registry.py` 为每个数据集声明它**真正标注过的类** $C_d$
   （uavid `{1,2,5,6,9,10}`、aeroscapes `{1,2,5,9,10,11}`、dlr3k `{9}`、dlrsd `{1,2,5,6,7,8,9}`、
   dstl `{1,2,5,6,8,9}`、potsdam `{1,2,5,6,9}`、hdmaps `{1,2,3,4}`）；
2. **标签 LUT**：`dataset._label_lut()` 把"该数据集 $C_d$ 之外的 id"（含扩展类 12..27）统一映射为
   `ignore=0`，避免出现"概率被屏蔽、却仍是监督目标"的矛盾；
3. **概率归一化**：`losses.masked_prob()` 把未标注类的概率置 0 后**重新归一化**再算 CE/Dice，
   等价于"只在 $C_d$ 上做 softmax"；评测时同样屏蔽后 argmax。

### 4.3 评测协议

`IoUMeter` 按（数据集 × 类）累计 TP/FP/FN，**只统计该数据集真正标注过的类**，
再给出 `ALL`（全局）和逐数据集 mIoU（`aic/README.md` §6.4）。

---

## 5. 改进 4：分阶段课程训练

### 5.1 动机

7 个数据集的**类别覆盖率极度不均**（实测）：

| 覆盖率 | 类 |
|---|---|
| 6/7 | `building(1)`、`road(2)`、`vehicle(9)` |
| 5/7 | `tree(5)` |
| 4/7 | `low_vegetation(6)` |
| 2/7 | `water(8)`、`person(10)` |
| 1/7 | `sidewalk(3)`、`parking(4)`、`terrain(7)`、`sky(11)` |

（严格"所有 7 个数据集都有"的类只有 `vehicle`。）
如果一开始就同时监督 11 类，稀有类（sky/water/person/sidewalk/parking/terrain 只来自 10~24 张图）
既样本少又容易干扰常见类的收敛。

### 5.2 做法（`train_aic.py: build_curriculum / set_stage`）

- 按 `--stage-coverage` 的覆盖率阈值分层（默认 `0.85,0.5,0`）：

| 阶段 | 类集合 | epoch（auto） | 依据 |
|---|---|---:|---|
| stage 1 | `{1 building, 2 road, 9 vehicle}` | 3 | 6/7 数据集都有（"几乎所有训练集都有的常见类"）|
| stage 2 | `+ {5 tree, 6 low_vegetation}` | 2 | 4~5/7 数据集有 |
| stage 3 | `+ {3,4,7,8,10,11}` → 全部 11 类 | 7 | 补齐 AIC-11 |

- epoch 分配默认 `auto`（按本阶段新增类数加权），也可 `--stage-epochs 3,2,7`；
- 也可用 `--stage-classes "1,2,9;1,2,5,6,9;all"` 完全自定义；
- **单调递增**：后一阶段包含前一阶段 → 不存在"学新类忘旧类"的灾难性遗忘；
- 阶段切换时同步更新：① loss 的类掩码（交集 $C_d \cap$ 本阶段类）② 标签 LUT（阶段外类像素 → ignore）
  ③ **query 的类别分配**（100 个 query 只在本阶段类里均匀分配，容量不被浪费）。

### 5.3 观测

每个 epoch 的验证同时打印全局 mIoU 与**本阶段类的平均 IoU**，`metrics.csv` 增加 `stage` 列：

```
epoch,stage,lr,train_loss,...,val_loss,val_miou,miou_uavid,...
```

> 注：验证始终按**全部 AIC-11 类**统计（含尚未开训的类），所以早期 `val_miou` 偏低是正常的；
> 关注点是"本阶段类 IoU"与后期全类 mIoU 的爬升。

---

## 6. 改进 5：Mask2Former 式语义组合与归一化

### 6.1 做法（`segstreamvggt/models/segstreamvggt.py: combine_query_predictions`）

$$\text{mask}_q(x)=\sigma(\text{query\_masks}_q(x)),\qquad p_q(c)=\mathrm{softmax}_c(\text{query\_logits}_q)$$

$$p(c\,|\,x)=\frac{\sum_{q} p_q(c)\,\text{mask}_q(x)}{\sum_{c'}\sum_{q} p_q(c')\,\text{mask}_q(x)}$$

- 分母的**逐像素归一到 1** 是必要的：$\sum_q p_q(c)\sigma(m_q)$ 的原始值并不构成分布，
  不归一化会让 CE/Dice 的尺度随 query 数漂移；归一化后等价于"在 query×类 联合空间做 softmax（丢弃 no-object）"；
- `no-object` 槽：query 的分配类在该图中不存在时，其 mask 会被逐 query 监督压到 0，
  于是该 query 对组合概率贡献≈0，no-object 语义自然出现。

### 6.2 与 SegVGGT 的差别

SegVGGT 用同一套 `query_masks` 做**实例**分割（query ↔ 实例一一对应，需 Hungarian 匹配）；
本项目把它用于**语义**分割（query ↔ 类的一部分），并用逐 query 辅助监督补上匹配信号。

---

## 7. 改进 6：crop 预算式 epoch 与任意分辨率输入

### 7.1 crop 预算

- `--max-samples` 的语义是**每 epoch 的 crop 预算**，样本数 = 预算 / `num_frames`
  （0 = 默认按全库图片数 × num_frames）；
- 每个 epoch 用 `seed+epoch` 重新有放回抽样 → 同一张图在不同 epoch 取到不同区域；
- 训练日志打印每个数据集的**采样占比 / 每 epoch crop 数 / crop-per-image**，可直接核对数据是否被充分使用。

### 7.2 任意分辨率

- `--img-size` 支持 int、`"518x728"`、`(H,W)`；H/W 只需是 patch_size(14) 的倍数；
- patch_embed 的 position embedding 按实际网格**双线性插值**，因此非正方形/更大分辨率开箱可用；
- 大图的另一种处理是**降分辨率**：`--scale-max` 允许裁比输入更大的区域再缩放进来
  （等效于更大 GSD 的全局上下文），与"原分辨率细看"形成多尺度互补。

---

## 8. 改进 7：统一缓存数据管线（含 dstl 16-bit TIFF 修正）

### 8.1 设计

`python -m aic_data.precache --datasets all --workers 8`（约 3.7 分钟）把 7 个数据集转成统一格式：

```
cache/<dataset>/<split>/img/<name>.jpg     # RGB (q95)
cache/<dataset>/<split>/lbl/<name>.png     # uint8 统一索引图 (AIC-11, 0=ignore)
cache/<dataset>/<split>/manifest.json      # [{name, group, width, height}]
```

理由：① 7 个数据集的标签格式/颜色/索引互不相同（上游 README §3/§4.4），映射只做一次；
② 大 TIFF/JPEG2000 反复解码慢；③ 训练/验证划分固定下来保证可比。

| dataset | 原始 → 缓存 | train/val |
|---|---|---:|
| uavid | 4K 斜视视频帧, RGB 彩色标签 | 200/70 |
| aeroscapes | 索引 PNG (0..11), 官方划分 | 2621/648 |
| dlr3k | `.samp` 旋转框 → vehicle mask（其余 ignore） | 8/2 |
| dlrsd | 调色板 PNG (0..17), 8:2 确定性划分 | 1680/420 |
| dstl | WKT 多边形栅格化（16-bit TIFF 见 §8.2） | 20/5 |
| potsdam | RGB TIFF 6 色（noBoundary 版） | 18/6 |
| hdmaps | 航拍 RGB 5 色（train/test 天然划分） | 10/10 |

### 8.2 dstl 的两个坑（已修）

1. **影像是 16-bit planar TIFF**：`BitsPerSample=(16,16,16)` 且为 planar 布局，
   PIL 会按 8-bit 交错解码出**棋盘伪影** → 改用 `tifffile` 读 planar 数据 + 1–99 百分位对比度拉伸；
2. **WKT 对齐校验**：上游 README 提醒"未能自动确认对齐"。我们用**标签边界梯度比**做客观校验：
   把标签边缘与影像梯度做相关，对齐时参考区间 1.8–2.6、随机 mask 为 1.00；
   实测 dstl 为 **1.833** → 确认对齐（`precache --verify` 可复现）。

---

## 9. 改进 8：可观测性（query 诊断 / 分层验证 / 可视化评测）

| 工具 | 作用 |
|---|---|
| 训练日志 `[q: Ncls areaX.XX]` | 每 `LOG_EVERY` 步打印当前 batch 的 query 类别数与 mask 平均面积 → **塌缩一眼可见** |
| 验证打印 `[query 诊断]` | 全局统计（unique classes / mask 面积 / 最大 mask 面积） |
| `[stage k] 本阶段类平均 IoU` | 课程进度可读 |
| `stratified_val_indices` | 验证子集按数据集分层 + 轮转交错，保证 n 很小时每个数据集都被评到 |
| `eval_aic.py` | 逐数据集 mIoU + 逐类 IoU + `RGB｜GT｜预测｜置信度` 四联可视化；`--baseline` 对比未训练分支 |
| `verify_pipeline.py`（无需 GPU） | 缓存可读性、标签 id ⊆ {0}∪$C_d$、`valid_class` 与 registry 一致、多帧/非方形形状 |
| `check_setup.py`（无需 GPU） | ① 预训练权重加载报告（按模块：ckpt 载入 vs 新初始化）② 模块级冻结表 ③ forward+backward 验证冻结参数无梯度 |
| TensorBoard | `epoch/train_*`、`epoch/val_miou`、`miou/<dataset>` |

---

## 10. 实测结果汇总

### 10.1 演进对照（同一验证口径：400 样本分层验证）

| 版本 | 数据方式 | query 辅助监督 | epoch 1 mIoU | query 诊断 |
|---|---|---|---:|---|
| v1 | 单块 518² | ❌ | 0.33（**塌缩**） | 1.0 类 / 面积 1.000 |
| v2 | 单块 518² | ✅ | 0.22（仅 400 crop 小测） | 2.2→4.2 类 / 面积 0.25 |
| **v3** | **4 帧 tile-grid** | ✅ | **0.5322** | **7.7 类 / 面积 0.179** |

### 10.2 v3 epoch 1/12 逐数据集与逐类

| dataset | mIoU | 主要类 IoU |
|---|---:|---|
| aeroscapes | 0.7775 | road 0.94, tree 0.96, vehicle 0.84, person 0.81 |
| potsdam | 0.5664 | building 0.80, vehicle 0.85, road 0.62 |
| uavid | 0.5583 | building 0.92, road 0.81, tree 0.57 |
| dlrsd | 0.4665 | water 0.71, road 0.67, building 0.58 |
| hdmaps | 0.3376 | building 0.83（sidewalk/parking 仅 10 张图，待收敛）|
| dstl | 0.1778 | 跨域最难（5 张验证瓦片）|
| dlr3k | 1.0000 | 只有 vehicle（易）|
| **ALL** | **0.5322** | building 0.72 road 0.85 tree 0.78 person 0.76 water 0.71 vehicle 0.61 sky 0.59 low_veg 0.21 |

### 10.3 可视化

`eval_aic.py` 输出四联图（RGB｜GT｜预测｜置信度）；抽查 UAVid 斜视场景：
预测与 GT 的**大结构一致**（道路带、林地、建筑块），细结构偏粗（当时仅训练 1/12 epoch）。

---

## 11. 与 SegVGGT / Mask2Former 的异同

| 维度 | Mask2Former | SegVGGT | 本项目 |
|---|---|---|---|
| query 交互位置 | Transformer decoder（独立解码器） | 塞进 VGGT aggregator 的每个 global block | 同 SegVGGT（挂在 StreamVGGT 因果 aggregator 上） |
| query 监督 | Hungarian 匹配（实例 + stuff 分类） | Hungarian 匹配（实例 GT） | **固定类别分配 + 逐 query CE/BCE/Dice**（无语义以外 GT） |
| 掩码来源 | 像素解码器特征 × query | VGGT DPT 特征 × query | StreamVGGT DPT 特征 × query（`einsum`） |
| 语义输出 | 逐 query 概率求和 | 未使用（做实例） | **归一化混合**（§6） + 数据集/阶段双掩码 |
| 帧/视图 | 单图 | 多视图（离线） | **多帧因果流式**（tile-grid / sequence），支持 KV-cache 推理 |

---

## 12. 局限与后续方向

1. **主干冻结**：当前只训 54.84M 分支。`--freeze-backbone 0 --backbone-lr-mult 0.1` 可全量微调；
   SegVGGT 的 LoRA 方案尚未移植（可显著省显存）。
2. **无实例监督**：逐 query 监督用"固定类别分配"近似匹配；若将来有实例/连通域伪标签，
   可换成 Hungarian 匹配获得更好的 query 利用率。
3. **多帧无几何监督**：`sequence`/`tile-grid` 只利用跨帧上下文，没有位姿/深度 GT 做一致性约束
   （AIC 数据集无位姿标注；UAVid/HD-Maps/DLR3K 有潜质）。
4. **类别极不平衡**：目前靠采样平衡 + 课程；`losses.seg_ce_loss` 已支持 `class_weights`，
   可加入逆频率/中位数频率加权（无效类自动屏蔽）。
5. **放大分辨率**：`--out-ratio 2`（259²）已比 H/4 细；若显存允许可试 `--out-ratio 1`。
6. **推理侧**：StreamVGGT 支持 KV-cache 增量推理，本项目 `Aggregator.forward(use_cache=True)`
   已兼容（query 在 cached 模式下只读当前帧 token）——可用于长视频/大图的在线分割。

---

## 附：本仓库文件与文档索引

| 文件 | 内容 |
|---|---|
| `README.md` | 使用手册（安装/数据/训练/评测/超参/命令） |
| `TECHNICAL_NOTES.md` | 本文（改进与创新的技术总结） |
| `segstreamvggt/models/aggregator.py` | StreamVGGT aggregator + object query 分支 |
| `segstreamvggt/models/segstreamvggt.py` | 模型封装 + 语义组合（`combine_query_predictions`） |
| `segstreamvggt/heads/semantic_head.py` | DPT 融合 + query 分类 |
| `segstreamvggt/layers/{attention,block}.py` | `CrossAttention` / `CrossBlock`（含帧注意力读出） |
| `aic_data/registry.py` | 7 数据集登记表与 $C_d$ |
| `aic_data/precache.py` | 统一缓存（含 dstl TIFF/WKT 处理） |
| `aic_data/dataset.py` | 多帧 tile-grid / sequence 采样、增强、部分标签 LUT |
| `aic_data/losses.py` | 屏蔽式 CE+Dice、逐 query 辅助监督、IoU 统计 |
| `train_aic.py` | 训练（DDP + 课程 + 诊断 + TensorBoard） |
| `eval_aic.py` | 评测与可视化 |
| `verify_pipeline.py` | 数据管线自检（无需 GPU） |
| `check_setup.py` | 初始化与冻结范围自检（预训练加载报告 / 模块级冻结表 / 梯度校验，无需 GPU） |
