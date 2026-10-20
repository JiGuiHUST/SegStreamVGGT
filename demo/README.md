# UAV 航迹增量 3D 点云 demo（RGB + 语义）

一个**定性** demo：让无人机沿一条真实航线飞行，**随着航点增加，地面场景的 RGB 点云逐渐补全**；
右栏用训练好的分割头给同一批点云染色，展示“分割点云”随飞行逐渐长出来的过程。

![preview](output/preview_grid.png)

视频：`output/uavid_seq16_route.mp4`（76 帧 / 10 fps / 9.6 s；双栏 RGB | 语义点云，含 3D 航线和左下角航线小地图）；GIF 预览：`output/uavid_seq16_route.gif`

## 它用到了什么

| 部分 | 来源 |
|---|---|
| 几何（depth / camera pose） | **冻结的 StreamVGGT 预训练头**（`/data/luhangze/aic/StreamVGGT/ckpt/checkpoints.pth`） |
| 语义分割 | **SegStreamVGGT v5 query 分支**（`logs_aic_v5/checkpoint_best.pth`，可换 v4） |
| 场景/航线 | UAVid `seq16` 的原始 `images.mp4`，**从第 15 帧开始每 30 帧取一帧**（1.5 s/航点，16 个航点，0.75–23.25 s，约 23 s 航线；避开视频开头的切镜） |
| 点云 | depth + pose 反投影：`p_cam = depth * K⁻¹[u,v,1]`，`p_world = Rᵀ(p_cam - t)`；最终保留 1.5M 点、渲染 400k |

## 复现

```bash
cd /data/luhangze/aic/SegStreamVGGT

# 1) 从 UAVid 完整 split 包中抽出 seq16 的视频
#    7z x /data1/luhangze/AIC/UAVid/uavid_v1.5_official_release_split.zip \
#       "uavid_v1.5_official_release/uavid_v1.5_official_release/uavid_val/seq16/images.mp4"
# 2) 从第 15 帧开始每 30 帧取一帧（1.5s/航点，共 16 帧），缩放到 924x518
ffmpeg -y -i images.mp4 \
       -vf "select='gte(n,15)*not(mod(n-15,30))',scale=924:518" -vsync 0 -frames:v 16 \
       demo/uavid_seq16/frames_route/f_%03d.png

# 3) 跑 demo（16 帧 392x700，4090 上约 15–20 GB 显存）
CUDA_VISIBLE_DEVICES=2 python demo/demo_uavid_route.py \
    --ckpt logs_aic_v5/checkpoint_best.pth --model-label v5 \
    --seq-label "seq16 (1.5s, 23s route)" --image-dir demo/uavid_seq16/frames_route --max-frames 16 \
    --img-size 392x700 --point-source depth --view-elev 22 --view-azim -60 --proj persp \
    --min-new-dist 0.10 --subframes 5 --fps 10 --out demo/output
```

输出：

| 文件 | 内容 |
|---|---|
| `output/uavid_seq16_route.mp4` | 16 航点 × 5 子帧 = 76 帧平滑动画（10 fps，9.6 s） |
| `output/uavid_seq16_route.gif` | 同一视频的 GIF 预览 |
| `output/preview_grid.png` | 4 个阶段（1/3/2/3/最后）的拼图 |
| `output/points_rgb.ply` | 最终 RGB 点云（1.5M 点，可用 MeshLab/Open3D 打开） |
| `output/points_seg.ply` | 最终语义染色点云 |
| `output/summary.json` | 每航点点数、相机中心、参数 |

## 实现要点

1. **几何**：`SegStreamVGGT(enable_camera=True, enable_depth=True)` 输出 `depth`、`pose_enc`；
   `pose_enc` 用 `pose_encoding_to_extri_intri` 解码出 `extrinsic/intrinsic`。
2. **depth + pose 反投影**：`p_cam = ray * depth`，`p_world = Rᵀ(p_cam - t)`。
   这条路径与相机轨迹同尺度，能拼出连续街道；**point head 的 `world_points` 是另一套归一化尺度**，
   直接叠加会出现重影/尺度错位（`--point-source point` 可切换，但不推荐）。
3. **过滤**：每帧按 `depth_conf` 的 10% 分位去掉低置信点；语义 argmax 概率阈值默认 0（保留全部）；
   ignore 类 0、sky 类 11 不进入点云（`--drop-sky 1`）。全局下采样到 1.5M，渲染 400k。
   新增点只取**距当前相机超过 `--min-new-dist`（默认 0.10）** 的区域，避免近处重复堆叠。
4. **平滑补全**：16 个航点之间各插 5 个子帧（`--subframes 5`），下一航点的点按“离该航点相机由近到远”
   逐批加入（near-to-far reveal），当前子帧新增的点用**黄色高亮**（`--highlight-new 1`），避免整帧跳变；视频 76 帧 @ 10 fps。
5. **航线**：相机中心用 PCHIP 按弧长拟合成平滑曲线；3D 场景里画灰色全航迹 + 红色已飞航迹 +
   黄色无人机 + 青色相机视锥；左下角另有一个 2D“Flight route”小地图（全航迹/已飞/当前点）。
6. **显示**：VGGT world 是 y-down，渲染时用 `(x, z, -y)` 让地面朝上；默认**透视、斜向下 22°、方位 -60°**
   （更平行地面），随时间缓慢旋转，能看出建筑立面和街道纵深。
7. **分割颜色**：AIC-11 调色板（与 `eval_aic.py` 一致）；颜色是**模型预测**，不是 GT。

## 已知限制

- 几何头是 **StreamVGGT 预训练权重、未在 AIC/UAVid 上微调**；UAVid 是 4K 斜视，即使 0.75 s 间隔，
  点云仍有沿视线的拖尾/外点，建筑立面会呈“扇形”。本 demo 重点是“随航点补全”和 RGB/语义双栏，
  **不是几何精度指标**。
- 没有做多帧深度融合、TSDF/泊松重建、回环或位姿优化；也没有跨帧标签投票。
- 16 帧覆盖约 23 s 飞行（0.75–23.25 s，避开开头切镜）；若要覆盖整段 45 s，可增大 `--max-frames` 或间隔，但更长间隔会出现更明显的漂移。
- 输入 392×700（16:9）是显存/速度折中；更高分辨率可改 `--img-size 518x924`（更慢、更吃显存）。
- UAVid 只有 10 个标注帧（每 100 帧一个）；视频帧没有逐帧 GT，所以这里用预测分割。

## 下一步

- 用 `use_cache=True` 的 streaming 版本（真正的在线增量）。
- 点云融合：体素/TSDF、统计外点剔除、多帧标签投票。
- 用 COLMAP/已知位姿替代预测位姿；或对每帧 depth 做尺度对齐，减轻扇形拖尾。
