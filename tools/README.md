# tools/

两个配套的 demo 工具，都由本仓库根目录的 SegStreamVGGT（冻结的 StreamVGGT 主干 + 训练好的 query 语义分支）驱动。

| 目录 | 作用 | 入口 |
|---|---|---|
| `code_demo/` | 离线渲染：把重建出来的 RGB / 语义点云做成带轨迹的演示视频（纯 CPU 渲染 + 逐帧出图） | `render_video.py` |
| `make_demo/` | 交互式：上传一段航拍视频，在线跑重建 + 分割，浏览器里看 3D 点云逐步长出来 | `server.py` / `run.sh` |

两者都需要根目录的 `segstreamvggt` 包和对应的 checkpoint（见根目录 README 的「环境与权重」一节）。
`make_demo/` 里的部署脚本（`tunnel_vps.sh`）已做脱敏：主机名、端口、IP 都替换成了占位符，按需填回自己的。
