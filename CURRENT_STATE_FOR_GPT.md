# 交通锥项目当前状态与 GPT 接手指南

更新时间：2026-09-10

本文是当前开发状态的首要入口。历史 `README.md`、`AGENT.md` 和
`script/README.md` 仍包含有价值的设计说明，但其中部分默认数据集、权重和环境名称
已经过时；发生冲突时，以本文列出的产物路径、哈希和验证边界为准。

## 1. 当前工作位置

- 当前主服务器目录：`/home/fs-ai/traffic_cone`
- 当前主机：`fs-ai-H610M-K-DDR4`（`10.8.30.60`）
- GPU：NVIDIA GeForce RTX 5060 Ti，16311 MiB
- Git 分支：`main`
- 同步时 Git HEAD：`7ac7cce54ffe2a8c4c947e61e145bb07814958a1`
- 2026-09-10 已从 `/mnt/cn600_1tb/ZY/traffic_cone` 完成增量同步；相同规则二次
  dry-run 为 0 个待传文件、0 bytes 差异。
- 推理生成视频未同步；`script/test-video` 中 7 个原始测试视频已同步。

开始任务前先执行：

```bash
cd /home/fs-ai/traffic_cone
git status --short
git log -5 --oneline
nvidia-smi
```

工作区包含未纳入 Git 的数据集、权重、测试脚本和开发产物。不要用 `git clean`、
`git reset --hard` 或覆盖式数据整理清除它们。

## 2. 远端运行环境

远端没有此前验证视频推理所用的 `jiahe` 环境。当前可用环境是：

```text
/home/fs-ai/miniconda3/envs/yolo-obb/bin/python
Python 3.10.19
Torch 2.10.0+cu130
CUDA available: True
Ultralytics 8.4.21
OpenCV 4.13.0
```

原服务器的完整视频验证环境曾是 Python 3.12.13、Torch 2.13.0+cu130、
Ultralytics 8.3.253、OpenCV 4.10.0。版本不同，因此远端第一次正式开发前应先跑
短视频冒烟测试，再跑目标完整视频；不要把“模型能加载”等同于行为完全一致。

远端 `yolo-obb` 暂未安装 `pytest`，核心测试可直接用标准库运行：

```bash
cd /home/fs-ai/traffic_cone/script
/home/fs-ai/miniconda3/envs/yolo-obb/bin/python -m unittest \
  test_uav_physical_tracking
```

2026-09-10 远端实测：21 项通过。

## 3. 当前主方案

主链路由一个三分类 YOLOv8s-OBB 检测器和一个可选 MobileNetV3-small 候选审核器
组成：

1. YOLO 输出 `cone / traffic_sign / person` OBB。
2. `person` 框用于抑制人体区域内的假锥桶。
3. 小目标过滤、物理 ID 跟踪和运动补偿后的物理 ROI 处理无人机视角变化。
4. route、gap 和时序状态机产生 `MISSING / SHIFT / FALLEN` 业务告警。
5. MobileNet 在 `fallen` 模式下只补充倒伏票；`not_cone` 仅记录，不删除检测。

关键模块：

- `script/infer_obb.py`：生产式推理入口和输出编排。
- `script/traffic_rules.py`：路线分组及 MISSING/SHIFT/FALLEN 规则。
- `script/uav_physical_cone_counter.py`：运动补偿物理 ID 和独立对比入口。
- `script/test_uav_physical_tracking.py`：物理 ID、ROI、route 对齐和告警转换测试。
- `script/analyze_cone_count_stability.py`：对推理 `events.jsonl` 做稳定性分析。

默认物理 ROI 模式只替换计数和 ROI 绘制的数据源；异常规则仍基于分析 route 和既有
时序状态机。`--no-physical-roi` 可用于旧图像坐标模式 A/B。

## 4. 当前数据集和模型

### YOLO 数据集

当前正式数据集：

```text
dataset/2026-08-14-merged-3class-no-tiny-raw1
```

- 类别：`cone / traffic_sign / person`
- train/val/test：1490 / 320 / 244
- 文件数：4112
- 大小约 3.8G
- `data.yaml` 使用 `path: .`，在远端目录中可移植。
- `manifest.json` 记录来源和过滤过程。
- 它由 `dataset/2026-08-13-merged-3class-hardneg-split` 派生，移除了 12 对只含
  tiny 目标的训练样本。

### YOLO 检测器

当前最新完成的候选检测器：

```text
script/runs/traffic_cone_uav_20260814_no_tiny_s_full_1280/weights/best.pt
SHA-256: 490639342fce10368b84bc9122446c641f1360f7d0009133f75f66a77045586e
```

该训练完成 150 epochs。固定 244 图测试中，cone precision 从旧候选的 88.50%
提高到 93.15%，multiangle recall 从 92.58% 提高到 96.51%；但 16 图
original-cone 子集 recall 从 98.18% 降到 90.00%。因此它是当前推理候选，不应只凭
“最新”自动宣布为生产替代品。

选择或比较 detector 时使用 `script/evaluate_obb_checkpoints.py`，固定
`imgsz=1280`、`conf=0.25`、polygon IoU 0.5，并记录完整权重路径和 SHA-256。
不要按 `best.pt` 修改时间选模型。

### MobileNet 候选审核器

```text
script/runs/mobilenetv3_small_cone_20260817_v2_context16_bp075/best.pt
SHA-256: a4e945b4ab7b55c3027ddcaaa3036d6f58c5e4b306694b454840b224e88edbcb
阈值: 同目录 thresholds.json
crop context: 1.6
```

独立测试的 thresholded fallen recall 为 58/66（87.88%），`not_cone` 测试支持集
仅 13 个。`fallen` 是当前推荐模式；`audit` 只记录；`full` 会在 route 之前删除
`not_cone` 候选并影响 MISSING/SHIFT/FALLEN，未经新的多场景门槛验证不要启用。

`thresholds.json` 中的历史绝对路径只是校准元数据；推理默认读取同目录权重和阈值，
不依赖这些历史路径定位当前权重。

## 5. 可直接执行的推理命令

所有命令从 `/home/fs-ai/traffic_cone` 执行。

先用 10 秒视频验证远端环境：

```bash
cd /home/fs-ai/traffic_cone
PYTHONPATH=script /home/fs-ai/miniconda3/envs/yolo-obb/bin/python \
  script/infer_obb.py \
  --weights script/runs/traffic_cone_uav_20260814_no_tiny_s_full_1280/weights/best.pt \
  --classifier-weights script/runs/mobilenetv3_small_cone_20260817_v2_context16_bp075/best.pt \
  --classifier-mode fallen \
  --classifier-crop-context 1.6 \
  --source script/test-video/test_10s.mp4 \
  --imgsz 1280 --conf 0.25 --iou 0.5 --device 0 \
  --output script/runs/remote_smoke_20260910/test_10s
```

短测通过后再跑当前多角度回归：

```bash
cd /home/fs-ai/traffic_cone
PYTHONPATH=script /home/fs-ai/miniconda3/envs/yolo-obb/bin/python \
  script/infer_obb.py \
  --weights script/runs/traffic_cone_uav_20260814_no_tiny_s_full_1280/weights/best.pt \
  --classifier-weights script/runs/mobilenetv3_small_cone_20260817_v2_context16_bp075/best.pt \
  --classifier-mode fallen \
  --classifier-crop-context 1.6 \
  --source script/test-video/26-08-07-多角度.mp4 \
  --imgsz 1280 --conf 0.25 --iou 0.5 --device 0 \
  --output script/runs/remote_latest_dual_20260910_multiangle
```

输出目录应包含：

- `*_result.mp4`：可视化视频。
- `events.jsonl`：逐帧检测、route、物理 ID、规则和分类器诊断。
- `alert.jsonl`：结构化业务告警激活记录。
- `alert.log`：人工复查日志。
- `cone_count.json`：稳定可见及累计轨迹统计。
- `run_config.json`：本次参数和模型配置。

完成标准：进程退出码为 0，视频可完整解码，事件帧数与输入解码帧数一致，关键告警
片段经人工查看；只生成文件不算完整验证。

## 6. 测试视频

`script/test-video` 当前包含：

- `test_10s.mp4`：最快冒烟测试。
- `test_4m38s-5m_small.mp4`：远距小目标片段。
- `26-08-03_small_cone.mp4`：小目标场景。
- `26-08-03_cone.mp4`：常规锥桶片段。
- `26-08-07-多角度.mp4`：当前完整多角度主回归，5447 帧。
- `DJI_1m22s_1m36s.mp4`：前向运动、视角变化和路线成员变化回归。
- `cone-07-24_raw.mp4`：长焦原始测试视频。

2026-09-01 的验证运行使用 `fallen + crop_context=1.6` 完整处理多角度视频，生成
5447 个事件帧。推理视频本次未复制到远端，但对应 `events.jsonl`、告警日志、统计和
配置仍在 `script/runs/infer_latest_dual_20260901_multiangle`。重新在远端生成视频后，
应将新结果与该目录的结构化输出对比。

## 7. 当前行为约束

- 当前推荐阈值为 `conf=0.25`；`physical-match-conf=0.20` 只允许低置信候选重关联
  已确认/保持的物理 ID，不创建新 ID，不进入异常规则。
- 默认 `min-cone-long-side-ratio=0.022222`。过小 OBB 仍写入诊断，但不进入
  route、告警或普通计数。
- 默认物理 ID 最近 5 帧至少 3 次观察后确认，短时漏检保持；物理 ROI 默认启用。
- 物理 ROI 按运动补偿参考平面分组，并受方向、横向距离和沿线 gap 限制。
- route 中当前检测少于 3 个锥桶时可以继续绘制 ROI，但不能产生稳定业务告警。
- `events.jsonl` 保留逐帧候选；`alert.jsonl` 和 `alert.log` 只记录稳定告警激活转换。
- `unique_seen_count` 是视频内重建轨迹的累计确认数。无人机大幅运动下不能解释为
  真实物理锥桶总数。
- 透视校正只有在俯角和有效水平视场角都可信时启用；要输出米还必须提供相机到锥桶
  地面的垂直高度。

## 8. GPT/Codex 工作流程

接手任何修改时按下列顺序执行：

1. 读取本文及待改模块，执行 `git status --short`，确认用户已有改动。
2. 用当前文件、`--help` 和真实产物核对路径/参数；历史文档只作背景。
3. 先区分问题属于 detector、分类器、物理 ID、route/ROI 还是业务后处理。
4. 做最小范围修改，保持 CLI、JSONL 字段和告警日志接口兼容。
5. 运行直接相关的单元测试；物理 ROI/route 变更至少跑 21 项核心测试。
6. 对用户指定的真实视频回放。需要验证无人机稳定性时，不以静态图片或 10 秒冒烟
   替代完整目标视频。
7. 报告时分开说明静态检查、单元测试、主机推理、完整视频回放和部署端验证。
8. 交付时先给输出目录和主视频，再给指标、限制和未完成项。

模型或阈值变更的验收重点是困难场景：原始锥桶 recall、multiangle recall、倒伏
recall、水面反光/背景误检、远距小目标、route/ROI 身份稳定和业务告警变化。聚合
accuracy 或单个漂亮片段不足以替代这些证据。

## 9. 已知边界和下一步

- 远端 `yolo-obb` 与先前 `jiahe` 版本不同。21 项单元测试已通过，但远端 GPU
  短视频和完整多角度回放尚未执行。
- 最新 YOLO 候选在 original-cone 子集有 recall 回退，生产切换仍需场景权衡。
- MobileNet `fallen` 独立测试 recall 未达到此前设定的 95% 目标，`full` 模式仍需
  更多独立倒伏轨迹和多样水面反光负样本。
- `unique_seen_count` 在跨视角 ID 重建下不是全局物理总量；若业务要求绝对计数，
  需要更可靠的地面映射、视觉定位或跨视角重关联。
- 交通锥专用 RKNN 的完整转换、板端运行、ONNX/RKNN 数值对比和渲染验证尚未完成。
  RKNN Toolkit 2.3.2 不支持 opset 20，继续时需保留源 ONNX、转换副本到 opset 19，
  并以板端/目标 runtime 推理而非仅 `build()` 成功作为完成标准。

## 10. 相关历史资料

需要深入设计背景时再读取：

- `无人机物理ID计数与ROI稳定化-PLAN-2026-08-27.md`
- `FALLEN光斑治理与MobileNetV3二分类器实施计划-2026-08-13.md`
- `doc/无人机巡航自适应ROI时序稳定PRD.md`
- `doc/SHIFT与MISSING误报问题及解决路径-2026-08-06.md`
- `锥桶候选三分类人工标注指南-2026-08-14.md`
- `交通锥倒伏识别成熟方法与公开数据集调研-2026-08-13.md`
- `cvat-nuclio-model-deployment/SKILL.md`

这些资料描述不同时间点的方案。使用其中命令前，先用本文第 4 节的当前数据集、权重
和哈希替换旧路径，并重新确认当前代码的 CLI。
