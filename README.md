# Traffic Cone Detection (OBB)

这是一个基于 YOLOv8s-OBB (Oriented Bounding Box) 的交通导改锥桶检测项目，专为无人机巡航场景设计。项目集成了空间路线自适应分组（Route Tracking）、局部间距平滑（Gap Tracking）、时序状态判定与告警过滤逻辑。

## 功能特性

- **三类别 OBB 检测**：支持 `cone` (锥桶)、`traffic_sign` (交通标志) 和 `person` (人员)。人员类别用于抑制人体区域内的假锥桶，不再单独训练 `vest`。
- **空间路线分组 (Route Grouping)**：自动将画面中符合逻辑距离的锥桶聚合成连续路径 (Route/ROI)。
- **时序跟踪与防抖**：采用多帧投票 (Voting Window) 和状态机，避免由于无人机视角变化导致的锥桶短暂漏检或抖动。
- **透视畸变校正 (Perspective Correction)**：支持设定无人机摄像头俯仰角、视场角和飞行高度，在针孔相机模型下将像素距离转换为物理地面距离 (米)。
- **结构化告警日志**：区分逐帧检测结果与经过业务防抖过滤的稳态告警，自动输出可视化视频、`events.jsonl` (详细)、`alert.jsonl` (结构化告警) 和 `alert.log` (人工复查日志)。

## 环境安装

请在项目根目录执行以下命令，使用 Conda 创建并激活所需环境：

```bash
conda env create -f script/environment.yml
conda activate yolo-obb
```

## 数据集处理与训练

### 构建融合数据集

如果需要混合 OBB 锥桶数据和 AABB 人员数据，可以使用自带脚本将人员水平框转换为四点 OBB：

```bash
python script/build_fused_dataset.py
```
历史四分类融合数据保留不变。三分类数据由下列命令单独生成：

```bash
python script/convert_obb_classes.py
```

当前默认训练集是 `dataset/2026-08-05-merged-3class/`，类别为
`cone / traffic_sign / person`。它在三分类基线上加入 2026-08-05 审核通过的
174 张纯负样本和 43 张正负补充图；新增数据只进入 train，val/test 不变，
来源、审核文件哈希和对象计数记录在 `manifest.json` 中。

### 训练模型

验证数据集后即可开始训练，以下提供推荐参数 (当前主力模型采用 `YOLOv8s-OBB`，输入尺寸 960)：

```bash
python script/validate_dataset.py --data-root dataset/2026-08-05-merged-3class
python script/train_obb.py \
  --data dataset/2026-08-05-merged-3class/data.yaml \
  --model yolov8s-obb.pt \
  --device 0 \
  --epochs 150 \
  --batch 16 \
  --imgsz 960 \
  --cache \
  --name traffic_cone_3class_yolov8s_obb_960
```
> 若意外中断，可通过 `--resume script/runs/.../weights/last.pt` 恢复训练。

## 视频/图像推理与跟踪

### 基础视频推理

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_3class_yolov8s_obb_960/weights/best.pt \
  --source script/test-video/test_10s.mp4 \
  --imgsz 1280 \
  --device 0
```
结果默认保存在 `script/runs/infer/` 目录下。

无人机远距小目标推理默认使用 `imgsz=1280`；当前显卡实测约 90 FPS，仍高于
29.97 FPS 输入视频。`imgsz=960` 约 145 FPS，可作为低延迟或历史回归配置，
但远距锥桶召回明显低于 1280。训练尺寸仍保持现有模型的 960 配置。

### 已标定无人机透视校正

推理默认使用像素距离。只有在俯角和当前焦段水平视场角可信时，才同时显式传入两个参数启用透视校正。再提供相机到锥桶地面的垂直高度后，距离单位为米：

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_3class_yolov8s_obb_960/weights/best.pt \
  --source script/test-video/DJI_1m22s_1m36s.mp4 \
  --imgsz 1280 \
  --device 0 \
  --camera-pitch-deg 45 \
  --camera-hfov-deg 84 \
  --drone-altitude-m 30 \
  --output script/runs/infer_videos_perspective45/DJI_1m22s_1m36s
```

## 日志与输出格式

推理结束后，输出目录中会包含以下内容：
- `*_result.mp4`: 渲染了 OBB 边框、Route ROI 区域及告警文本的视频。
- `events.jsonl`: 包含每一帧的检测框、空间分组原始结果、相机外参、时序跟踪细节。
- `alert.jsonl`: 只记录从黄色疑似升级的红色稳态告警，不会每帧重复。
- `alert.log`: 供业务人员直接阅读的纯文本日志，包含发生时间与告警类型（如 `MISSING`、`SHIFT`、`FALLEN`）。

## 核心规则与业务逻辑约束

- **路线识别下限**：有效 ROI 必须包含至少 3 个锥桶才会产生稳态的业务告警，不足 3 个（比如只有 2 个锥桶在画面边缘）会持续绘制 ROI 但抑制误报。
- **动态基准 (Adaptive Reference)**：针对曲线或锥桶摆放不均的场景，系统采用局部最近邻间距的平滑基准，避免全局单一阈值带来的漏报。
- **近距污染保护**：针对无人机飞近、高度下降引起的画面锥桶突然放大，动态阈值做了上限与异常过滤机制，保障检测的稳定性。
- **可信历史更新**：只有已确认且至少包含 5 个锥桶的路线才更新间距基线；锥桶数量减少或成员变化时复用并冻结既有基线。
- **端部边缘保护**：路线最外侧锥桶接近画面边缘时不判断对应端部缺失，路线内部 gap 仍照常分析。
- **合法拐点保护**：SHIFT 检出受支持拐点后对两条臂分别拟合，拐点附近不再跨臂取邻居；同臂内真实偏移仍会告警。
- **两级缺失告警**：3/5 候选先显示黄色 `MISSING?`；强超限连续 5 帧或可信黄色持续 0.8 秒后才升级红色 `MISSING`。
