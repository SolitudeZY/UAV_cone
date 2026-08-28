# 交通导改 OBB 训练与推理

当前主方案面向 RTX 5060 Ti，使用单个 `YOLOv8s-OBB` 三分类模型：`cone`、`traffic_sign`、`person`。`person` 用于抑制人体区域内的锥桶误检，业务不需要单独输出 `vest`。交通锥原始四点 OBB 保留不变，OBB 四角可直接计算中心点和长轴角度，无需另外标中心点。

## 1. 环境

在项目根目录执行：

```bash
conda env create -f script/environment.yml
conda activate yolo-obb
```

## 2. 构建融合数据集

输入：

- `dataset/data-26-07-14/`：交通锥和交通标志 OBB 数据。
- `dataset/person/PPE-supplement/20260612/`：PPE AABB 数据，只使用 person 和 vest。

构建命令：

```bash
python script/build_fused_dataset.py
```

输出到 `dataset/2026-07-15/`：

- `images/{train,val,test}/`：源图片的硬链接，不重复占用图片磁盘空间。
- `labels/{train,val,test}/*.txt`：统一的四点 OBB 标签。
- `data.yaml`：Ultralytics 数据配置。
- `manifest.json`：源路径、类别映射、图片数和目标数。

训练集中的交通锥图片默认重复 3 份，以缓解 PPE 图片数量更多造成的类别偏置；验证集和测试集不重复。该步骤生成历史四分类融合集，构建脚本不会修改源数据，并拒绝覆盖非空输出目录。

从审核后的四分类数据生成三分类分支：

```bash
python script/convert_obb_classes.py \
  --source dataset/2026-08-04-long-neg47 \
  --output dataset/2026-08-04-long-neg47-3class
```

该工具保留图片和 train/val/test 划分，只删除类别 `3 (vest)` 的标签行，并在
`manifest.json` 中记录转换前后对象数与来源摘要。它不会覆盖非空目录。

## 3. 数据校验与训练

当前默认训练集是 `dataset/2026-08-05-merged-3class/`，类别为
`cone / traffic_sign / person`。它在三分类基线上加入 2026-08-05 审核通过的
174 张纯负样本和 43 张正负补充图；新增数据只进入 train，val/test 原样保留。
审核清单和合并统计记录在 `dataset/audit_previews/26-08-05_merge_review.md`
及合并集的 `manifest.json` 中。

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

`train_obb.py` 的默认参数已是上述融合数据、`YOLOv8s-OBB` 和 960 尺寸，因此也可直接执行 `python script/train_obb.py --device 0 --cache`。

训练输出在 `script/runs/traffic_cone_3class_yolov8s_obb_960/`：`weights/best.pt` 是验证集最优权重，`weights/last.pt` 是最后一轮检查点，`results.csv` 和 `results.png` 是训练曲线，混淆矩阵和验证预览图用于检查各类别效果。

断点续训：

```bash
python script/train_obb.py \
  --resume script/runs/traffic_cone_3class_yolov8s_obb_960/weights/last.pt \
  --device 0
```

## 4. 图片或视频推理

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_3class_yolov8s_obb_960/weights/best.pt \
  --source script/test-video/test_10s.mp4 \
  --imgsz 960 \
  --device 0
```

`--source` 可输入单张图片、图片目录、视频、`rtsp://...` 推流地址或摄像头编号 `0`。结果默认保存在 `script/runs/infer/`：图片输出为 `*_result.jpg`，视频输出为 `*_result.mp4`，逐帧结构化结果为 `events.jsonl`。

视频推理默认启用运动补偿后的物理锥桶 ID 计数和稳定 ROI。该模式只替换计数与 ROI 绘制的数据源，`FALLEN / SHIFT / MISSING` 仍使用原有路线跟踪和告警状态机：

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_uav_20260814_no_tiny_s_full_1280/weights/best.pt \
  --source script/test-video/test_10s.mp4 \
  --imgsz 1280 \
  --device 0 \
  --output script/runs/infer_physical_roi/test_10s
```

默认物理跟踪使用最近 5 帧至少 3 次观察确认、最多保持 12 个漏检帧和 55 像素参考平面匹配距离。检测采用双阈值：`--conf 0.25` 是新建物理 ID、确认 tentative ID 和进入 `FALLEN / SHIFT / MISSING` 的高阈值；`--physical-match-conf 0.20` 是低阈值，只允许重新匹配已经 `confirmed/held` 的物理 ID。未匹配的低阈值候选不会创建 ID、不会计数、不会进入异常检测，也不会绘制；成功恢复时显示 `RECOVER P<ID>`。`physical-match-conf` 必须小于或等于 `conf`。

物理 ROI 在运动补偿后的参考平面中按路线分组：候选成员需要具有相近主方向，同时满足法向距离不超过典型间距的 `0.6` 倍；沿主方向排序后，相邻锥桶距离超过组内最近邻间距中位数的 `2.5` 倍会强制拆组。稳定 ROI 重新关联时，共享物理 ID 在参考平面中的位移也必须通过相同的横向/纵向距离限制。因此平行分离的锥桶列和同方向但距离很远的两批锥桶都不会合并。每组至少 2 个锥桶并连续 3 帧确认，边界使用 `0.25` EMA，漏检显示保持 8 帧，后台 `roi_id` 身份记忆 30 帧。需要 A/B 回归时传入 `--no-physical-roi`，可回退原有图像坐标计数和路线 ROI。

只比较旧图像坐标计数和新物理 ID 计数、不运行异常检测时，可使用独立入口；它使用相同的 `0.25/0.20` 双阈值策略：

```bash
python script/uav_physical_cone_counter.py \
  --weights script/runs/traffic_cone_uav_20260814_no_tiny_s_full_1280/weights/best.pt \
  --source script/test-video/test_10s.mp4 \
  --imgsz 1280 --device 0 \
  --output script/runs/uav_physical_dual_conf/test_10s
```

标注视频中的绿色 OBB 是当前帧模型检测框，`P<ID>` 是统一物理 ID，青色 `PHYSICAL ROI <id>` 是一条物理路线的稳定区域；并列路线分别显示不同的 ROI ID。当前帧漏检但仍在保持窗口内的物理 ID 使用橙色叉号和 `HOLD` 标记，不再额外绘制重复的绿色方框或圆圈。分组参数可用 `--physical-roi-min-cones`、`--physical-roi-confirm-frames`、`--physical-roi-lateral-ratio`、`--physical-roi-link-ratio`、`--physical-roi-max-angle` 和 `--physical-roi-identity-memory-frames` 调整，实际值会写入 `run_config.json`。

三分类模型默认使用同一次前向中的 `person` 框抑制人体区域内的假锥桶，不需要二阶段人员模型，也不会产生第二次完整推理的 FPS 开销。需要关闭时传入 `--exclude-classes ''`。使用历史四分类权重时可显式传入 `--exclude-classes person,vest`。

过小锥桶默认在空间和时序规则之前排除：`--min-cone-long-side-ratio 0.022222`
表示 OBB 长边必须至少达到画面短边的约 2.22%，即 1920x1080 画面为 24px、
1280x720 画面为 16px。被排除目标仍以灰色 `SMALL` 框显示，但不进入
route、MISSING、SHIFT、FALLEN 或业务告警，也不计入 `cone_count`。原始检测保留在
`events.jsonl.detections`，标记为 `postprocess_status: too_small`；每帧排除数量记录在
`small_cone_suppressed`。传入 `--min-cone-long-side-ratio 0` 可关闭该过滤。

物理 ID 默认对尺寸过滤增加一条独立放行规则：`--physical-small-cone-conf 0.30`
表示尺寸不足但置信度达到 0.30 的候选可以只进入物理 ID 跟踪；它们仍不进入
`FALLEN / SHIFT / MISSING`、普通 route 或普通计数。未匹配的小目标继续保持
`too_small` 并不绘制，匹配成功后标记为 `physical_small_matched`。该阈值必须不低于
`--conf`，用于避免用弱小框创建新物理目标。

视频推理还会对过滤后的有效锥桶做单目标时序计数。默认最近 5 帧至少命中 3 帧后才确认一个锥桶，确认后的轨迹最多保持 5 个漏检帧；短时漏检恢复时沿用原稳定 ID，不会重复累计。可按视频帧率调整：

```text
--cone-count-confirm-window 5
--cone-count-confirm-votes 3
--cone-count-max-missed-frames 5
--cone-count-match-distance-ratio 0.8
```

`events.jsonl` 中保留原始逐帧 `cone_count`，并新增 `stable_visible_count`、`unique_seen_count`、`new_cone_ids`、`active_cone_ids`、`held_cone_ids`、`stable_cone_ids` 和 `tracked_cones`；推理结束另写 `cone_count.json` 保存累计结果。其中 `unique_seen_count` 是本视频累计确认过的稳定锥桶数，标注视频显示稳定 ID 以及 `CONE visible=<n> total=<n>`。累计数是基于稳定轨迹的确认实例数；无人机大幅运动且没有可靠地面/全局运动补偿时，不应把它解释为物理锥桶的绝对总数。

MobileNetV3-small 可直接接入完整业务推理。新权重使用 `1.6x` 候选上下文，阈值文件
默认从权重同目录的 `thresholds.json` 读取：

```bash
PYTHONPATH=script python script/infer_obb.py \
  --weights script/runs/traffic_cone_uav_20260814_no_tiny_s_full_1280/weights/best.pt \
  --source script/test-video/26-08-07-多角度.mp4 \
  --imgsz 1280 --device 0 \
  --classifier-weights script/runs/mobilenetv3_small_cone_20260817_v2_context16_bp075/best.pt \
  --classifier-mode full --classifier-crop-context 1.6 \
  --output script/runs/infer_obb_mobilenet/full
```

- `--classifier-mode audit`：记录三类概率和决策，不改变任何业务结果。
- `--classifier-mode fallen`：`fallen_cone` 作为额外 FALLEN 票，仍需通过原 7 帧/5 票时序；`not_cone` 只记录。
- `--classifier-mode full`：在 `fallen` 基础上，将达到校准阈值的 `not_cone` 在 route 生成前剔除，因此会实际影响 MISSING、SHIFT 和 FALLEN。视频用橙色 `REJECT` 框保留这些原始 YOLO 候选供复核。
- 不传 `--classifier-weights` 时保持原 YOLO + 规则链路。`events.jsonl` 中的 `classifier_policy`、`classifier_counts`、每个 detection 的 `classifier` 和 `classifier_timing_ms` 可用于 A/B 审计。

视频推理会先按当前帧生成空间 route，再进行 route 级时序跟踪。每个已确认的锥桶列拥有稳定 ID、独立丢失计数和经过平滑的 ROI；短暂漏检时显示 `AUTO ROI <id> HOLD`，重新连续匹配后恢复。历史 route 只能辅助拆分本帧已经连在一起的组，不能跨越本帧不连通的空间组进行合并。

推理默认根据锥桶最近邻间距生成多个自适应 ROI。相距较远的锥桶列会分别排序、连线和计算告警，不会跨列连接。`--route-link-ratio` 默认为 `3.0`；调小会更积极地拆分锥桶列，调大则允许连接更宽的缺失空档。仍可用像素 ROI 先限定需要分析的区域：

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_3class_yolov8s_obb_960/weights/best.pt \
  --source script/test-video/test_10s.mp4 \
  --roi 100,200,1800,1000 \
  --device 0
```

无人机巡航场景的时序参数默认值如下：

- `--route-confirm-frames 3`：新 route 连续出现 3 帧才确认，抑制单帧误检 ROI。
- `--route-recovery-frames 2`：保持中的 route 连续匹配 2 帧后恢复业务告警。
- `--roi-memory-frames 5`：每条已确认 route 独立保留 5 个漏检帧；其他 route 仍可见时不会重置它。
- `--route-match-ratio 2.5`：候选 route 与预测 route 的匹配距离上限，相对于局部锥距计算。
- `--route-max-angle 45`：匹配允许的 route 主方向最大差值，单位为度。
- `--roi-smoothing 0.35`：缓慢运动时的 ROI 观测权重，越小越平稳但跟随更慢。
- `--roi-max-smoothing 0.75`：快速运动或预测误差较大时的最大观测权重。
- `--no-temporal-roi`：关闭时序 route 跟踪，退回逐帧自适应 ROI，适合对照诊断。

每个稳定 ROI 会独立学习可信的正常间距历史。只有路线已确认且当前至少包含足够多的锥桶时才更新；锥桶刚进入、离开或数量较少时冻结已学基线，避免用不完整视野重新估算：

- `--gap-history-frames 30`：每个 ROI 最多保留 30 个高支持度帧的正常 gap 中位数。
- `--gap-history-min-frames 3`：至少 3 个可信历史帧后才允许激活缺失告警。
- `--gap-history-min-cones 5`：当前路线至少有 5 个锥桶才允许更新间距历史；少量锥桶仍可复用此前已建立的基线，但不能自行建立新基线。
- `--gap-close-ratio 0.55`：当前参考低于历史基准的 55% 时视为近距污染，不写入历史。
- `--gap-max-drop-ratio 0.05`：历史基准每帧最多降低 5%，既防止突降，也允许无人机飞高或变焦后逐步适应。
- `--gap-min-pixels 0`：仅旧像素模式使用的固定安全下限；透视校正模式不会把像素值混入地面距离。
- `--gap-vote-window 5` 和 `--gap-confirm-votes 3`：同一个稳定 gap 在最近 5 帧内至少 3 帧为候选后进入黄色 `MISSING?`。
- `--gap-alert-ratio 2.2` 和 `--gap-alert-confirm-frames 5`：可信强超限连续 5 帧后升级红色 `MISSING`。
- `--gap-suspect-seconds 0.8`：可信黄色持续 0.8 秒后升级红色，程序使用视频 FPS 换算帧数。
- `--gap-max-ratio-disagreement 0.6`：透视倍率与像素倍率冲突时仅保留黄色，不升级红色。
- `--gap-clear-frames 3`：已确认 gap 连续 3 帧正常后解除告警，避免阈值附近反复闪烁。
- `--gap-membership-cooldown 5`：锥桶进入或离开 route 后，相邻 gap 暂停 5 帧；参数只允许 3-5 帧。
- `--min-alert-cones 3`：当前 ROI 少于 3 个锥桶时不产生稳定业务告警。`--min-route-cones` 仍默认为 2，因此两锥桶 ROI 会继续画框展示。
- `--missing-edge-margin 0.02`：路线最外侧锥桶进入画面边缘 2% 时，对应端部 gap 不判断缺失；路线内部 gap 不受影响。

SHIFT 对受支持拐点做独立豁免，避免车辆周边 L 形/直角围挡被当成锥桶移位：

- `--shift-neighbor-count 2`：SHIFT 只使用候选点左右各 2 个最近邻拟合局部路线，避免弯曲路线的远端点拉偏结果。
- `--shift-corner-min-arm-cones 2`：拐点两侧各至少有 2 个锥桶。
- `--shift-corner-min-angle 45`：两臂无方向夹角至少 45 度。
- `--shift-corner-max-arm-residual-ratio 0.25`：每条臂的最大直线残差不超过路线中位锥距的 25%。
- `--shift-corner-max-vertex-distance-ratio 0.45`：角点到任一臂延长线的距离不超过中位锥距的 45%。
- `--shift-corner-protect-neighbors 1`：确认拐点后，跳过角点及左右各 1 个锥桶的 SHIFT；其他点只使用同臂邻居拟合。
- `--no-shift-corner-exemption`：关闭拐点豁免，用于 A/B 回放。
- `--isolated-endpoint-gap-ratio 2.5`：路线端点与其余锥桶列相隔超过内部中位间距 2.5 倍时，将端点标记为 `ISOLATED`，保留检测框但不纳入 ROI 和几何规则。
- `--shift-confirm-frames 3`：同一稳定 ROI 连续 3 帧出现 SHIFT 后才画红并写入业务告警；原始逐帧 `displaced` 仍保留。
- `--shift-membership-stable-frames 10`：ROI 成员数变化后稳定 10 帧才重新累计 SHIFT，排除锥桶进出画面造成的短暂几何重排。

FALLEN 使用单锥桶稳定 ID 做独立时序确认，默认参数如下：

- `--fallen-vote-window 7` 和 `--fallen-confirm-votes 5`：观察满最近 7 帧且至少 5 帧为倒伏，当前帧仍为倒伏时才首次激活；漏检帧按反对票处理。
- `--fallen-match-distance-ratio 0.6`：用局部锥距的 60% 限制同一锥桶的中心运动，并使用局部速度预测跟随平滑相机运动。
- `--fallen-max-angle-jump 20`：相邻有效倒伏票的 OBB 长轴角度变化不超过 20 度，角度按 180 度等价计算。
- `--fallen-reactivation-cooldown-frames 30`：告警解除后，同一稳定锥桶 30 帧内不重复激活。

消融实验可以显式退化或关闭 FALLEN 时序：`--fallen-vote-window 1 --fallen-confirm-votes 1 --fallen-clear-frames 1 --fallen-max-missed-frames 0 --fallen-reactivation-cooldown-frames 0` 表示逐帧基线；完整策略使用默认的 `7/5/3/5/30`。`--no-fallen-missing-suppression` 用于关闭“已确认倒伏锥桶占据预期位置时抑制 MISSING”的跨异常互斥规则。

上述策略只收紧稳定 `FALLEN` 告警，原始逐帧 `fallen` 仍保留在 `events.jsonl`，实际参数记录在 `fallen_alert_policy`。SHIFT 拐点逻辑不修改 `MISSING/FALLEN` 的空间判定。直线中单个锥桶横移时，两侧臂方向仍近似平行，
不会满足拐点角度条件，因此仍会报告 SHIFT。

历史基准使用高支持度帧中正常 gap 的中位数，不会永久取历史最大值。原始规则已判定为疑似缺失的异常大 gap、边缘端部 gap 和异常近距 gap 都不会反向污染基线。持续发生尺度变化时基线仍可缓慢适应。

例如，使用最终稳定配置推理两个测试视频：

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_3class_yolov8s_obb_960/weights/best.pt \
  --source script/test-video/test_10s.mp4 \
  --imgsz 960 \
  --device 0 \
  --output script/runs/infer_videos_gap_stable/test_10s

python script/infer_obb.py \
  --weights script/runs/traffic_cone_3class_yolov8s_obb_960/weights/best.pt \
  --source script/test-video/DJI_1m22s_1m36s.mp4 \
  --imgsz 960 \
  --device 0 \
  --output script/runs/infer_videos_gap_stable/DJI_1m22s_1m36s
```

输出目录包含标注视频 `*_result.mp4`、逐帧诊断 `events.jsonl`、锥桶累计结果 `cone_count.json`、本次运行完整参数与权重/视频 SHA-256 的 `run_config.json`、结构化告警激活记录 `alert.jsonl` 和人工复查日志 `alert.log`。同一告警持续多帧时，后两者只在激活瞬间写一条，不会每帧重复。`alert.log` 示例：

```text
1\ttime=2.500s\troi=ROI_4\ttype=MISSING
```

时间是视频播放秒数，不是推理机器的系统时间。`events.jsonl` 中：

- `adaptive_routes` 是当前帧的原始空间分组，包含 `route_id`、`stable_route_id`、原始 ROI 和成员中心点。
- `tracked_routes` 是时序结果，包含稳定 ID、`tentative/confirmed/held` 状态、平滑 ROI、原始 ROI、速度、命中数、丢失帧数和 `alert_ready`。
- `physical_roi.rois` 是默认物理模式的分组结果，包含稳定 `roi_id`、成员物理 ID、保持成员、参考平面主方向、状态和图像边界；它只用于计数与绘制，不改变异常检测路线。
- `classifier_counts.physical_small_candidates` 和 `physical_small_matched` 分别记录尺寸过滤但达到放行阈值的小目标数，以及其中已绑定物理 ID 的数量。
- `gap_thresholds` 按稳定 ROI 输出历史基准、当前帧均值、历史均值、距离单位、置信度、被排除的近距样本和阈值来源；透视模式使用 `*_corrected` 字段，旧像素模式使用 `*_pixels` 字段。
- `tracked_gaps` 包含稳定 gap ID、`normal/suspect/alert` 状态、倍率、投票、持续帧数和测量冲突原因。
- `missing_edge_ignored` 列出因路线外端点接近画面边缘而停止判断的 gap 起始索引。
- `shift_corner_exemptions` 记录拐点索引、受保护索引、两臂支撑索引、夹角、直线残差比例和角点距离比例。
- `isolated` 记录未纳入 ROI 和几何规则的端部孤立点索引；`shift_alert_policy` 记录 SHIFT 连续帧确认参数。
- `missing`、`displaced`、`fallen` 保留原始逐帧告警，兼容既有分析流程。
- `suspect_missing` 是黄色疑似，`stable_missing` 是红色业务告警；`stable_displaced`、`stable_fallen` 保持原语义。

包含近距污染保护的最新测试输出位于 `script/runs/infer_videos_gap_stable/test_10s/` 和 `script/runs/infer_videos_gap_stable/DJI_1m22s_1m36s/`。

旧的两分类锥桶权重仍可选用外置 person 模型抑制反光服假锥桶：

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_obb_960/weights/best.pt \
  --source script/test-video/test_10s.mp4 \
  --imgsz 960 \
  --person-weights yolo26n.pt \
  --person-classes 0 \
  --device 0
```

如使用项目 PPE 权重，person 和 vest 类为 `0,2`；当前测试视频中的边缘截断人员仍建议使用通用 `yolo26n.pt`。

规则默认值：

- 自适应 ROI：中心距离不超过局部最近邻间距的 `3.0` 倍时归入同一序列。
- 缺失：每个序列内局部相邻间距大于邻域中位数的 `1.8` 倍。
- 偏移：有标定时超过 `0.5m`；未标定时超过中位锥距的 `25%`。
- 倒伏：完整 OBB 长轴相对画面竖直方向偏转至少 `30` 度。
- 边缘保护：画面边缘 `5%` 内且角度失真的框标为 `EDGE SKIP`，不参与空间规则。

推理默认使用像素距离。仅当同时显式提供可信的光轴俯角和当前焦段水平视场角时，才启用针孔相机平地投影：

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_3class_yolov8s_obb_960/weights/best.pt \
  --source script/test-video/DJI_1m22s_1m36s.mp4 \
  --camera-pitch-deg 45 \
  --camera-hfov-deg 84
```

未提供高度时，校正距离单位为 `camera_heights`（相机离地高度的倍数），足以进行缺失间距比例和相对偏移判断，但不是米。提供相机离锥桶所在平面的垂直高度后才输出米制距离：

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_3class_yolov8s_obb_960/weights/best.pt \
  --source script/test-video/DJI_1m22s_1m36s.mp4 \
  --camera-pitch-deg 45 \
  --camera-hfov-deg 84 \
  --drone-altitude-m 30
```

`events.jsonl` 同时保留 `gap_pixels`/`distance_pixels` 和 `gap_corrected`/`distance_corrected`，并在 `perspective_calibration` 中记录俯角、视场角、高度、像素焦距和单位。

旧的局部像素标定仍可用于小范围固定机位；显式传入 `--meters-per-pixel` 时会使用旧模式而不叠加透视校正。例如地面上已知长度为 10m 的线段在图中为 250px：

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_obb_960/weights/best.pt \
  --source script/test-video/test_10s.mp4 \
  --meters-per-pixel 0.04
```

无人机画面需保持云台滚转稳定。透视校正假设锥桶位于同一水平地面；地面有明显坡度、相机存在滚转、画面被裁切或实际视场角未知时，保持默认像素模式，或使用实测内外参和地面单应性矩阵。

## 5. 候选分类数据筛选

使用当前 YOLOv8s-OBB 以低阈值扫描全部图片，生成两倍正方形上下文裁剪。无标记裁剪供后续训练，带 OBB 轮廓的预览图只供人工审核：

```bash
PYTHONPATH=script conda run -n jiahe python script/mine_classifier_candidates.py \
  --weights script/runs/traffic_cone_uav_20260813_hardneg_s_ft/weights/selected.pt \
  --images dataset/2026-08-13-merged-3class-hardneg-split/images \
  --labels dataset/2026-08-13-merged-3class-hardneg-split/labels \
  --imgsz 1280 --conf 0.05 --iou 0.5 --match-iou 0.5 \
  --device 0 --batch 1 --crop-context 2.0 --crop-size 224 \
  --output dataset/cone_candidate_classifier_20260814/candidates
```

2026-08-14 尺寸审计确认 `data-26-07-14` 中 4 张 `RAW_1` 帧的全部锥桶
在 1920x1080 原图上的 OBB 长边均小于 24px，超出当前业务检测范围。清理后的
YOLO 数据集为 `dataset/2026-08-14-merged-3class-no-tiny-raw1/`：从 train
完整排除这些帧的 `r0/r1/r2` 共 12 份图片及标签，val/test 保持不变。排除计划
记录在 `dataset/audit_previews/2026-08-14_tiny_raw1_exclusion.json`；对应的 243 个
分类器候选全部标为 `ignore`，不得进入 MobileNet 的 train/val/test。

启动候选筛选页：

```bash
PYTHONPATH=script conda run -n jiahe python script/candidate_review_web.py \
  --data-root dataset/cone_candidate_classifier_20260814/candidates \
  --host 0.0.0.0 --port 8765
```

页面提供 `upright_cone / fallen_cone / not_cone` 三个训练类别和一个
`ignore` 审核结果，点击后自动保存并进入下一张。`ignore` 用于人眼无法确认目标或姿态的候选，后续不得进入训练、验证或测试集。顶部“标注状态”可切换未标注、全部状态、倒伏、直立、非锥桶和忽略；桌面端也可直接点击右侧对应类别的数量进入该类别。上一张/下一张只在当前标注状态、候选池和 split 的交集内切换，便于逐类复查和修正。页面会显示每张图片现有的标注类别，再次点击类别即可修正旧标注。标注保存在 `dataset/cone_candidate_classifier_20260814/candidates/review/annotations.csv`；写入使用临时文件原子替换，刷新或重启服务不会丢失已有进度。

斜立、模糊、错误框、相邻双锥桶和画面边缘截断等边界情况统一按照项目根目录的
[`锥桶候选三分类人工标注指南-2026-08-14.md`](../锥桶候选三分类人工标注指南-2026-08-14.md) 执行。

审核完成后，构建按原 `source_split` 隔离、按裁剪图 SHA-256 去重的
ImageFolder 数据集。`ignore` 会被完全排除；标签冲突、完全相同裁剪跨 split、
完全相同源图跨 split 均会使构建失败：

```bash
PYTHONPATH=script conda run -n jiahe python script/build_cone_classifier_dataset.py \
  --candidates-root dataset/cone_candidate_classifier_20260814/candidates \
  --output dataset/cone_posture_classifier_20260814
```

产物中的 `metadata/samples.csv` 记录每个训练样本的来源、原候选、类别、
crop/source SHA-256 和感知哈希；`metadata/manifest.json` 记录审核数量、去重数量和
各 split 类别数。模型输入使用 `crop_path` 对应的无框 224x224 图片，带蓝/黄框的
`preview_path` 只用于人工审核，禁止用于训练。

MobileNetV3-small 先训练分类头 4 epoch，再只解冻最后 4 个 feature block 微调。
平方根加权采样用于缓和类别不均衡，模型按验证集 macro-F1、倒伏召回、非锥桶召回、
accuracy 的顺序选 `best.pt`，并只在选定权重后运行独立 test：

```bash
PYTHONPATH=script conda run -n jiahe python script/train_mobilenet_classifier.py \
  --data dataset/cone_posture_classifier_20260814 \
  --output script/runs/mobilenetv3_small_cone_20260814 \
  --device 0 --batch 64 --workers 8
```

复评任意锁定 split 或批量输出三类概率：

```bash
PYTHONPATH=script conda run -n jiahe python script/evaluate_mobilenet_classifier.py \
  --weights script/runs/mobilenetv3_small_cone_20260814/best.pt \
  --data dataset/cone_posture_classifier_20260814/test \
  --output script/runs/mobilenetv3_small_cone_20260814/test_metrics_recheck.json

PYTHONPATH=script conda run -n jiahe python script/infer_mobilenet_classifier.py \
  --weights script/runs/mobilenetv3_small_cone_20260814/best.pt \
  --source dataset/cone_posture_classifier_20260814/test \
  --output script/runs/mobilenetv3_small_cone_20260814/test_predictions.csv
```

YOLO 检测器的本轮全量微调从官方 `yolov8s-obb.pt` 初始化，不从历史
`selected.pt` resume。使用清除极小 RAW_1 样本后的完整数据集，且不冻结骨干：

```bash
conda run -n yolo-obb python script/train_obb.py \
  --data dataset/2026-08-14-merged-3class-no-tiny-raw1/data.yaml \
  --model yolov8s-obb.pt --device 0 --epochs 150 --imgsz 1280 --batch 1 \
  --workers 8 --patience 30 --optimizer AdamW --lr0 0.001 --lrf 0.01 \
  --warmup-epochs 3 --freeze 0 --save-period 5 \
  --name traffic_cone_uav_20260814_no_tiny_s_full_1280
```

## 6. 独立 PPE/person 抑制模型（兼容旧权重）

数据位于 `dataset/person/PPE-supplement/20260612`。训练脚本会自动把源 YAML 中失效的 `/mnt/data/...` 改写到运行时配置，不修改原 YAML。

```bash
python script/train_ppe.py \
  --device 0 \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --name ppe_person_detector
```

最佳权重为 `script/runs/ppe_person_detector/weights/best.pt`。使用方式：

```bash
python script/infer_obb.py \
  --weights script/runs/traffic_cone_obb_960/weights/best.pt \
  --source script/test-video/test_10s.mp4 \
  --imgsz 960 \
  --person-weights script/runs/ppe_person_detector/weights/best.pt \
  --person-classes 0,2 \
  --device 0
```

## 数据注意事项

融合数据集共 1021 张训练图、231 张验证图、153 张测试图。交通标志牌全部划分合计仍只有 18 个目标（训练重复后统计），样本远不足以可靠训练；现阶段模型评估应以锥桶、人员和反光服为主，后续需要补充标志牌的正面、侧面、遮挡、尺度和光照样本。
