# 无人机物理 ID 计数与 ROI 稳定化实施计划

## 目标

在无人机视频中，让锥桶计数和可视化 ROI 使用经过相机运动补偿的统一物理锥桶 ID，避免当前 `adaptive_routes -> RouteTracker` 逐帧重分组造成的 ROI 闪动。

本阶段只修正计数和 ROI 绘制路径，不改变 `FALLEN`、`SHIFT`、`MISSING` 的判定规则、告警状态机或告警日志契约。

## 已确认问题

- `traffic_rules._adaptive_route_groups()` 每帧根据当前检测集合和最近邻间距重建路线组；漏检或新增检测会使同一条路线拆分、合并或改变成员。
- `RouteTracker` 跟踪的是路线组，不是单个物理锥桶；`track.roi` 会随着新的路线边界重新平滑。
- 路线未匹配时进入 `held`，只使用预测 ROI；恢复匹配后又切回观测 ROI。
- 路线拆分后的子路线共享 `business_stable_id`，绘制时可能出现同色同编号的竞争 ROI。
- `ConeCounter` 在 `RouteTracker` 之后运行，不能反向导致 ROI 变化。
- `FALLEN` 在路线生成前可能影响 `upright` 集合；`SHIFT` 和 `MISSING` 在路线生成后运行。本阶段不改变这一关系。

## 设计

### 数据流

```text
YOLO 原始锥桶候选
    ├──> UAV 运动估计 + 参考平面物理 ID 跟踪
    │       └──> 计数、物理 ID、稳定 ROI 绘制
    └──> analyze_cones + RouteTracker
            └──> FALLEN / SHIFT / MISSING（本阶段保持原路径）
```

### 改动边界

1. 扩展 `UavPhysicalCounter` 的结果接口，公开已确认/保持中的物理轨迹位置和状态；调用方不得访问其私有 `_states`。
2. 在 `infer_obb.py` 接入物理 ROI；首轮以 `--physical-roi` 验证，通过后提升为视频默认模式，并保留 `--no-physical-roi` 用于 A/B 对比。
3. 实验模式用物理 ID 结果驱动计数和 ROI 绘制；路线跟踪仍运行并写入原有异常诊断字段，但不再驱动物理 ROI。
4. 物理 ROI 使用参考平面坐标和滞回成员管理：新增成员复用物理 ID 的 `3/5` 观察确认，移除成员需连续 8 帧未匹配；运动估计退化时保持上一状态且不创建新 ID。
5. 物理 ROI 的几何边界使用低增益 EMA 更新，短暂漏检保留 `held` 轨迹；显示 ID 使用唯一 `physical_id`，不使用共享的父路线 ID。
6. 在 `events.jsonl` 增加物理跟踪诊断字段，保留原有 `tracked_routes`，以便比较旧路线 ROI 与新物理 ROI。

## 推荐初始参数

```text
confirm_window=5
confirm_votes=3
max_missed_frames=12
match_distance_pixels=55
membership_add_confirm_frames=3
membership_remove_confirm_frames=8
roi_smoothing=0.25
new_id_on_degraded_motion=False
```

## 验收标准

使用以下视频进行旧路径/物理 ID 路径 A/B：

- `script/test-video/test_10s.mp4`
- `script/test-video/test_4m38s-5m_small.mp4`
- `script/test-video/26-08-03_cone.mp4`
- `script/test-video/26-08-07-多角度.mp4` 的代表性片段

每次运行记录并比较：

- `unique_seen_count`
- 每帧活动物理 ID 数量
- ROI 中心位移的 P95 和最大值
- ROI 消失/重建次数
- 重复 ID 次数
- 运动估计退化帧数
- 异常字段是否与默认模式保持一致

验收要求：新模式不引入重复物理 ID；短暂漏检不导致 ROI 消失；异常检测字段和告警日志在未启用新异常策略时保持原有语义。

## 实施顺序

1. 先提交本计划文档。
2. 扩展物理跟踪结果接口并补充单元测试。
3. 接入物理 ROI，首轮不修改默认路径；验证通过后再提升为视频默认模式。
4. 对真实视频运行 A/B，检查事件日志和标注视频。
5. 根据实测结果再决定是否将物理 ROI 设为默认，以及是否继续处理异常检测与物理 ID 的映射。

## 2026-08-27 实施与首轮验证

- 已完成首轮 `--physical-roi` 验证；根据验证结果，物理模式现已成为视频默认路径，`--no-physical-roi` 可回退旧模式。
- 已公开物理轨迹快照，并增加物理 ROI 的 EMA 和 8 帧漏检保持。
- `test_10s.mp4` 使用 331 帧完整推理，输出视频完整解码。
- 物理计数为 36，同行运行的旧图像坐标计数为 41；当前没有逐锥桶人工真值，因此这只能说明减少了 5 次 ID 碎片，不能直接作为绝对计数精度结论。
- 物理 ROI 确认后的消失/重建次数为 0；中心位移 P95 为 22.59 像素、最大值为 69.45 像素。同次运行旧路线 ROI 的最近邻中心位移 P95 为 50.21 像素、最大值为 432.75 像素。
- 物理 ID 同帧重复映射次数为 0；运动估计退化 1 帧，未因此创建新 ID。
- 开启和关闭物理 ROI 的两次运行中，331 帧的检测、路线、FALLEN、SHIFT、MISSING 和告警日志逐项一致。
- 已精简可视化：绿色 OBB 表示当前检测，青色外框表示整体物理 ROI，橙色叉号表示短时漏检但仍保留的 `HOLD` 物理 ID；不再叠加重复的绿色方框和圆圈。

## 2026-08-27 并列路线分组修正

### 问题与约束

- 单 ROI 实现会对全部已确认/保持物理轨迹取总外接矩形，因此主列旁边出现的另一列或小堆锥桶也会被纳入同一个 ROI。
- 仅比较主方向不能解决该问题，因为两条平行锥桶列方向本来就相同。
- 分组必须在运动补偿后的 `map_point` 中完成，并保持独立稳定 `roi_id`；不能直接复用逐帧 `adaptive_routes`，否则会重新引入 ROI 闪动。
- `FALLEN`、`SHIFT`、`MISSING` 继续使用原路线跟踪路径，不读取新的物理 ROI 分组。

### 实现

- 复用 MISSING 几何判定中的主方向、纵向投影和法向残差思想，使用直线支撑候选提取路线组。
- 同组成员同时满足：方向差不超过 20 度、法向残差不超过典型间距的 0.6 倍、相邻纵向间距不超过典型间距的 3.0 倍。
- 每组至少 2 个锥桶，连续 3 帧确认后显示；多个组分别生成 `PHYSICAL ROI <id>`。
- ROI 边界继续使用 0.25 EMA；成员漏检显示保持 8 帧。
- 显示保持与身份记忆分离：ROI 超过 8 帧不满足成组条件后停止绘制，但后台保留身份 30 帧；恢复时通过物理 ID 重叠，或主方向、法向距离和纵向距离匹配复用原 `roi_id`。

### 验证

- 7 个聚焦回归测试通过，覆盖平行列拆分、左侧新组 3 帧确认、短时漏检保持、成员物理 ID 替换后的 ROI ID 恢复和保持到期隐藏。
- `test_10s.mp4` 完整推理 331 帧并完整解码；最终物理累计计数仍为 36。
- 13 帧无物理 ROI、213 帧为 1 个 ROI、105 帧为 2 个 ROI；全片只产生稳定 ROI ID 1 和 2，确认后整体 ROI 只出现一次由 0 到非 0 的初始化过程。
- 与分组前结果比较，逐帧检测、`adaptive_routes`、`tracked_routes`、FALLEN、SHIFT、MISSING 相关字段的摘要完全一致，`alert.jsonl` 和 `alert.log` 逐字节一致。
- 验证输出位于 `script/runs/infer_physical_grouped_v2_20260827/test_10s/`。

## 2026-08-27 双阈值检测与物理 ID 恢复

### 策略边界

- `--conf 0.25` 保持为主检测阈值：只有达到该阈值的锥桶可以创建新物理 ID、为 tentative ID 累积确认票，并进入 FALLEN、SHIFT、MISSING 路径。
- `--physical-match-conf 0.20` 只扩大物理跟踪的候选输入：低于主阈值但不低于恢复阈值的锥桶，只能匹配已经 `confirmed/held` 的物理 ID。
- 未匹配的低阈值候选不创建 ID、不计数、不进入异常检测且不绘制；匹配成功的候选标记为 `physical_recovery_matched`，视频中显示 `RECOVER P<ID>`。
- 同模型 `person` 排除框仍使用主阈值，避免低置信度 person 候选改变原有锥桶后处理结果。

### 回归修正

- 候选匹配同时按“高阈值优先”和距离排序，不再假设全列表按距离单调递增。距离超限时跳过当前候选并继续检查后续候选，避免一个远处高阈值候选阻断近处低阈值恢复候选。
- 新增最小回归场景：一个远处高阈值候选与一个近处低阈值候选同时出现时，前者可创建新 ID，后者仍必须恢复已有 ID。
- 12 个物理 ID、双阈值提取、入口参数和并列 ROI 聚焦测试全部通过。

### `test_10s.mp4` 实测

- 331 帧完整推理并完整解码；共得到 168 个低阈值恢复候选，其中 137 个成功匹配已有物理 ID，31 个未匹配候选没有创建 ID。
- 累计确认物理 ID 从单阈值的 36 降为 31，`held` 轨迹逐帧累计观察数从 801 降为 737、同帧最大 `held` 数从 10 降为 9。该变化说明 ID 碎片减少，但没有逐锥桶人工真值时，不能把 31 直接认定为绝对正确计数。
- 2641 个主阈值检测框、逐帧 `adaptive_routes/tracked_routes` 以及 FALLEN、SHIFT、MISSING 结果与单阈值基线逐项哈希一致；`alert.jsonl` 和 `alert.log` 逐字节一致。
- 物理 ROI 未显示帧数仍为 13，确认后没有整体消失；双阈值会通过更连续的成员观测影响各分组持续时间，这是预期的物理 ROI 路径变化，不会反馈到异常检测路径。
- 默认推理验证输出位于 `script/runs/infer_physical_dual_conf_v2_20260827/test_10s/`；不运行异常检测的独立 A/B 输出位于 `script/runs/uav_physical_dual_conf_20260827/test_10s/`，其物理计数和恢复统计一致。

## 2026-08-27 尺寸过滤目标的物理 ID 放行

- 新增 `--physical-small-cone-conf`，默认值为 `0.30`。尺寸小于当前最小长边但置信度达到该值的候选，只进入物理候选集合，用于创建或匹配物理 ID；普通 route、FALLEN、SHIFT、MISSING 和普通计数仍使用尺寸过滤后的 `cones`。
- 未达到该阈值的小目标继续保持 `too_small`；达到阈值但尚未匹配物理 ID 时不绘制，匹配成功后标记 `physical_small_matched` 并显示物理 ID。
- `test_4m38s-5m_small.mp4` 实测：79 个尺寸过滤候选中 69 个达到 0.30，69 个全部成功匹配已有物理 ID；10 个低于阈值仍被过滤。累计物理计数保持 37，HOLD 观察数由 264 降为 263，异常字段和告警不变。
- 输出位于 `script/runs/infer_physical_small_conf_20260827/test_4m38s-5m_small/`。
