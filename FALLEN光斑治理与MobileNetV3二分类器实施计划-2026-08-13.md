# FALLEN 光斑治理与 MobileNetV3-small 候选真假/姿态分类实施计划

日期：2026-08-13  
状态：执行中。FALLEN 时序收紧、候选挖掘、6530 条人工三分类审核、去重数据集构建和 MobileNetV3-small 首轮训练均已完成；YOLOv8s-OBB 全量训练已启动，分类器阈值校准、视频级融合回放和部署尚未实施。文件名中保留“二分类器”仅为了延续已有审查路径，正文以本版三分类方案为准。

## 1. 目标与边界

本计划处理四个相互关联但必须分开评估的问题：

1. YOLOv8s-OBB 把水面光斑、反光碎片等背景误检为 `cone`。
2. 误检框呈横向或角度不稳定时，被业务规则进一步判为 `FALLEN`。
3. 真实倒伏锥桶朝镜头或沿画面竖直方向倒下时，OBB 长轴仍可能接近竖直，单靠角度会漏报。
4. 旧策略中许多间隔很小的 FALLEN 日志是同一个真实倒伏锥桶因 ID 不稳定被重复激活，不能统称为误报。

最终目标不是只让当前 9 秒片段不告警，而是同时满足：

- 未参与训练的桥下光斑回放中 `FALLEN=0`，cone 误检接近 0。
- 真实倒伏锥桶仍能稳定告警。
- 新航拍小锥桶测试集的 recall 不低于当前 selected 模型的 82%。
- 多角度独立 test 的 cone recall 不低于 92%。
- `MISSING`、`SHIFT` 行为不因本次改造改变。

MobileNetV3-small 只承担 YOLO cone 候选的真假和姿态复核，不替代 YOLOv8s-OBB。它能修正“YOLO 已经检出 cone，但光斑被当成 cone”和“真锥桶的 OBB 角度没有过 FALLEN 阈值”；如果 YOLO 完全没有产生候选框，候选分类器无法找回该目标，这仍需通过检测器数据和训练解决。

## 2. 当前问题证据

固定回放：`/tmp/cone_glare_58_67.mp4`，269 帧，约 9 秒。

- 模型：`script/runs/traffic_cone_uav_20260813_hardneg_s_ft/weights/selected.pt`
- 推理参数：`imgsz=1280`、`conf=0.25`、`iou=0.5`
- 旧 FALLEN 时序策略：16 条 FALLEN 激活。
- 新 7/5 策略：0 条 FALLEN 激活。
- 新策略下仍有 49 帧存在原始 `fallen` 候选，说明检测层误判仍然存在。
- 问题时段共有 652 个 cone 检测，平均置信度约 0.73，最高约 0.97，因此提高 YOLO 置信度阈值不能根治。

现有训练集在原视频约 58.903 秒和 59.458 秒有水面反光画面，但主要是明亮水面上的密集小光点；实际错误集中在约 61.7--63.2 秒的暗水面稀疏拉长光斑，两者不是同一视觉分布。现有 5 张显式 `negative_cone` 又来自约 140--154 秒，未覆盖本次错误核心时段。

## 3. 已实施的 FALLEN 时序策略

默认参数：

| 参数 | 新默认值 | 作用 |
|---|---:|---|
| `fallen_vote_window` | 7 | 使用最近 7 帧观察 |
| `fallen_confirm_votes` | 5 | 至少 5 帧为有效倒伏票 |
| `fallen_match_distance_ratio` | 0.6 | 限制同一稳定锥桶的中心运动 |
| `fallen_max_angle_jump` | 20 度 | 相邻有效倒伏票的 OBB 长轴角度必须连续 |
| `clear_frames` | 3 | 已确认目标连续 3 帧正常后解除 |
| `max_missed_frames` | 5 | 已确认目标短时漏检的保持上限 |
| `reactivation_cooldown_frames` | 30 | 清除后抑制同一目标快速重复激活 |

首次激活还必须满足：

- 观察窗口已经填满，不能在第 3 或第 5 帧提前激活。
- 当前帧本身也是有效倒伏票，不能在已经恢复正常的帧凭历史票数激活。
- 漏检帧按反对票写入最近窗口，不能通过间断出现慢慢凑票。
- 匹配使用局部速度预测；平滑相机运动可以保持 ID，突然跨位置的光斑不能继承旧 ID。
- OBB 长轴按 180 度等价计算角度差，避免 0/180 度表示切换造成假跳变。

当前回归结果：

| 回放 | 旧策略 | 新策略 |
|---|---:|---:|
| 桥下 9 秒片段 FALLEN 激活 | 16 | 0 |
| 完整多角度视频 FALLEN 激活 | 29 | 16 |
| 完整多角度视频 MISSING 激活 | 5 | 5 |
| 完整多角度视频 SHIFT 激活 | 39 | 39 |

完整多角度视频的关键时间已人工对照：新策略减少的主要是 ID 抖动造成的同一真实物理事件重复激活，但 `145.312--145.612s` 存在真实倒伏漏报。该漏报的上游原因是角度条件只间歇命中，不能只通过放宽时序条件来解决。

## 4. 三分类器任务定义

模型：`torchvision.models.mobilenet_v3_small`，ImageNet 预训练，输出三个类别：

- `upright_cone`：真实且业务上未倒伏的锥桶，包括倾斜、局部压扁或斜靠其他物体但主体没有实际躺地的状态。
- `fallen_cone`：真实且主体已经明确侧躺、贴地或翻倒的锥桶，包括朝镜头/背离镜头倒伏、画面中仍近似竖直以及 OBB 角度错误的倒伏。
- `not_cone`：水面光斑、强反光、污渍、标线、浮物、路牌、反光服、石块、阴影等任何被 YOLO 错检成 cone 的背景。

不再使用 `cone / not_cone` 二分类：它只能阻断光斑，不能解决“锥桶已经倒伏，但 OBB 长轴仍接近竖直”的漏报。也不拆出 `glare`、`background` 等多个负类，所有非锥桶候选统一归入 `not_cone`。

以下样本标为 `ignore`，不进入训练：

- 人工也无法确认是否有锥桶。
- 锥桶仅剩几个像素，已没有可辨认外观。
- 裁剪错误，目标大部分落在图外。
- 同时包含多个候选，无法判断当前候选对应哪一个物体。

边界标注以 [`锥桶候选三分类人工标注指南-2026-08-14.md`](./锥桶候选三分类人工标注指南-2026-08-14.md) 为准。核心原则是按画面中可观察到的真实姿态判断，不根据移除支撑物后的假设状态或 OBB 角度猜测；任何身份、目标或姿态不确定的候选均标为 `ignore`。

## 5. 数据必须如何产生

### 5.1 候选驱动，而不是任意裁图

训练数据必须经过与部署一致的 YOLO 候选生成过程：

1. 使用准备部署的 `selected.pt` 对源图片/视频运行 `conf=0.05` 的高召回推理。
2. 对每个 cone OBB 候选记录原始四点、置信度、帧号、视频时间和源文件。
3. 以 OBB 外接矩形中心为中心，裁取候选宽高最大值的 2.0 倍正方形上下文。
4. 超出画面的区域使用固定灰色 letterbox，不拉伸原图。
5. 最终缩放到 224x224，训练与部署必须使用完全相同的裁剪函数。

上下文不能裁得只剩亮点。水面纹理、锥桶底座和周围桥面是分类器区分真假的重要信息。

### 5.2 三类样本来源

现有 OBB 标签只能证明“这是 cone”，不能证明它是 `upright_cone` 还是 `fallen_cone`。因此现有标签只用于自动生成待审裁剪，两种真锥桶姿态都必须由人工打 Tag。OBB 角度可用来排序审核顺序，但不能自动产生姿态真值，否则会把本次需要修复的“角度接近竖直的倒伏”错标成直立。`not_cone` 优先从实际 YOLO 硬误检候选生成，也需人工确认，避免将未标注或小尺寸真锥桶当成负样本。

对 YOLO 漏检的真实锥桶，可以围绕人工 OBB 生成分类裁剪，但必须另外记录 `candidate_source=ground_truth`。这些样本有助于分类器学会困难姿态，但不代表端到端系统能检出它们。

首版可先做一个小而真实的可行性数据集，不必等到几千张倒伏图片：

| 类别 | 最低可行规模 | 更可靠的首版目标 | 关键要求 |
|---|---:|---:|---|
| `fallen_cone` | 150--250 个去重裁剪 | 400--800 | 至少 30 个独立姿态/轨迹，并覆盖所有倒向 |
| `upright_cone` | 800--1500 | 2000--4000 | 包含特殊俯视、倾斜、遮挡和小目标 |
| `not_cone` | 1000--2000 | 3000--6000 | 以 YOLO 实际硬误检为主，不用随机背景凑数 |

这里的“独立”指不同锥桶、不同摆放姿态、不同飞行轨迹或不同拍摄批次。同一锥桶的 500 个相邻视频帧不等于 500 个独立样本。训练时可对少数类采样加权，但不复制文件、不把相邻帧跨 split 分配。

### 5.3 倒伏样本少时的低成本补拍方案

这个场景可以主动构造，不需要等待真实事故发生。使用 3--5 个现场同款锥桶，每个锥桶设置至少 8 个倒向，并专门覆盖：

- 左/右横向倒伏。
- 顺着相机视线朝镜头倒伏、背离镜头倒伏。
- 投影到画面后长轴仍接近竖直的倒伏，这是当前最缺的样本。
- 底座朝上、桶身朝上、局部被杂物或其他锥桶遮挡。
- 锥桶靠近画面边缘被截断，以及远处小锥桶。

拍摄不要只绕一圈。建议在 2--3 个高度、3 个方位、明暗各一组下拍短视频，并配对拍摄同一个锥桶的直立状态。每个独立姿态只保留 3--8 张视角有明显变化的帧，可在一次半天的受控拍摄中得到首版所需的 150--250 张有效倒伏裁剪。

必须将至少一次完整拍摄批次留作 test，另一次留作 val。训练数据只用其余批次。不能将同一段视频随机拆帧到三个 split。

### 5.4 负样本来源

负样本分四组采集：

| 负样本组 | 建议数量 | 来源 |
|---|---:|---|
| 桥下暗水面拉长光斑 | 1000--2000 | 不同日期、太阳角度和飞行方向的独立视频 |
| 普通水面反光 | 800--1500 | 明水面、密集闪点、波纹和不同水色 |
| YOLO 历史硬误检 | 1000--2000 | 所有视频中 `conf>=0.05` 且不匹配真实 OBB 的候选 |
| 通用易混淆背景 | 500--1000 | 污渍、标线、石块、路牌、反光服、浮物和阴影 |

最有价值的负样本是“YOLO 实际给出 cone 框的局部裁剪”，而不是整张没有锥桶的图片。整图空标签适合训练检测器，不等价于候选分类器的局部负样本。

### 5.5 当前桥下失败片段的使用规则

现有 `/tmp/cone_glare_58_67.mp4` 保持锁定，只作为最终回归 test，不能拿去训练分类器，否则 `FALLEN=0` 只能证明记住了测试片段。

需要重新采集至少两次独立桥下飞行：

- 一次作为 train，覆盖进入桥下、桥缝光斑、水面明暗转换。
- 一次作为 val，使用不同时间或阳光角度。
- 当前已知失败片段继续作为固定 test。

如果暂时无法补拍，可以把同视频按连续时间块拆分并留 2 秒 guard，但该结果只能视为开发验证，不能作为跨场景泛化结论。

### 5.6 去重和拆分

拆分单位必须是“源视频/拍摄批次”，不能把同一视频的相邻帧随机分到 train、val、test。

执行以下检查：

- SHA-256 去除完全重复图片。
- 感知哈希或图像 embedding 去除连续帧近重复裁剪。
- 同一个真实锥桶 track 的所有裁剪只能进入一个 split。
- 同一个光斑 track 的所有裁剪只能进入一个 split。
- 输出 provenance CSV，至少包含 `crop_path, class, source, frame, time, polygon, yolo_conf, split, reviewer`。

建议按拍摄批次约 70% / 15% / 15% 拆分，而不是按裁剪数量机械随机拆分。

## 6. 推荐目录与人工审核方式

建议生成以下目录：

```text
dataset/cone_candidate_classifier_202608xx/
  images/
    train/upright_cone/
    train/fallen_cone/
    train/not_cone/
    val/upright_cone/
    val/fallen_cone/
    val/not_cone/
    test/upright_cone/
    test/fallen_cone/
    test/not_cone/
  metadata/
    candidates.csv
    source_groups.json
    duplicates.json
    audit_report.md
  review/
    contact_sheets/
    ignored.csv
```

推荐在 CVAT 中使用“裁剪图分类任务”，设置四个 Tag：`upright_cone`、`fallen_cone`、`not_cone`、`ignore`。每张裁剪图必须且只能选择一个 Tag，不需要再画 OBB。导出后根据 Tag 生成 ImageFolder 目录。

人工审核顺序：

1. 先审高置信度负样本，尤其 `conf>=0.5` 的光斑。
2. 再审真实倒伏锥桶，尤其是 OBB 角度接近竖直的倒伏，防止它们被误归到 `upright_cone` 或 `not_cone`。
3. 最后抽查普通直立锥桶、特殊俯视的直立锥桶和低置信度背景。
4. 每类每个 split 随机生成 contact sheet，人工二次确认。

## 7. MobileNetV3-small 训练方案

建议训练配置：

- 输入：224x224 RGB。
- 初始化：ImageNet 预训练权重。
- 损失和采样：首选普通交叉熵 + WeightedRandomSampler，让每个 batch 能看到足够的 `fallen_cone`。如果采样后仍存在稳定的类别偏置，再改用温和的加权交叉熵；不同时使用强类别权重和过量采样。
- 优化器：AdamW，`lr=3e-4` 训练分类头 3--5 epoch。
- 解冻：先只解冻 MobileNet 最后两个 stage，`lr=1e-5` 训练 10--20 epoch；小数据首版不默认全网络解冻。只有 val 稳定且明显欠拟合时，才试验全量解冻。
- batch：64；显存不足时降到 32。
- early stopping：主要观察 val `fallen_cone` recall、真锥桶被判为 `not_cone` 的比例和桥下光斑拒绝率，不只看 accuracy。
- 导出：ONNX；部署前用同一批裁剪比较 PyTorch 与 ONNX 输出误差。
- 稳定性：test 始终锁定；在 train/val 拍摄批次上做 3 折 grouped cross-validation，每折按拍摄批次分组，并至少运行 3 个随机种子。报告均值和最差折；如果 `fallen_cone` recall 在不同折之间剧烈波动，结论应是补数据，不是挑最好的一次权重。

允许的数据增强：

- 亮度、对比度、饱和度的小幅变化。
- 轻微模糊、JPEG 压缩、噪声。
- 小范围平移和缩放。
- 水平翻转。
- 小角度旋转（建议在 +/-15 度内），用于摄像机滚转和轻微姿态变化。

禁止或谨慎使用：

- 大幅随机裁剪，可能裁掉锥桶底座。
- 强 CutMix/MixUp，可能制造不存在的半透明光斑锥桶。
- 只对负样本做特殊增强，模型可能学习增强痕迹。
- 用 90 度旋转把直立锥桶伪造成倒伏：这会连地面、阴影和透视关系一起旋转，不是真实倒伏分布。
- 使用生成图作为 val/test；如果后续尝试合成数据，它只能作为训练集的低权重辅助，不能替代真实拍摄。

### 7.1 小样本下的主动学习循环

首轮只需训练一个基线，不要急于部署。用它扫描未参与训练的视频，每轮优先送审以下 200--500 个去重候选：

- 三类概率接近、模型最不确定的候选。
- 分类器判为 `fallen_cone`、但 OBB 角度判为正常的候选。
- 分类器判为 `upright_cone`、但 OBB 角度强烈指向倒伏的候选。
- YOLO 置信度很高但分类器判为 `not_cone` 的候选，尤其是水面光斑。
- 新的摄像机高度、光照、背景或锥桶外观中，与已有样本 embedding 距离较大的候选。

人工标注后只将 train/val 来源的新样本加回对应 split，锁定 test 不回流。通常做 2--3 轮这样的定向挖掘，比继续增加大量容易的直立锥桶或相邻帧更有价值。

## 8. 三分类器阈值和角度融合逻辑

不能直接对 softmax 取最大类或直接采用 0.5 阈值。先在独立 val 上分别校准 `P(not_cone)` 和 `P(fallen_cone)`，再按以下优先级选阈值：

1. 真实 cone 被判为 `not_cone` 的比例要足够低，不能为拒绝光斑而误杀真锥桶。
2. 倒伏事件优先保召回；在样本少的首版中，将不确定候选交给时序确认，不要用过高的单帧 `fallen_cone` 阈值直接压掉。
3. 在上述约束内，最大化桥下光斑和其他硬误检的 `not_cone` 拒绝率。

推荐业务链路：

```text
YOLOv8s-OBB cone candidate
  -> 所有 cone 候选都生成 2.0x 上下文裁剪
  -> MobileNetV3-small 输出 not_cone / upright_cone / fallen_cone 三个概率
  -> not_cone 高置信度：拒绝候选，但保留原始记录供审计
  -> 真锥桶候选：先进入稳定物理 ID 跟踪
  -> fallen_cone 达阈值：记一张倒伏票，不要求 OBB 角度先命中
  -> 分类器不确定但 OBB 强倒伏：也记倒伏票，角度作为辅助证据
  -> 分类器明确 upright 且 OBB 也正常：记正常票
  -> 倒伏票经时序确认
  -> 输出 FALLEN 业务告警
```

三分类器必须查看所有 YOLO cone 候选，因为朝镜头倒伏的锥桶可能根本不会进入原始角度 FALLEN 分支。但首次集成必须先用旁路模式：对所有候选运行分类器并记录结果，暂不改变 `MISSING/SHIFT/FALLEN`。完成 A/B 回放和阈值校准后，再先开启姿态票融合；是否让 `not_cone` 拒绝同时影响 `MISSING/SHIFT` 要单独审批。

每帧保留 `yolo_conf`、OBB 角度及角度票、三类原始概率、最终单帧票、稳定物理 ID 和时序决策原因。这样可以分清检测器漏检、姿态分类错误、角度规则错误、ID 抖动和时序过滤错误。

## 9. 验收矩阵

### 9.1 分类器离线指标

- 独立 test 中真实 cone 被判为 `not_cone` 的比例 <= 1%。
- 独立 test `fallen_cone` recall 的最终目标 >= 95%；如果 test 中只有少量独立倒伏姿态，必须同时报告“命中数/总数”和按物理事件的结果，不用一个看似精确的百分比掩盖小样本不确定性。
- 对“朝镜头/背离镜头倒伏”和“画面长轴近似竖直的倒伏”单独报告 recall，这两组不允许被水平倒伏样本的好指标掩盖。
- 锁定桥下光斑 test 的 `not_cone` 拒绝率目标 >= 99.5%。
- 按场景和拍摄批次分别报告，不允许只给总体 accuracy。

### 9.2 检测与告警端到端指标

- 桥下锁定片段：cone FP 显著下降，`FALLEN=0`。
- 至少一段独立桥下视频：`FALLEN=0`。
- 真实倒伏视频：倒伏事件能被报告，同一物理锥桶持续倒伏不因 ID 抖动反复激活。告警延迟必须按实际帧率和最终时序窗口单独报告，不预先锁死为 7 帧。
- 新 supplement test：cone recall >= 82%。
- 多角度 test：cone recall >= 92%。
- 完整 `26-08-07-多角度.mp4`：逐条人工复核 FALLEN 真值，计算事件级 precision/recall。
- 同一 YOLO 权重关闭/开启三分类器做 A/B，确认 `MISSING`、`SHIFT` 没有意外变化。

### 9.3 性能指标

- 记录 YOLO、裁剪、分类器、后处理各阶段耗时。
- 在所有 cone 候选上实测单帧裁剪数、候选分类 batch 耗时及整帧总延迟，根据现场帧率设定验收值；在有实测基线前不预先声称 <= 5 ms/帧。
- 多候选采用单 batch 分类，不能逐候选单独启动推理。

## 10. 分阶段实施与审查点

### 阶段 A：数据生成工具

交付：候选挖掘脚本、统一裁剪函数、CVAT 导入清单、provenance CSV、contact sheet。

审查点 A：随机检查至少 200 张，确认裁剪上下文、正负定义和 ignore 边界正确。

### 阶段 B：补拍与标注

交付：独立 train/val 桥下视频候选，CVAT Tag 标注结果，去重和拆分报告。

审查点 B：确认锁定 test 未参与训练，倒伏样本达到最低可行规模，且“不同倒向/轨迹”而不只是裁剪图数量达到要求。

### 阶段 C：训练与离线选择

交付：每 epoch 指标、场景分组混淆矩阵、PR 曲线、selected 分类器和 ONNX 一致性报告。

审查点 C：达到真锥桶不误杀、倒伏召回和光斑拒绝三项门槛后才能进入业务集成。

### 阶段 D：候选姿态分类旁路集成

交付：对所有 cone 候选运行但暂不改变业务结果的三分类旁路、分阶段耗时、三类原始概率、OBB 辅助票和决策原因。

审查点 D：桥下、真实倒伏和完整多角度三组 A/B 回放通过后才允许部署。

### 阶段 E：部署观察

先旁路运行：分类器只记录决定，不阻断告警。确认一段现场运行后，先开启 `fallen_cone` 姿态票与 OBB 票的融合；`not_cone` 是否真正删除候选要经过另一次审批。保留开关，可立即退回纯 YOLO + 时序策略。当前 7/5 参数在姿态分类器集成前仍视为实验参数，不在本计划审查前继续放宽或直接部署。

## 11. 不建议采用的捷径

- 只提高 YOLO `conf`：高置信度光斑仍会通过，并增加小锥桶漏检。
- 只增加全图空标签：检测器不一定学到候选局部的关键差异。
- 把当前失败片段全部放进训练：会破坏独立验收，容易得到记忆式零误报。
- 给光斑增加一个 YOLO `glare` 类后直接互斥：需要大量完整 OBB 标注，并且新型干扰物仍要扩类。
- 只靠 7/5 时序宣称根治：当前回放告警为零，但原始误检仍存在，持续稳定光斑仍可能通过。

## 12. 本次已完成和待批准事项

已完成：

- FALLEN 7/5 完整窗口确认。
- 漏检反对票、当前帧有效票、角度连续性和中心运动约束。
- 参数可配置，并写入每帧 `fallen_alert_policy`。
- 110 个单元测试通过。
- 桥下回放从 16 条 FALLEN 降为 0。
- 完整多角度回放 FALLEN 从 29 条降为 16，MISSING/SHIFT 不变。
- 实现 `mine_classifier_candidates.py`，生成无标记训练裁剪、带轮廓审核预览和可追溯 CSV。
- 用锁定权重扫描 2066 张图片，生成 6530 个候选，输出于 `dataset/cone_candidate_classifier_20260814/candidates`。
- 实现 Web 筛选页：三个训练类别加 `ignore` 审核结果、前后浏览、当前标注状态和 `review/annotations.csv` 原子持久化。

待审查批准后执行：

- 完成 6530 个候选的人工三分类。
- 实现审核 CSV 到 ImageFolder 的数据生成和审计工具。
- 新的独立桥下 train/val 视频采集与标注。
- MobileNetV3-small 三分类训练、阈值选择和 ONNX 导出。
- 所有 cone 候选的三分类旁路集成，以及后续的分步启用。

## 13. 可执行工单：从当前数据到首轮分类器

以下步骤是首轮实施顺序。`mine_classifier_candidates.py` 和 `candidate_review_web.py` 已实现并完成本轮扫描；`prepare_classifier_dataset.py`、`train_classifier.py` 仍是待实现的工具接口。在人工筛选和数据审计完成前不开始训练。

方法和外部数据的可用性、标签边界和许可风险见
`./交通锥倒伏识别成熟方法与公开数据集调研-2026-08-13.md`。外部数据未完成许可登记和人工姿态重标前，不进入本工单的训练输入。

### 13.1 准备和锁定基线

1. 锁定 YOLO 权重：
   `script/runs/traffic_cone_uav_20260813_hardneg_s_ft/weights/selected.pt`。
2. 锁定数据集：
   `dataset/2026-08-13-merged-3class-hardneg-split`。
3. 锁定推理参数：`imgsz=1280` 、`conf=0.05` 、`iou=0.5`。`conf=0.05` 只用于候选挖掘，不用于产品告警。
4. 当前基线哈希：权重 `974d056f6cdaca63be3885b075425b7e2b9da0e54bbe79526e811e51c7bef76e`，数据集 manifest `09cada50b7b6d1b4596d4e85e45c2f4fd58c260b1670dc1b6828325061440d02`，代码 `0cfeb0aee8528c9943a3941d9a498efaf948fe99`。执行时应重新计算；如与此处不同，将新值写入当轮 `run_manifest.json`，不使用旧哈希伪装同一基线。
5. 在扫描前运行数据集完整性检查：

```bash
conda run -n jiahe python script/validate_dataset.py \
  --data-root dataset/2026-08-13-merged-3class-hardneg-split
```

记录当次权重 SHA-256、数据集 manifest 哈希、代码 git commit 和环境版本，后续所有裁剪、训练、回放都引用这一基线。

### 13.2 扫描全部图片并生成候选

待实现的挖掘工具应支持图片目录和视频，对每个 YOLO `cone` 候选输出：

- 源文件、原始 split、帧号/时间、OBB 四点、外接框、YOLO 置信度。
- 角度规则的原始 `fallen_candidate` 和角度差值。这个字段只作预分组，不作姿态标签。
- 与人工 OBB 的匹配状态：`matched_gt` / `unmatched_gt_candidate` / `no_gt_in_image`。
- 固定裁剪参数和裁剪图路径。

```bash
conda run -n jiahe python script/mine_classifier_candidates.py \
  --weights script/runs/traffic_cone_uav_20260813_hardneg_s_ft/weights/selected.pt \
  --images dataset/2026-08-13-merged-3class-hardneg-split/images \
  --labels dataset/2026-08-13-merged-3class-hardneg-split/labels \
  --imgsz 1280 --conf 0.05 --iou 0.5 \
  --crop-context 2.0 --crop-size 224 \
  --output dataset/cone_candidate_classifier_20260813/candidates
```

输出必须包含 `candidates.csv` 、`crops/` 、`contact_sheets/by_source/` 和 `run_manifest.json`。如果工具不能将 YOLO 候选与人工 OBB 匹配，该轮只能作为候选审核，不能进入训练数据。

### 13.3 按四个候选池人工审核

不按“只有 FALLEN 才审”执行，而是按以下顺序审全部 cone 候选：

1. `fallen_candidate=true` 池：优先标出横向倒伏、光斑和角度规则错判。
2. `fallen_candidate=false` 池：重点找“物理上倒伏、但角度没触发”的样本，这一池不能跳过。
3. `prediction_without_gt` 池：审核高置信度光斑、水面条纹、反光和其他硬误检；同时检查是否存在人工漏标，不得自动标为 `not_cone`。
4. `gt_without_prediction` 池：用人工 OBB 裁剪的真锥桶候选，用于补充姿态分类数据并单独统计 YOLO 漏检。这些裁剪可训分类器，但不得计为端到端可检出样本。

CVAT Tag 只允许 `fallen_cone` / `upright_cone` / `not_cone` / `ignore` 之一。审核人不确定时必须选 `ignore`，不强行投票。每类每个 split 至少随机抽查 100 张；`fallen_cone` 必须全量复核。

### 13.4 生成 ImageFolder 和数据审计

CVAT 导出后由待实现的数据准备工具根据 Tag 复制或硬链到：

```text
dataset/cone_candidate_classifier_20260813/images/{train,val,test}/{fallen_cone,upright_cone,not_cone}/
```

```bash
conda run -n jiahe python script/prepare_classifier_dataset.py \
  --candidates dataset/cone_candidate_classifier_20260813/candidates/candidates.csv \
  --annotations dataset/cone_candidate_classifier_20260813/cvat_export \
  --preserve-source-split \
  --output dataset/cone_candidate_classifier_20260813
```

它必须保留原候选的 `source_image` 和 `source_split`，并报告：各类数量、独立源批次数、SHA-256 重复、感知重复、split 跨界 track、`fallen_cone` 姿态覆盖表。任一个 track 跨 split 或锁定 test 回流，该轮数据作废重来。

### 13.5 训练、选模型和回放

待实现的训练工具首轮执行：

```bash
conda run -n jiahe python script/train_classifier.py \
  --data dataset/cone_candidate_classifier_20260813 \
  --model mobilenet_v3_small --pretrained \
  --image-size 224 --batch 64 --head-epochs 5 --finetune-epochs 15 \
  --sampler weighted --seed 20260813 \
  --output script/runs/cone_candidate_mobilenetv3_20260813
```

选权重不看单一 accuracy，必须同时输出三类混淆矩阵、按场景报告、`fallen_cone` 召回、真锥桶被误拒绝率、桥下光斑拒绝率、PR 曲线和 PyTorch/ONNX 误差。首轮权重只能进入旁路回放，不能自动替换 YOLO 部署权重。

### 13.6 旁路和分步启用

1. 旁路：分类器扫描所有 YOLO cone 候选，只写入 `events.jsonl`，不改任何告警。
2. 回放：桥下锁定片段、独立桥下视频、`26-08-07-多角度.mp4`、真实倒伏视频各做关闭/开启 A/B。
3. 先启用 `fallen_cone` 姿态票与 OBB 票的融合，再单独评估 `not_cone` 是否可以影响 `MISSING/SHIFT`。
4. 出现倒伏漏报、真锥桶被拒绝、延迟超标或误杀率上升时，立即回退旁路模式，保留全量输出供下一轮主动学习。
