#!/usr/bin/env python3
"""Build auditable tables and charts from the three cone ablation replays."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "script" / "runs" / "ablation_patent_summary_20260825"


def load_summary(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def by_group(summary: dict[str, object], group: str, video: str | None = None) -> dict[str, object]:
    runs = summary["runs"]
    for item in runs:
        output = Path(item["output"])
        if (output.name == video if video is not None else output.name == group) and (
            output.parent.name == group if video is not None else True
        ):
            return item
    raise KeyError((group, video))


def load_event_metrics(output: Path) -> dict[str, int]:
    events = [json.loads(line) for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "suppressed_missing_by_fallen": sum(len(event.get("suppressed_missing_by_fallen", [])) for event in events),
        "too_small": sum(int(event.get("small_cone_suppressed", 0)) for event in events),
        "isolated": sum(len(event.get("isolated", [])) for event in events),
        "edge_ignored": sum(len(event.get("edge_ignored", [])) for event in events),
        "missing_edge_ignored": sum(len(event.get("missing_edge_ignored", [])) for event in events),
    }


def row(test: str, scenario: str, group: str, item: dict[str, object], output: Path) -> dict[str, object]:
    event_metrics = load_event_metrics(output)
    alerts = item["activation_alerts"]
    by_type = alerts["by_type"]
    raw = item["raw_and_stable_alert_events"]
    classifier = item.get("classifier") or {}
    classifier_counts = classifier.get("counts", {})
    statuses = item["detections_by_postprocess_status"]
    return {
        "test": test,
        "scenario": scenario,
        "group": group,
        "frames": item["frames"],
        "detections": item["cone"]["detections"],
        "unique_seen": item["cone"]["unique_seen_count"],
        "stable_visible_frames": item["cone"]["stable_visible_frames"],
        "raw_fallen": raw["fallen"],
        "stable_fallen": raw["stable_fallen"],
        "raw_missing": raw["missing"],
        "stable_missing": raw["stable_missing"],
        "raw_displaced": raw["displaced"],
        "stable_displaced": raw["stable_displaced"],
        "fallen_activations": by_type.get("FALLEN", 0),
        "missing_activations": by_type.get("MISSING", 0),
        "shift_activations": by_type.get("SHIFT", 0),
        "total_activations": alerts["count"],
        "not_cone_suppressed": classifier_counts.get("not_cone_suppressed", 0),
        "fallen_votes": classifier_counts.get("fallen_votes", 0),
        "too_small": event_metrics["too_small"],
        "isolated": event_metrics["isolated"],
        "edge_ignored": event_metrics["edge_ignored"],
        "missing_edge_ignored": event_metrics["missing_edge_ignored"],
        "suppressed_missing_by_fallen": event_metrics["suppressed_missing_by_fallen"],
        "postprocess_accepted": statuses.get("accepted", 0),
        "postprocess_classifier_not_cone": statuses.get("classifier_not_cone", 0),
        "postprocess_too_small": statuses.get("too_small", 0),
    }


def save_csv(rows: list[dict[str, object]]) -> None:
    path = OUT / "ablation_metrics.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bar_chart(path: Path, title: str, labels: list[str], series: dict[str, list[float]], ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=180)
    x = np.arange(len(labels))
    width = 0.78 / max(1, len(series))
    palette = ("#2f5597", "#c55a11", "#70ad47", "#7f7f7f", "#a64d79")
    for index, (name, values) in enumerate(series.items()):
        ax.bar(x + (index - (len(series) - 1) / 2) * width, values, width, label=name, color=palette[index % len(palette)])
    ax.set_title(title, fontsize=13)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x, labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(3, len(series)))
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_markdown(rows: list[dict[str, object]], hashes: dict[str, str]) -> None:
    def get(test: str, scenario: str, group: str) -> dict[str, object]:
        return next(r for r in rows if r["test"] == test and r["scenario"] == scenario and r["group"] == group)

    lines = [
        "# 三次视频消融实验汇总（专利证据版）",
        "",
        "本汇总只记录固定视频片段上的回放证据，用于说明时序身份、三分类证据融合、资格过滤和 FALLEN-MISSING 互斥机制在相应测试场景中的技术效果。片段计数不等同于全数据集 precision、recall 或泛化性能。",
        "",
        "## 一、统一条件",
        "",
        "- OBB 权重 SHA-256：`" + hashes["obb"] + "`",
        "- MobileNet 权重 SHA-256：`" + hashes["classifier"] + "`",
        "- 阈值文件 SHA-256：`" + hashes["thresholds"] + "`",
        "- 输入尺寸：1280；YOLO `conf=0.25`、`iou=0.5`；分类器 `full` 模式使用 1.6 倍候选上下文。",
        "- A/B/C/D/E 的完整参数、模型哈希和视频哈希保存在各次运行的 `run_config.json`。",
        "",
        "## 二、第一组：水面光斑和倒伏/正向倒伏片段",
        "",
        "水面片段 269 帧；倒伏片段 75 帧。`raw_fallen` 和 `stable_fallen` 是事件帧统计，`activation` 是业务激活条数。",
        "",
        "| 场景 | 组 | raw/stable FALLEN | raw/stable MISSING | FALLEN 激活 | MISSING 激活 | 总激活 | unique |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in ("water_glare", "fallen_positive"):
        for group in "ABCDE":
            r = get("test1", scenario, group)
            lines.append(f"| {scenario} | {group} | {r['raw_fallen']}/{r['stable_fallen']} | {r['raw_missing']}/{r['stable_missing']} | {r['fallen_activations']} | {r['missing_activations']} | {r['total_activations']} | {r['unique_seen']} |")
    lines += [
        "",
        "水面片段中，A 到 B 的 FALLEN 业务激活由 103 降为 1，说明身份时序明显抑制逐帧光斑告警重复触发；C/D 使用三分类后，累计稳定轨迹由 44 降为 19，水面候选被显著压缩；D 组合配置水面 FALLEN 激活为 0。倒伏片段中，旧 OBB 角度分支本身的候选能力有限，B 的稳定 FALLEN 为 0，因此该结果只能说明当前片段下的时序抑制现象，不能写成倒伏召回提升结论。",
        "",
        "## 三、第二组：过小锥桶和孤立点片段",
        "",
        "输入为 `26-08-03_small_cone.mp4`，3942 帧，约 131.8 秒。",
        "",
        "| 组 | too_small | classifier_not_cone | isolated | missing_edge_ignored | stable FALLEN | stable MISSING | 总激活 | unique |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("D", "E"):
        r = get("test2", "small_cone", group)
        lines.append(f"| {group} | {r['too_small']} | {r['not_cone_suppressed']} | {r['isolated']} | {r['missing_edge_ignored']} | {r['stable_fallen']} | {r['stable_missing']} | {r['total_activations']} | {r['unique_seen']} |")
    lines += [
        "",
        "E 组新增过滤 613 个过小候选，累计稳定轨迹由 105 降为 102，并产生 286 条边缘缺失抑制记录。该片段同时改变过小过滤和孤立端点参数，因而只能证明资格过滤组合的场景效果，不能单独分离孤立端点规则的贡献。",
        "",
        "## 四、第三组：FALLEN-MISSING 互斥片段",
        "",
        "从 `26-08-07-多角度.mp4` 截取原视频 36–50 秒，420 帧，约 14.014 秒。",
        "",
        "| 组 | raw/stable MISSING | raw/stable FALLEN | suppressed_missing_by_fallen | MISSING 激活 | FALLEN 激活 | 总激活 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("D", "E"):
        r = get("test3", "fallen_missing", group)
        lines.append(f"| {group} | {r['raw_missing']}/{r['stable_missing']} | {r['raw_fallen']}/{r['stable_fallen']} | {r['suppressed_missing_by_fallen']} | {r['missing_activations']} | {r['fallen_activations']} | {r['total_activations']} |")
    lines += [
        "",
        "E 组首次互斥记录位于测试片段 1.935 秒，对应原视频约 37.935 秒；最后一处约对应原视频 40.238 秒。E 保留 5 次 FALLEN 激活，同时将 MISSING 激活由 2 次降为 1 次，并记录 57 条倒伏占据缺口预期位置的审计记录。这是本次最直接的 FALLEN-MISSING 互斥证据。",
        "",
        "## 五、专利表述边界",
        "",
        "本次实验支持的技术效果是：对连续视频帧中的候选目标建立稳定身份，以时序投票抑制单帧检测抖动；将 OBB 几何姿态与三分类候选证据融合，降低水面光斑等非锥桶候选进入异常规则的概率；对过小、边缘或孤立候选进行资格约束；当已确认倒伏目标占据缺口预期位置时，保留倒伏异常并抑制同一物理事件的缺失业务告警。",
        "",
        "不应从这些片段直接推出全场景准确率、召回率、误报率或部署性能；不应把 `unique_seen` 当作物理锥桶真值数量；不应把 D/E 在未触发互斥的片段上的差异写成 FALLEN-MISSING 机制证据。",
        "",
        "## 六、图表",
        "",
        "![水面光斑 FALLEN 消融](charts/01_water_fallen.png)",
        "",
        "![倒伏片段 FALLEN 消融](charts/02_fallen_positive.png)",
        "",
        "![过小和边缘资格过滤](charts/03_small_cone_de.png)",
        "",
        "![FALLEN-MISSING 互斥](charts/04_fallen_missing_de.png)",
        "",
        "## 七、证据目录",
        "",
        "- 第一组：`script/runs/ablation_20260824_old_yolo_glare_fallen/`",
        "- 第二组：`script/runs/ablation_20260825_small_cone/`",
        "- 第三组：`script/runs/ablation_20260825_fallen_missing_36_50/`",
        "- 结构化指标：`ablation_metrics.csv`；机器汇总：`source_summaries.json`。",
    ]
    (OUT / "ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "charts").mkdir(exist_ok=True)
    first = load_summary(ROOT / "script/runs/ablation_20260824_old_yolo_glare_fallen/summary.json")
    second = load_summary(ROOT / "script/runs/ablation_20260825_small_cone/summary.json")
    third = load_summary(ROOT / "script/runs/ablation_20260825_fallen_missing_36_50/summary.json")
    summaries = {"test1": first, "test2": second, "test3": third}
    (OUT / "source_summaries.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows: list[dict[str, object]] = []
    for scenario in ("water_glare", "fallen_positive"):
        for group in "ABCDE":
            item = by_group(first, group, scenario)
            rows.append(row("test1", scenario, group, item, ROOT / "script/runs/ablation_20260824_old_yolo_glare_fallen" / group / scenario))
    for group in ("D", "E"):
        item = by_group(second, group)
        rows.append(row("test2", "small_cone", group, item, ROOT / "script/runs/ablation_20260825_small_cone" / group))
        item = by_group(third, group)
        rows.append(row("test3", "fallen_missing", group, item, ROOT / "script/runs/ablation_20260825_fallen_missing_36_50" / group))
    save_csv(rows)
    hashes = {
        "obb": "974d056f6cdaca63be3885b075425b7e2b9da0e54bbe79526e811e51c7bef76e",
        "classifier": "a4e945b4ab7b55c3027ddcaaa3036d6f58c5e4b306694b454840b224e88edbcb",
        "thresholds": "b139728d33ba2179202d7d45df8253dc11bf3c001d51a17aeee36d4d494ead62",
    }
    build_markdown(rows, hashes)
    get = lambda test, scenario, group: next(r for r in rows if r["test"] == test and r["scenario"] == scenario and r["group"] == group)
    labels = list("ABCDE")
    water = [get("test1", "water_glare", g) for g in labels]
    fallen = [get("test1", "fallen_positive", g) for g in labels]
    bar_chart(OUT / "charts/01_water_fallen.png", "Water-glare clip: FALLEN ablation", labels, {"raw FALLEN": [r["raw_fallen"] for r in water], "stable FALLEN": [r["stable_fallen"] for r in water], "FALLEN activations": [r["fallen_activations"] for r in water]}, "count")
    bar_chart(OUT / "charts/02_fallen_positive.png", "Fallen-positive clip: temporal and classifier ablation", labels, {"raw FALLEN": [r["raw_fallen"] for r in fallen], "stable FALLEN": [r["stable_fallen"] for r in fallen], "FALLEN activations": [r["fallen_activations"] for r in fallen]}, "count")
    small = [get("test2", "small_cone", g) for g in ("D", "E")]
    bar_chart(OUT / "charts/03_small_cone_de.png", "Small-cone clip: D versus E", ["D", "E"], {"too small": [r["too_small"] for r in small], "not-cone suppressed": [r["not_cone_suppressed"] for r in small], "edge missing ignored": [r["missing_edge_ignored"] for r in small], "total activations": [r["total_activations"] for r in small]}, "count")
    fm = [get("test3", "fallen_missing", g) for g in ("D", "E")]
    bar_chart(OUT / "charts/04_fallen_missing_de.png", "FALLEN-MISSING mutual exclusion: D versus E", ["D", "E"], {"stable MISSING": [r["stable_missing"] for r in fm], "stable FALLEN": [r["stable_fallen"] for r in fm], "suppressed by FALLEN": [r["suppressed_missing_by_fallen"] for r in fm], "MISSING activations": [r["missing_activations"] for r in fm]}, "count")
    print(OUT)


if __name__ == "__main__":
    main()
