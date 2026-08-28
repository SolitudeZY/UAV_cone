# 2026-08-27 script cleanup archive

This directory preserves scripts moved out of the active `script/` root before
the dual-confidence physical-ID work. No source file was deleted.

## Archived scripts

| Original path | Reason archived |
| --- | --- |
| `script/analyze_cone_count_stability.py` | One-off physical-ID fragmentation diagnostic used on 2026-08-27 |
| `script/audit_20260813_supplement.py` | Date-specific completed dataset audit |
| `script/build_20260813_supplement_dataset.py` | Date-specific completed dataset build |
| `script/build_oversampled_train_config.py` | Historical oversampling experiment helper |
| `script/evaluate_obb_predictions.py` | Older event-log evaluator superseded by `evaluate_obb_checkpoints.py` for current checkpoint evaluation |
| `script/exclude_obb_images.py` | Completed historical dataset exclusion helper |
| `script/generate_ablation_docx.py` | One-off 2026-08-25 patent ablation document generator |
| `script/generate_ablation_report.py` | One-off 2026-08-25 patent ablation report generator |
| `script/rebalance_historical_additions.py` | Completed historical dataset rebalancing helper |
| `script/split_multiangle_dataset.py` | Completed multi-angle dataset split helper |
| `script/summarize_inference.py` | Historical inference-summary helper used by the archived ablation workflow |

The active inference, training, dataset validation, current checkpoint
evaluation, MobileNet, RTMP, physical tracking, and regression-test entrypoints
remain in `script/`.

## Restore

Move a needed file back to `script/` without changing its basename. Scripts in
this archive retain their original code and may still assume repository-root
execution and imports from the active `script/` directory.

Verify the archived files from this directory with:

```bash
sha256sum -c FILES.sha256
```
