python scripts/add_returns_to_lerobot.py \
  --repo-id ori_lerobot \
  --root /path/to/ori_lerobot \
  --output-dir /path/to/lerobot_with_return \
  --failure-penalty 300 \
  --normalization global_minmax \
  --success-labels /path/to/success_labels.json