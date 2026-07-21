# Change LeWM's h5 file to lerobot

python scripts/h5_to_lerobot.py \
  --input-h5 /path/to/data.h5 \
  --output-dir /path/to/lerobot_with_return \
  --repo-id lerobot_with_return \
  --task "Pick up the shrimps and squids into the sink, and then press the red button." \
  --fps 10 \
  --failure-penalty 300 \
  --normalization global_minmax

#   --video-codec h264_nvenc \
#   --video-workers 2 \