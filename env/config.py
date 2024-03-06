ckpt = 'weights/real-world_ccsr.ckpt'
config = 'configs/model/ccsr_stage2.yaml'
steps = 45
sr_scale = 4
repeat_times = 1
tiled = False
tile_size = 512
tile_stride = 256
color_fix_type = "adain"
t_max = 0.6667
t_min = 0.3333
show_lq = False
skip_if_exist = False
seed = 233
device = "cuda"

# minio 配置
sr_bucket = "test"
# input = 'preprocess_img/input_x2'  # 这个值需要被函数参数覆盖
# output = '/path/to/output'  # 这个值将被函数参数覆盖
