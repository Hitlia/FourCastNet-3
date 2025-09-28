import argparse
from pathlib import Path
from datetime import datetime
CHANNELS = ["t2m", "d2m", "u10m", "v10m", "tp", "t50", "t100", "t150", "t200", "t250", "t300", "t400", "t500", "t600", "t700", "t850", "t925", "t1000", "u50", "u100", "u150", "u200", "u250", "u300", "u400", "u500", "u600", "u700", "u850", "u925", "u1000", "v50", "v100", "v150", "v200", "v250", "v300", "v400", "v500", "v600", "v700", "v850", "v925", "v1000", "z50", "z100", "z150", "z200", "z250", "z300", "z400", "z500", "z600", "z700", "z850", "z925", "z1000", "q50", "q100", "q150", "q200", "q250", "q300", "q400", "q500", "q600", "q700", "q850", "q925", "q1000"]
AUX_CHANNELS = ["zxen", "landmask", "soil", "topo"]


def get_pangu_model_args():
    parser = argparse.ArgumentParser('FourcastNet3', add_help=False)
    parser.add_argument('--channel_names', default=CHANNELS, type=list)
    parser.add_argument('--aux_channels', default=AUX_CHANNELS, type=list)
     
    # Training parameters
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=70, type=int)
    parser.add_argument('--device', default="cuda", type=str)

    parser.add_argument('-g', '--gpuid', default=0, type=int, help="which gpu to use")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--master_addr", default="127.0.0.1", type=str)
    parser.add_argument("--master_port", default="12356", type=str)
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
 
    # Model parameters
    parser.add_argument('--predict-steps', default=20, type=int, help='predict steps')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
    # parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON', help='Optimizer Epsilon (default: 1e-8)')
    # parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA', help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=32, metavar='NORM', help='Clip gradient norm (default: None, no clipping)')
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=3e-6, help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER', help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', help='learning rate (default: 5e-4)')
    # parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct', help='learning rate noise on/off epoch percentages')
    # parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT', help='learning rate noise limit percent (default: 0.67)')
    # parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV', help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR', help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=0, metavar='LR', help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N', help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N', help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N', help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N', help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE', help='LR decay rate (default: 0.1)')

    # Pipline training parameters
    parser.add_argument('--pp_size', type=int, default=8, help='pipeline parallel size')
    parser.add_argument('--chunks', type=int, default=1, help='chunk size')

    return parser.parse_args()

def get_pangu_data_args(cur_proj): 
    data_args = {
        # 数据路径
        'root_path':            Path('/fcn3/'),
        'npy_path':             Path('/npy-data/'),
        'tp6hr_path':           Path('/tp6hr-data/'),

        # norm
        'norm_path':            f"/dataset/norms/norm.npy",
        'diff_path':            f"/dataset/norms/diff.npy",

        # log
        'train_log_path':       f"output/{cur_proj}/logs/fcn3_train.log",
        'test_log_path':        f"output/{cur_proj}/logs/fcn3_test.json",

        # pt
        'latest_model_path':    f"output/{cur_proj}/ckpts/fcn3_latest.pt",
        'best_model_path':      f"output/{cur_proj}/ckpts/fcn3_best.pt",

        # 配置参数
        'train_start_datetime': datetime(2018, 1, 1, 1),  # 起始时间
        'train_end_datetime':   datetime(2023, 12, 31, 19),  # 结束时间
        'valid_start_datetime': datetime(2024, 1, 1, 1),
        'valid_end_datetime':   datetime(2024, 12, 31, 19),
        'test_start_datetime':  datetime(2025, 1, 1, 1),
        'test_end_datetime':    datetime(2025, 3, 31, 19),
        't_in':                 1,
        't_pretrain_out':       1,
        't_finetune_out':       8,
        't_final_out':          32,
    }
    return data_args
