# region package
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pytz
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from timm.scheduler import create_scheduler
from torch.amp import GradScaler

from params import get_pangu_model_args, get_pangu_data_args
from weather_dataset import WeatherPanguData
from fourcastnet3 import AtmoSphericNeuralOperatorNet
from loss_crps import FourCastNet3Loss

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# endregion

# region

def load_model(model, optimizer=None, lr_scheduler=None, loss_scaler=None, path=None, only_model=False):
    """
    加载模型
    """
    print("=> loading checkpoint '{}'".format(path))
    start_epoch = 0
    start_step = 0
    min_loss = np.inf
    if path.exists():
        ckpt = torch.load(path, map_location="cpu")
        if only_model:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            if lr_scheduler is not None:
                lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            if ckpt['loss_scaler'] is not None:
                loss_scaler.load_state_dict(ckpt['loss_scaler'])
            start_epoch = ckpt["epoch"]
            start_step = ckpt["step"]
            min_loss = ckpt["min_loss"]

    return start_epoch, start_step, min_loss

def save_model(model, epoch=0, step=0, min_loss=0, optimizer=None, lr_scheduler=None, loss_scaler=None, path=None, only_model=False):
    """
    保存模型
    """
    print("=> saving checkpoint '{}'".format(path))
    if only_model:
        states = {
            'model': model.state_dict(),
        }
    else:
        states = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'loss_scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
            'epoch': epoch,
            'step': step,
            'min_loss': min_loss
        }

    torch.save(states, path)

def generate_times(start_date, end_date, freq='6h'):
    start_date = start_date.replace(tzinfo=pytz.UTC)
    end_date = end_date.replace(tzinfo=pytz.UTC)
    return pd.date_range(start=start_date, end=end_date, freq=freq, tz='UTC').to_pydatetime().tolist()

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()  # 总进程数
    return rt

def compute_rmse(out, tgt, quad_weight):
    quad = torch.sum(((out - tgt) ** 2) * quad_weight, dim=(-2, -1))
    return torch.sqrt(quad.mean())

def autoregressive_rollout(model, x, xzen, surface_mask, steps, residual=False):
    """
    使用模型进行自回归预测。
    
    参数:
        model: 预测模型
        input_x (List): 包含多网格图和输入张量
        steps (int): 预测步数

    返回:
        pred_norm_list (List[Tensor]): 每一步预测结果组成的列表，每个元素形状为 [bs, 1, c, h, w]
    """
    input_seq = x.clone()
    pred_norm_list = []

    for k in range(steps):
        input_seq_all = torch.concat([input_seq, xzen, surface_mask.unsqueeze(0).repeat(input_seq.shape[0],1,1,1)], dim=1)   
        with torch.amp.autocast('cuda', enabled=False):
            out = model(input_seq_all)                     # [bs, c, h, w]
        pred_norm_list.append(out)
        input_seq = out
    return torch.stack(pred_norm_list, dim=1)

# endregion

# region main
@torch.no_grad()
def valid_one_epoch(model, data_args, dataloaders, surface_mask, criterion, device="cuda:0"):
    loss_all = torch.tensor(0.).to(device)
    count = torch.tensor(0.).to(device)
    score_all = torch.tensor(0.).to(device)

    # selected_channels = list(np.arange(0,66))+[67]+[68]+[70]
    norm_dict = torch.load(data_args['root_path'] / data_args['norm_path'])
    var_mean = norm_dict['var_mean'][:70].cuda(non_blocking=True).view(1, 1, 70, 1, 1)
    var_std = norm_dict['var_std'][:70].cuda(non_blocking=True).view(1, 1, 70, 1, 1)
    surface_mask = surface_mask.cuda() # 60-0 80-140
    
    pbar = tqdm(dataloaders, total=len(dataloaders))

    model.eval()
    for step, batch in enumerate(pbar):
        x_phys = batch['input'].cuda(non_blocking=True)             # [bs, t_in, 70, h, w]
        y_phys = batch['target'].cuda(non_blocking=True)  # [bs,t_pretrain_out, 70, h, w]
        xzen = batch['input_zenith'].cuda(non_blocking=True) # [bs, t_in, 1, h, w]

        x_norm = (x_phys - var_mean) / var_std
        y_norm = (y_phys - var_mean) / var_std
        
        #构建surface和air输入
        x_norm = x_norm.squeeze(1)
        xzen = xzen.squeeze(1)
        
        pred = autoregressive_rollout(model, x_norm, xzen, surface_mask, data_args['t_pretrain_out'])
        loss = criterion(pred.unsqueeze(2), y_norm)

        loss_all += loss.item()
        count += 1

        pred_phy_surface = pred[:,:,-5:,20:-20,20:-20] * var_std[:,:,-5:,:,:] + var_mean[:,:,-5:,:,:]
        
        B,T,C,H,W = pred_phy_surface.shape
        jacobian = torch.clamp(torch.sin(torch.linspace(0, torch.pi, 721)), min=0.0)
        dtheta = torch.pi / 721#img_shape[0]
        dlambda = 2 * torch.pi / 1440# img_shape[1]
        dA = dlambda * dtheta
        quad_weight = dA * jacobian.unsqueeze(1)
        quad_weight = quad_weight.tile(1, 1440)
        # numerical precision can be an issue here, make sure it sums to 4pi:
        quad_weight = quad_weight * (4.0 * torch.pi) / torch.sum(quad_weight)
        quad_weight = torch.flip(quad_weight[361+20:361+H+20,20:W+20],dims=[0])
        
        score = compute_rmse(pred_phy_surface, y_phys[:,:,-5:,20:-20,20:-20], quad_weight.cuda())
        score_all += score
        
        pbar.set_postfix({
            'Valid Aver Loss': f"{(loss_all / count).item():.4f}",
            'Valid Eval Score': f"{(score_all / count).item():.4f}",
        })

        # if step % 50 == 0 and device == 0:
        #     print("Step: ", step, " | Valid Aver Loss:", (loss_all / count).item(),
        #           " | Valid Eval Score: ", (score_all / count).item(), flush=True)

    return loss_all / count, score_all / count

def train_one_epoch(epoch, model, data_args, dataloaders, surface_mask, criterion, optimizer, device="cuda:0"):
    loss_all = torch.tensor(0.).to(device)
    count = torch.tensor(0.).to(device)
    model.train()

    # selected_channels = list(np.arange(0,66))+[67]+[68]+[70]
    norm_dict = torch.load(data_args['root_path'] / data_args['norm_path'])
    var_mean = norm_dict['var_mean'][:70].cuda(non_blocking=True).view(1, 1, 70, 1, 1)
    var_std = norm_dict['var_std'][:70].cuda(non_blocking=True).view(1, 1, 70, 1, 1)
    
    surface_mask = surface_mask.cuda() # 60-0 80-140

    scaler = GradScaler("cuda", enabled=False)
    pbar = tqdm(dataloaders, total=len(dataloaders), desc=f"Epoch [{epoch}/{70}]")
    
    for step, batch in enumerate(pbar):
        # x, y = [x.cuda(non_blocking=True) for x in batch]                                          # [bs, t, c, h, w]
        x_phys = batch['input'].cuda(non_blocking=True)             # [bs, t_in, 70, h, w]
        y_phys = batch['target'].cuda(non_blocking=True)  # [bs, t_pretrain_out, 70, h, w]
        xzen = batch['input_zenith'].cuda(non_blocking=True) # [bs, t_in, 1, h, w]

        x_norm = (x_phys - var_mean) / var_std
        y_norm = (y_phys - var_mean) / var_std
        
        #构建surface和air输入
        x_norm = x_norm.squeeze(1)
        xzen = xzen.squeeze(1)

        optimizer.zero_grad()
        
        pred = autoregressive_rollout(model, x_norm, xzen, surface_mask, data_args['t_pretrain_out'])
        loss = criterion(pred.unsqueeze(2), y_norm)

        loss_all += reduce_tensor(loss).item()  # 有多个进程，把进程0和1的loss加起来平均
        count += 1

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        pbar.set_postfix({
            'Train Aver Loss': f"{(loss_all / count).item():.4f}",
        })

        # if step % 50 == 0 and device == 0:
        #     print("Step: ", step, " | Train Aver Loss:", (loss_all / count).item(), flush=True)

    return loss_all / count

def train(local_rank, model_args, data_args, proj):
    '''
    Args:
        local_rank: 本地进程编号
        rank: 进程的global编号
        local_size: 每个节点几个进程
        model_args, data_args: 配置参数
        word_size: 进程总数
    '''
    rank = local_rank
    gpu = local_rank
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:60066',
                            rank=rank, world_size=model_args.world_size)

    # Path
    log_path = data_args['root_path'] / data_args['train_log_path']
    latest_model_path = data_args['root_path'] / data_args['latest_model_path']
    train_times = generate_times(data_args['train_start_datetime'], data_args['train_end_datetime'])
    valid_times = generate_times(data_args['valid_start_datetime'], data_args['valid_end_datetime'])

    # Model
    model = AtmoSphericNeuralOperatorNet(channel_names=model_args.channel_names, aux_channel_names=model_args.aux_channels).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=model_args.lr, betas=(0.9, 0.95),
                                  weight_decay=model_args.weight_decay)
    lr_scheduler, _ = create_scheduler(model_args, optimizer)

    criterion = FourCastNet3Loss(channel_names=model_args.channel_names)

    start_epoch, start_step, min_loss = load_model(model, optimizer=optimizer, lr_scheduler=lr_scheduler, path=latest_model_path)
    if torch.cuda.device_count() > 1:
        model = DDP(model, device_ids=[gpu])

    # Dataloader
    train_dataset = WeatherPanguData(train_times, data_args['npy_path'], data_args['tp6hr_path'], input_window_size=data_args['t_in'], output_window_size=data_args['t_pretrain_out'])
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(dataset=train_dataset, batch_size=model_args.batch_size,
                              sampler=train_sampler,
                              drop_last=True, shuffle=False, num_workers=4, pin_memory=True)
    valid_dataset = WeatherPanguData(valid_times, data_args['npy_path'], data_args['tp6hr_path'], input_window_size=data_args['t_in'], output_window_size=data_args['t_pretrain_out'])
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=model_args.batch_size,
                              drop_last=True, num_workers=4, pin_memory=True)
    
    land_mask_all = np.load("./constant_mask/land_mask.npy")
    land_mask = land_mask_all[360+240:360:-1,320+720:560+720].copy()
    land_mask = torch.FloatTensor(land_mask)
    soil_type_all = np.load("./constant_mask/soil_type.npy")
    soil_type = soil_type_all[360+240:360:-1,320+720:560+720].copy()
    soil_type = torch.FloatTensor(soil_type)
    topography_all = np.load("./constant_mask/topography.npy")
    topography = topography_all[360+240:360:-1,320+720:560+720].copy()
    topography = torch.FloatTensor(topography)
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0)

    best_score = 1e3
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(start_epoch, 71):
            train_sampler.set_epoch(epoch)
            train_loss = train_one_epoch(epoch, model, data_args, train_loader, surface_mask, criterion, optimizer, device=gpu)

            lr_scheduler.step(epoch)
            if gpu == 0:
                val_loss, val_score = valid_one_epoch(model, data_args, valid_loader, surface_mask, criterion, device=gpu)

                print(f"Epoch {epoch} | LR: {optimizer.param_groups[0]['lr']:.6f} | Train loss: {train_loss.item():.6f} | Val loss: {val_loss.item():.6f}, Val score: {val_score.item():.6f}", flush=True)

                save_model(model, epoch=epoch + 1, min_loss=min_loss, optimizer=optimizer, lr_scheduler=lr_scheduler, path=latest_model_path)
                with open(log_path, 'a') as f:
                    f.write(f"Epoch {epoch} | LR: {optimizer.param_groups[0]['lr']:.6f} | Train loss: {train_loss.item():.6f} | Val loss: {val_loss.item():.6f}, Val score: {val_score.item():.6f}\n")

                if val_score < best_score:
                    best_score = val_score
                    min_loss = val_loss
                    save_model(model, epoch=epoch, min_loss=min_loss, path=data_args['root_path'] / f"output/{proj}/ckpts/epoch{epoch + 1}_{val_score:.6f}_best.pt", only_model=True)
            dist.barrier()



if __name__ == '__main__':

    torch.manual_seed(2023)
    np.random.seed(2023)
    cudnn.benchmark = True
    
    model_args = get_pangu_model_args()
    proj='fcn3_new'
    data_args = get_pangu_data_args(proj)
    index = 1
    proj_base = 'fcn3_new'
    while os.path.exists(str(data_args['root_path']) + "/output/" + proj):
        proj = proj_base +"_" +str(index)
        index += 1
    ckpt_path = str(data_args['root_path']) + "/output/" + proj + "/ckpts"
    log_path = str(data_args['root_path']) + "/output/" + proj + "/logs"
    os.makedirs(ckpt_path)
    os.makedirs(log_path)
    
    mp.spawn(train, args=(model_args, data_args, proj), nprocs=1, join=True)

# endregion 