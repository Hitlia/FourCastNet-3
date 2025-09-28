import torch

def compute_rmse(out, tgt, quad_weight):
    quad = torch.sum(((out - tgt) ** 2) * quad_weight, dim=(-2, -1))
    return torch.sqrt(quad.mean())


pred_phy_surface = torch.zeros([8,4,5,200,200]).cuda()
y_phys = torch.zeros([8,4,70,240,240]).cuda()

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
print(score)