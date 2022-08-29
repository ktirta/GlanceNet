import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils
from utils.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num) 
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return sub_pc

class SelfAtt(nn.Module):
    def __init__(self, num_channel, num_head):
        super(SelfAtt, self).__init__()
        self.mha = nn.MultiheadAttention(num_channel, num_head, batch_first=True)

    def forward(self, x):
        x, _ = self.mha(x, x, x)
        return x

class LQBlock(nn.Module):
    def __init__(self, num_lq, num_channel, num_head):
        super(LQBlock, self).__init__()
        self.lq = nn.Parameter(torch.rand(1, num_lq, num_channel))
        self.mha = nn.MultiheadAttention(num_channel, num_head, batch_first=True)

    def forward(self, x):
        batch_size = x.shape[0]
        x, _ = self.mha(self.lq.repeat(batch_size, 1, 1), x, x)
        return x

class MLPBlock(nn.Module):
    def __init__(self, c_in, c_out, bias = False, use_dropout = False, last_affine = False):
        super(MLPBlock, self).__init__()
        if use_dropout:
            self.block = nn.Sequential(
                nn.Linear(c_in, c_out, bias = bias),
                nn.LayerNorm(c_out),
                nn.GELU(),
                nn.Dropout(0.5)
            )
        else:
            self.block = nn.Sequential(
                nn.Linear(c_in, c_out, bias = bias),
                nn.LayerNorm(c_out),
                nn.GELU(),
            )
        
        if last_affine:
            self.last_block = nn.Linear(c_out, c_out, bias = bias)
        self.last_affine = last_affine
    
    def forward(self, x):
        x = self.block(x)
        if self.last_affine:
            x = self.last_block(x)
        return x

class LQ_Encoder(nn.Module):
    def __init__(self, c_in, c_out, num_lq, num_head):
        super(LQ_Encoder, self).__init__()
        self.mlp1 = MLPBlock(c_in, c_out, bias=True)
        self.mlp2 = MLPBlock(c_out, c_out, bias=True)

        self.mha1 = LQBlock(num_lq, c_out, num_head)

    def forward(self, x):
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mha1(x)

        return x

class Fold(nn.Module):
    def __init__(self, in_channel , step , hidden_dim = 512):
        super().__init__()

        self.in_channel = in_channel
        self.step = step

        a = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(1, step).expand(step, step).reshape(1, -1)
        b = torch.linspace(-1., 1., steps=step, dtype=torch.float).view(step, 1).expand(step, step).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).cuda()

        self.folding1 = nn.Sequential(
            nn.Conv1d(in_channel + 2, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

        self.folding2 = nn.Sequential(
            nn.Conv1d(in_channel + 3, hidden_dim, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim//2, 1),
            nn.BatchNorm1d(hidden_dim//2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim//2, 3, 1),
        )

    def forward(self, x):
        num_sample = self.step * self.step
        bs = x.size(0)
        features = x.view(bs, self.in_channel, 1).expand(bs, self.in_channel, num_sample)
        seed = self.folding_seed.view(1, 2, num_sample).expand(bs, 2, num_sample).to(x.device)

        x = torch.cat([seed, features], dim=1)
        fd1 = self.folding1(x)
        x = torch.cat([fd1, features], dim=1)
        fd2 = self.folding2(x)

        return fd2

class GlanceNet(nn.Module):
    def __init__(self, cfg):
        super(GlanceNet, self).__init__()
        self.cfg = cfg
        

        self.coarse_encoders = nn.ModuleList()
        self.coarse_decoders = nn.ModuleList()

        for idx in range(len(cfg.NETWORK.COARSE_ENC_NUM_QUERY)):
            if idx == 0:
                self.coarse_encoders.append(LQ_Encoder(c_in=3, 
                                                        c_out=cfg.NETWORK.COARSE_ENC_DIM[idx], 
                                                        num_lq=cfg.NETWORK.COARSE_ENC_NUM_QUERY[idx],
                                                        num_head=4))
            else:
                self.coarse_encoders.append(LQ_Encoder(c_in=cfg.NETWORK.COARSE_ENC_DIM[idx-1], 
                                                        c_out=cfg.NETWORK.COARSE_ENC_DIM[idx], 
                                                        num_lq=cfg.NETWORK.COARSE_ENC_NUM_QUERY[idx],
                                                        num_head=4))
        self.coarse_predictor = LQ_Encoder(c_in = cfg.NETWORK.COARSE_ENC_DIM[-1], 
                                            c_out = cfg.NETWORK.NUM_COARSE_POINTS, 
                                            num_lq = 3, 
                                            num_head=4)

        self.global_pool = LQ_Encoder(c_in=cfg.NETWORK.COARSE_ENC_DIM[-1], 
                                        c_out=cfg.NETWORK.GLOBAL_FEATS_DIM, 
                                        num_lq=1, 
                                        num_head=4)

        for idx in range(len(cfg.NETWORK.COARSE_DEC_NUM_QUERY)):
            if idx == 0:
                self.coarse_decoders.append(LQ_Encoder(c_in=cfg.NETWORK.GLOBAL_FEATS_DIM + 3, 
                                                        c_out=cfg.NETWORK.COARSE_DEC_DIM[idx], 
                                                        num_lq=cfg.NETWORK.COARSE_DEC_NUM_QUERY[idx],
                                                        num_head=1))
            else:
                self.coarse_decoders.append(LQ_Encoder(c_in=cfg.NETWORK.COARSE_DEC_DIM[idx-1], 
                                                        c_out=cfg.NETWORK.COARSE_DEC_DIM[idx], 
                                                        num_lq=cfg.NETWORK.COARSE_DEC_NUM_QUERY[idx],
                                                        num_head=1))

        fold_step = cfg.NETWORK.FOLD_STEP
        self.fold = Fold(in_channel=cfg.NETWORK.GLOBAL_FEATS_DIM + 3, 
                            step=fold_step, 
                            hidden_dim=512)

        self.localSelfAtt = nn.ModuleList()
        for idx in range(cfg.NETWORK.LOCAL_SELF_ATT):
            self.localSelfAtt.append(SelfAtt(num_channel=3, num_head=1))

        if cfg.NETWORK.LOSS == 'CDL1':
            self.criterion = ChamferDistanceL1()
            print('[LOSS] CDL1')
        else:
            self.criterion = ChamferDistanceL2()
            print('[LOSS] CDL2')

    def forward(self, data_dic):
        xyz_feats = data_dic['incomplete_pc']
        batch_size = xyz_feats.shape[0]


        for coarse_encoder in self.coarse_encoders:
            xyz_feats = coarse_encoder(xyz_feats)
        # xyz_feats = self.LQ_Encoder1(xyz)
        coarse_points = self.coarse_predictor(xyz_feats).reshape(batch_size, -1, 3) # B M 3
        global_feats = self.global_pool(xyz_feats)

        coarse_feats = torch.cat((coarse_points, global_feats.repeat(1, coarse_points.shape[1], 1)), dim = -1) # B M C+3
        
        for coarse_decoder in self.coarse_decoders:
            coarse_feats = coarse_decoder(coarse_feats)
        
        coarse_feats = coarse_feats.reshape(batch_size*coarse_feats.shape[1], -1) # BM C+3

        xyz_est = self.fold(coarse_feats).reshape(batch_size, coarse_points.shape[1], 3, -1) # B M 3 N

        xyz_est = xyz_est + coarse_points.unsqueeze(-1) # B M 3 N
        # data_dic['mid_pc_pred'] = xyz_est.reshape(batch_size, -1, 3)

        xyz_est = xyz_est.permute(0, 1, 3, 2).reshape(xyz_est.shape[0]*xyz_est.shape[1], -1, 3) # (B M 3 N) --> (B M N 3) --> (BM N 3)
        for local_SA in self.localSelfAtt:
            xyz_est = local_SA(xyz_est)

        xyz_est = xyz_est.reshape(batch_size, -1, 3)
        data_dic['coarse_pc_pred'] = coarse_points
        data_dic['fine_pc_pred'] = xyz_est

        return data_dic

    def get_loss(self, data_dic):
        missing_pc_gt = data_dic['missing_pc']
        fine_pc_pred = data_dic['fine_pc_pred']
        # mid_pc_pred = data_dic['mid_pc_pred']
        coarse_pc_pred = data_dic['coarse_pc_pred']

        coarse_cd = self.criterion(coarse_pc_pred, missing_pc_gt)
        fine_cd = self.criterion(fine_pc_pred, missing_pc_gt)
        # mid_cd = self.criterion(mid_pc_pred, missing_pc_gt)

        loss = 0.5*coarse_cd  + fine_cd

        loss_dict = {
            'coarse_loss': coarse_cd.item(),
            'fine_loss': fine_cd.item(),
            'mid_loss': 0,
            'loss': loss.item()
        }
        return loss, loss_dict