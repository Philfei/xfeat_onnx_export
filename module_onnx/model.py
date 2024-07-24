import torch
import torch.nn as nn
import torch.nn.functional as F
from module_onnx.interpolator import InterpolateSparse2d


class BasicLayer(nn.Module):
    """
      Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super().__init__()
        self.layer = nn.Sequential(
                                      nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
                                      nn.BatchNorm2d(out_channels, affine=False),
                                      nn.ReLU(inplace = True),
                                    )

    def forward(self, x):
        return self.layer(x)


class XFeatModel(nn.Module):
    """
       Implementation of architecture described in
       "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    """

    def __init__(self):
        super().__init__()
        # self.scales = nn.Parameter(torch.tensor([1.0, 1.0], dtype=torch.float32))
        self.norm = nn.InstanceNorm2d(1)

        ########### ⬇️ CNN Backbone & Heads ⬇️ ###########

        self.skip1 = nn.Sequential(	nn.AvgPool2d(4, stride = 4),
                                     nn.Conv2d (1, 24, 1, stride = 1, padding=0) )

        self.block1 = nn.Sequential(
                                        BasicLayer( 1,  4, stride=1),
                                        BasicLayer( 4,  8, stride=2),
                                        BasicLayer( 8,  8, stride=1),
                                        BasicLayer( 8, 24, stride=2),
                                    )

        self.block2 = nn.Sequential(
                                        BasicLayer(24, 24, stride=1),
                                        BasicLayer(24, 24, stride=1),
                                     )

        self.block3 = nn.Sequential(
                                        BasicLayer(24, 64, stride=2),
                                        BasicLayer(64, 64, stride=1),
                                        BasicLayer(64, 64, 1, padding=0),
                                     )
        self.block4 = nn.Sequential(
                                        BasicLayer(64, 64, stride=2),
                                        BasicLayer(64, 64, stride=1),
                                        BasicLayer(64, 64, stride=1),
                                     )

        self.block5 = nn.Sequential(
                                        BasicLayer( 64, 128, stride=2),
                                        BasicLayer(128, 128, stride=1),
                                        BasicLayer(128, 128, stride=1),
                                        BasicLayer(128,  64, 1, padding=0),
                                     )

        self.block_fusion =  nn.Sequential(
                                        BasicLayer(64, 64, stride=1),
                                        BasicLayer(64, 64, stride=1),
                                        nn.Conv2d (64, 64, 1, padding=0)
                                     )

        self.heatmap_head = nn.Sequential(
                                        BasicLayer(64, 64, 1, padding=0),
                                        BasicLayer(64, 64, 1, padding=0),
                                        nn.Conv2d (64, 1, 1),
                                        nn.Sigmoid()
                                    )


        self.keypoint_head = nn.Sequential(
                                        BasicLayer(64, 64, 1, padding=0),
                                        BasicLayer(64, 64, 1, padding=0),
                                        BasicLayer(64, 64, 1, padding=0),
                                        nn.Conv2d (64, 65, 1),
                                    )


        ########### ⬇️ Fine Matcher MLP ⬇️ ###########

        self.fine_matcher =  nn.Sequential(
                                            nn.Linear(128, 512),
                                            nn.BatchNorm1d(512, affine=False),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(512, 512),
                                            nn.BatchNorm1d(512, affine=False),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(512, 512),
                                            nn.BatchNorm1d(512, affine=False),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(512, 512),
                                            nn.BatchNorm1d(512, affine=False),
                                            nn.ReLU(inplace = True),
                                            nn.Linear(512, 64),
                                        )
        
        # self.dev = torch.device('cuda')
        self.dev = torch.device('cpu')
        self.interpolator = InterpolateSparse2d('bicubic')
        self.top_k = 1000

    def _unfold2d(self, x, ws = 2):
        """
            Unfolds tensor in 2D with desired ws (window size) and concat the channels
        """
        B, C, H, W = x.shape 
        # The current ONNX export does not support dynamic shape unfold
        if torch.onnx.is_in_onnx_export():
            x = x[..., :x.shape[2] // ws * ws, :x.shape[3] // ws * ws]
            B, C, H, W = x.shape
            return torch.reshape(x, (B, C, H // ws, ws, W // ws, ws)).permute(0, 1, 3, 5, 2, 4).flatten(1, 3)
        else:
            x = x.unfold(2,  ws , ws).unfold(3, ws,ws).reshape(B, C, H//ws, W//ws, ws**2)
            return x.permute(0, 1, 4, 2, 3).reshape(B, -1, H//ws, W//ws)
        # B, C, H, W = x.shape
        # unfolded = torch.zeros(B, C, H // ws, W // ws, ws ** 2)
        # for i in range(H // ws):
        #     for j in range(W // ws):
        #         patch = x[:, :, i * ws:(i + 1) * ws, j * ws:(j + 1) * ws]
        #         unfolded[:, :, i, j] = patch.reshape(B, C, ws ** 2)
        # return unfolded.permute(0, 1, 4, 2, 3).reshape(B, -1, H // ws, W // ws)



    def detectAndCompute(self, x):
        """
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:
                feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
                keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
                heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map

        """
        x, rh1, rw1 = self.preprocess_tensor(x)
        _, _, _H1, _W1 = x.shape
        #dont backprop through normalization
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=True)
            x = self.norm(x)

        # main backbone
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        #pyramid fusion
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
       
       
        feats = self.block_fusion(x3 + x4 + x5)
		#heads
        heatmap = self.heatmap_head(feats) # Reliability map
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8)) #Keypoint map logits
        # keypoints = self.keypoint_head(self._unfold2d2(x)) #Keypoint map logits

        # return feats, keypoints, heatmap

        _, _, _H1, _W1 = x.shape
        M1, K1, H1 = feats, keypoints, heatmap
        M1 = F.normalize(M1, dim=1)

        # Convert logits to heatmap and extract kpts
        K1h = self.get_kpts_heatmap(K1)
        mkpts = self.NMS(K1h, threshold=0.05, kernel_size=5)

        # Compute reliability scores
        _nearest = InterpolateSparse2d('nearest')
        _bilinear = InterpolateSparse2d('bilinear')
        scores = (_nearest(K1h, mkpts, _H1, _W1) * _bilinear(H1, mkpts, _H1, _W1)).squeeze(-1)
        scores[torch.all(mkpts == 0, dim=-1)] = -1

        # Select top-k features
        idxs = torch.argsort(-scores)
        mkpts_x = torch.gather(mkpts[..., 0], -1, idxs)[:, :self.top_k]
        mkpts_y = torch.gather(mkpts[..., 1], -1, idxs)[:, :self.top_k]
        mkpts = torch.cat([mkpts_x[..., None], mkpts_y[..., None]], dim=-1)
        scores = torch.gather(scores, -1, idxs)[:, :self.top_k]

        # Interpolate descriptors at kpts positions
        feats = self.interpolator(M1, mkpts, H=_H1, W=_W1)

        # L2-Normalize
        feats = F.normalize(feats, dim=-1)

        # Correct kpt scale
        # mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, 1, -1)

        rw1_var = torch.tensor(rw1, device=x.device, dtype=x.dtype)
        rh1_var = torch.tensor(rh1, device=x.device, dtype=x.dtype)
        mkpts = mkpts * torch.stack([rw1_var, rh1_var], dim=-1).view(1, 1, -1)

        # with torch.no_grad():
        #     mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1, 1, -1)

        # temp_tensor = torch.tensor([rw1, rh1], device=x.device, dtype=x.dtype).view(1, 1, -1)
        # mkpts = mkpts * temp_tensor.detach()

        valid = scores > 0
        return {'keypoints': mkpts[valid],
                'descriptors': feats[valid],
                'scores': scores[valid]}
    
    def detectAndComputeDense(self, x):
        top_k = self.top_k
        mkpts, sc, feats = self.extract_dualscale(x, top_k)
        return {'keypoints': mkpts[0],
        'descriptors': feats[0],
        'scales': sc[0] }
    
    def extract_dualscale(self, x, top_k):
        s1, s2 = 0.6, 1.3
        x1 = F.interpolate(x, scale_factor=s1, align_corners=False, mode='bilinear')
        x2 = F.interpolate(x, scale_factor=s2, align_corners=False, mode='bilinear')

        B, _, _, _ = x.shape

        mkpts_1, feats_1 = self.extractDense(x1, int(top_k*0.20))
        mkpts_2, feats_2 = self.extractDense(x2, int(top_k*0.80))

        mkpts = torch.cat([mkpts_1/s1, mkpts_2/s2], dim=1)
        sc1 = torch.ones(mkpts_1.shape[:2], device=mkpts_1.device) * (1/s1)
        sc2 = torch.ones(mkpts_2.shape[:2], device=mkpts_2.device) * (1/s2)
        sc = torch.cat([sc1, sc2],dim=1)
        feats = torch.cat([feats_1, feats_2], dim=1)

        return mkpts, sc, feats
    
    def extractDense(self, x, top_k ):
        x, rh1, rw1 = self.preprocess_tensor(x)

        M1, K1, H1 = self.detectAndCompute2(x)

        B, C, _H1, _W1 = M1.shape

        xy1 = (self.create_xy(_H1, _W1, M1.device) * 8).expand(B,-1,-1)

        M1 = M1.permute(0,2,3,1).reshape(B, -1, C)
        H1 = H1.permute(0,2,3,1).reshape(B, -1)

        _, top_k = torch.topk(H1, k = min(len(H1[0]), top_k), dim=-1)

        feats = torch.gather( M1, 1, top_k[...,None].expand(-1, -1, 64))
        mkpts = torch.gather(xy1, 1, top_k[...,None].expand(-1, -1, 2))
        # mkpts = mkpts * torch.tensor([rw1, rh1], device=mkpts.device).view(1,-1)
        rw1_var = torch.tensor(rw1, device=x.device, dtype=x.dtype)
        rh1_var = torch.tensor(rh1, device=x.device, dtype=x.dtype)
        mkpts = mkpts * torch.stack([rw1_var, rh1_var], dim=-1).view(1,-1)

        return mkpts, feats

    def detectAndCompute2(self, x):
        """
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:
                feats     ->  torch.Tensor(B, 64, H/8, W/8) dense local features
                keypoints ->  torch.Tensor(B, 65, H/8, W/8) keypoint logit map
                heatmap   ->  torch.Tensor(B,  1, H/8, W/8) reliability map

        """
        # x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
        #dont backprop through normalization
        with torch.no_grad():
            x = x.mean(dim=1, keepdim=True)
            x = self.norm(x)

        # main backbone
        x1 = self.block1(x)
        x2 = self.block2(x1 + self.skip1(x))
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)

        #pyramid fusion
        x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
        x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
       
       
        feats = self.block_fusion(x3 + x4 + x5)
		#heads
        heatmap = self.heatmap_head(feats) # Reliability map
        keypoints = self.keypoint_head(self._unfold2d(x, ws=8)) #Keypoint map logits
        # keypoints = self.keypoint_head(self._unfold2d2(x)) #Keypoint map logits

        return feats, keypoints, heatmap
    
    def create_xy(self, h, w, dev):
        y, x = torch.meshgrid(torch.arange(h, device = dev), 
                                torch.arange(w, device = dev), indexing='ij')
        xy = torch.cat([x[..., None],y[..., None]], -1).reshape(-1,2)
        return xy
    
    def preprocess_tensor(self, x):
        """ Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
        x = x.to(self.dev).float()
        H, W = x.shape[-2:]
        _H, _W = (H // 32) * 32, (W // 32) * 32
        rh, rw = H / _H, W / _W
        x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
        return x, rh, rw
    
    def get_kpts_heatmap(self, kpts, softmax_temp=1.0):
        scores = F.softmax(kpts * softmax_temp, 1)[:, :64]
        B, _, H, W = scores.shape
        heatmap = scores.permute(0, 2, 3, 1).reshape(B, H, W, 8, 8)
        heatmap = heatmap.permute(0, 1, 3, 2, 4).reshape(B, 1, H * 8, W * 8)
        return heatmap
    
    def NMS(self, x, threshold=0.05, kernel_size=5):
        _, _, H, W = x.shape
        local_max = F.max_pool2d(
            x,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            return_indices=False,
        )
        pos = (x == local_max) & (x > threshold)
        return pos.squeeze().nonzero().flip(-1).reshape(1, -1, 2)
    

    def matchDense(self, mkpts0, feats0, sc0, mkpts1, feats1):
        idxs0, idxs1 = self.match(feats0, feats1)
        kpt0, kpt1 = self.refine_matches( mkpts0, feats0, idxs0,
                                    mkpts1, feats1, idxs1,
                                    sc0)
        return {
            "kpt0": kpt0,
            "kpt1": kpt1,
        }


    def match(self, feats1, feats2):
        min_cossim = 0.6
        cossim = feats1 @ feats2.t()
        cossim_t = feats2 @ feats1.t()

        _, match12 = cossim.max(dim=1)
        _, match21 = cossim_t.max(dim=1)

        idx0 = torch.arange(match12.shape[0], device=match12.device)
        mutual = match21[match12] == idx0

        cossim, _ = cossim.max(dim=1)
        good = cossim > min_cossim
        idx0 = idx0[mutual & good]
        idx1 = match12[mutual & good]

        return idx0, idx1


    def refine_matches(self, mkpts0, d0, idx0, mkpts1, d1, idx1, sc0):
        fine_conf = 0.25
        feats1 = d0[idx0]  # [idx0_b[:, 0], idx0_b[:, 1]]
        feats2 = d1[idx1]  # [idx1_b[:, 0], idx1_b[:, 1]]
        mkpts_0 = mkpts0[idx0]  # [idx0_b[:, 0], idx0_b[:, 1]]
        mkpts_1 = mkpts1[idx1]  # [idx1_b[:, 0], idx1_b[:, 1]]
        sc0 = sc0[idx0]  # [idx0_b[:, 0], idx0_b[:, 1]]

        # Compute fine offsets
        offsets = self.fine_matcher(torch.cat([feats1, feats2], dim=-1))
        conf = F.softmax(offsets * 3, dim=-1).max(dim=-1)[0]
        offsets = self.subpix_softmax2d(offsets.view(-1, 8, 8))

        mkpts_0 += offsets * (sc0[:, None])  # *0.9 #* (sc0[:,None])
        mkpts_1 += offsets * (sc0[:, None])  # *0.9 #* (sc0[:,None])

        mask_good = conf > fine_conf
        mkpts_0 = mkpts_0[mask_good]
        mkpts_1 = mkpts_1[mask_good]

        # match_mkpts = torch.cat([mkpts_0, mkpts_1], dim=-1)
        # batch_index = idx0[mask_good]  # idx0_b[mask_good, 0]
        return mkpts_0, mkpts_1
    
    def subpix_softmax2d(self, heatmaps):
        temp=3
        N, H, W = heatmaps.shape
        heatmaps = torch.softmax(temp * heatmaps.view(-1, H * W), -1).view(-1, H, W)
        x, y = torch.meshgrid(torch.arange(H, device=heatmaps.device), torch.arange(W, device=heatmaps.device),
                              indexing='ij')
        x = x - (W // 2)
        y = y - (H // 2)

        coords_x = (x[None, ...] * heatmaps)
        coords_y = (y[None, ...] * heatmaps)
        coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H * W, 2)
        coords = coords.sum(1)

        return coords