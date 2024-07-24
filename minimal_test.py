"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Minimal example of how to use XFeat.
"""

import numpy as np
import os
import torch
import tqdm
import cv2
from time import time


# from modules.xfeat import XFeat
from module_onnx import model

# os.environ['CUDA_VISIBLE_DEVICES'] = '' #Force CPU, comment for GPU


def putText(canvas, text, org, fontFace, fontScale, textColor, borderColor, thickness, lineType):
    # Draw the border
    cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
                color=borderColor, thickness=thickness+2, lineType=lineType)
    # Draw the text
    cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
                color=textColor, thickness=thickness, lineType=lineType)
    
def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)

def parse_input(x):
    if len(x.shape) == 3:
        x = x[None, ...]
    if isinstance(x, np.ndarray):
        x = torch.tensor(x).permute(0,3,1,2)
    return x


# 读取 assets 文件夹内图片
# img1 = cv2.imread("assets/sacre_coeur1.jpg")
# img2 = cv2.imread("assets/sacre_coeur2.jpg")
img1 = cv2.imread("assets/ref.png")
img2 = cv2.imread("assets/tgt.png")
t0 = time()


# matches = xfeat.match_xfeat_star(img1, img2, top_k = 400)
# points1 = matches[0]
# points2 = matches[1]

xfeatModel = model.XFeatModel()
xfeatModel.eval()
xfeatModel.load_state_dict(torch.load('./weights/xfeat.pt'))

res = xfeatModel(parse_input(img1))
kpts1, desc1, sc1 = res['keypoints'], res['descriptors'], res['scores']
for kp in kpts1:
    x, y = kp
    cv2.circle(img1, (int(x), int(y)), 2, (0, 255, 0), -1)
# plt.figure(figsize=(20, 20))
cv2.imshow("img", img1)
cv2.waitKey(0)



# # 计算特征
# output1 = xfeat.detectAndCompute(torch.tensor(img1).permute(2,0,1).float()[None], top_k = 800)[0]
# output2 = xfeat.detectAndCompute(torch.tensor(img2).permute(2,0,1).float()[None], top_k = 800)[0]
# # output1 = xfeat.detectAndComputeDense(torch.tensor(img1).permute(2,0,1).float()[None], top_k = 800)[0]
# # output2 = xfeat.detectAndComputeDense(torch.tensor(img2).permute(2,0,1).float()[None], top_k = 800)[0]
# kpts1, descs1 = output1['keypoints'], output1['descriptors']
# kpts2, descs2 = output2['keypoints'], output2['descriptors']

# # 匹配特征
# idx1, idx2 = xfeat.match(descs1, descs2, 0.82)

# points1 = kpts1[idx1].cpu().numpy()
# points2 = kpts2[idx2].cpu().numpy()


# # Find homography
# H, inliers = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, 4.0, maxIters=700, confidence=0.995)
# inliers = inliers.flatten() > 0

# if inliers.sum() < 50:
#     H = None

# kp1 = [cv2.KeyPoint(p[0],p[1], 5) for p in points1[inliers]]
# kp2 = [cv2.KeyPoint(p[0],p[1], 5) for p in points2[inliers]]
# good_matches = [cv2.DMatch(i,i,0) for i in range(len(kp1))]

# print(1.0 / (time() - t0))
# # Draw matches
# matched_frame = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, matchColor=(0, 200, 0), flags=2)

# color = (240, 89, 169)
# # Adding captions on the top frame canvas
# putText(canvas=matched_frame, text="XFeat Matches: %d"%(len(good_matches)), org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
#     fontScale=0.9, textColor=(0,0,0), borderColor=color, thickness=1, lineType=cv2.LINE_AA)

# cv2.imshow("XFeat Matches", matched_frame)
# cv2.waitKey(0)