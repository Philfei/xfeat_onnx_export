"""
	"XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
	https://www.verlab.dcc.ufmg.br/descriptors/xfeat_cvpr24/

    Real-time homography estimation demo. Note that scene has to be planar or just rotate the camera for the estimation to work properly.
"""

import cv2
import numpy as np
import torch

from time import time, sleep
import argparse, sys, tqdm
import threading
import os

from modules.xfeat import XFeat

# os.environ['CUDA_VISIBLE_DEVICES'] = '' #Force CPU, comment for GPUs

def argparser():
    parser = argparse.ArgumentParser(description="Configurations for the real-time matching demo.")
    parser.add_argument('--width', type=int, default=640, help='Width of the video capture stream.')
    parser.add_argument('--height', type=int, default=480, help='Height of the video capture stream.')
    parser.add_argument('--max_kpts', type=int, default=150, help='Maximum number of keypoints.')
    return parser.parse_args()
    
class getImg():
    def __init__(self) -> None:
        self.imgPath = 'H:\\datas\\realsense\\left\\'
        self.savePath = 'H:\\datas\\realsense\\left_xfeat\\'
        self.imgCount = 500
    
    def getFrame(self, count):
        framePath = self.imgPath + str(count) + '.png'
        self.img = cv2.imread(framePath)
        return self.img

class CVWrapper():
    def __init__(self, mtd):
        self.mtd = mtd
    def detectAndCompute(self, x, mask=None):
        return self.mtd.detectAndCompute(torch.tensor(x).permute(2,0,1).float()[None])[0]

class Method:
    def __init__(self, descriptor, matcher):
        self.descriptor = descriptor
        self.matcher = matcher

def init_method(max_kpts):
    return Method(descriptor=CVWrapper(XFeat(top_k = max_kpts)), matcher=XFeat())


class MatchingDemo:
    def __init__(self, args):
        self.args = args
        self.width = args.width
        self.height = args.height
        self.ref_frame = None
        self.ref_precomp = [[],[]]
        self.corners = [[50, 50], [640-50, 50], [640-50, 480-50], [50, 480-50]]
        self.current_frame = None
        self.H = None

        #Init frame grabber thread
        # self.frame_grabber = FrameGrabber(self.cap)
        # self.frame_grabber.start()

        self.grabber = getImg()

        #Homography params
        self.min_inliers = 50
        self.ransac_thr = 4.0

        #FPS check
        self.FPS = 0
        self.time_list = []
        self.max_cnt = 30 #avg FPS over this number of frames

        #Set local feature method here -- we expect cv2 or Kornia convention
        # self.method = init_method(max_kpts=args.max_kpts)
        self.method = Method(descriptor=CVWrapper(XFeat(top_k = 150)), matcher=XFeat())
        
        # Setting up font for captions
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.9
        self.line_type = cv2.LINE_AA
        self.line_color = (0,255,0)
        self.line_thickness = 3

        self.window_name = "Real-time matching - Press 's' to set the reference frame."

        # Removes toolbar and status bar
        cv2.namedWindow(self.window_name, flags=cv2.WINDOW_GUI_NORMAL)
        # Set the window size
        cv2.resizeWindow(self.window_name, self.width*2, self.height*2)
        #Set Mouse Callback
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.total_t = 0

    def draw_quad(self, frame, point_list):
        if len(self.corners) > 1:
            for i in range(len(self.corners) - 1):
                cv2.line(frame, tuple(point_list[i]), tuple(point_list[i + 1]), self.line_color, self.line_thickness, lineType = self.line_type)
            if len(self.corners) == 4:  # Close the quadrilateral if 4 corners are defined
                cv2.line(frame, tuple(point_list[3]), tuple(point_list[0]), self.line_color, self.line_thickness, lineType = self.line_type)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.corners) >= 4:
                self.corners = []  # Reset corners if already 4 points were clicked
            self.corners.append((x, y))

    def putText(self, canvas, text, org, fontFace, fontScale, textColor, borderColor, thickness, lineType):
        # Draw the border
        cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
                    color=borderColor, thickness=thickness+2, lineType=lineType)
        # Draw the text
        cv2.putText(img=canvas, text=text, org=org, fontFace=fontFace, fontScale=fontScale, 
                    color=textColor, thickness=thickness, lineType=lineType)

    def warp_points(self, points, H, x_offset = 0):
        points_np = np.array(points, dtype='float32').reshape(-1,1,2)

        warped_points_np = cv2.perspectiveTransform(points_np, H).reshape(-1, 2)
        warped_points_np[:, 0] += x_offset
        warped_points = warped_points_np.astype(int).tolist()
        
        return warped_points

    def create_top_frame(self):
        top_frame_canvas = np.zeros((480, 1280, 3), dtype=np.uint8)
        top_frame = np.hstack((self.ref_frame, self.current_frame))
        color = (3, 186, 252)
        cv2.rectangle(top_frame, (2, 2), (self.width*2-2, self.height-2), color, 5)  # Orange color line as a separator
        top_frame_canvas[0:self.height, 0:self.width*2] = top_frame
        
        # Adding captions on the top frame canvas
        self.putText(canvas=top_frame_canvas, text="Reference Frame:", org=(10, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)

        self.putText(canvas=top_frame_canvas, text="Target Frame:", org=(650, 30), fontFace=self.font, 
                    fontScale=self.font_scale,  textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)
        
        self.draw_quad(top_frame_canvas, self.corners)
        
        return top_frame_canvas

    def process(self):
        # Create a blank canvas for the top frame
        top_frame_canvas = self.create_top_frame()

        # Match features and draw matches on the bottom frame
        bottom_frame = self.match_and_draw(self.ref_frame, self.current_frame)

        # Draw warped corners
        if self.H is not None and len(self.corners) > 1:
            self.draw_quad(top_frame_canvas, self.warp_points(self.corners, self.H, self.width))

        # Stack top and bottom frames vertically on the final canvas
        canvas = np.vstack((top_frame_canvas, bottom_frame))

        cv2.imshow(self.window_name, canvas)

    def match_and_draw(self, ref_frame, current_frame):

        matches, good_matches = [], []
        kp1, kp2 = [], []
        points1, points2 = [], []

        # Detect and compute features

        current = self.method.descriptor.detectAndCompute(current_frame)
        kpts1, descs1 = self.ref_precomp['keypoints'], self.ref_precomp['descriptors']
        kpts2, descs2 = current['keypoints'], current['descriptors']
        idx0, idx1 = self.method.matcher.match(descs1, descs2, 0.82)
        points1 = kpts1[idx0].cpu().numpy()
        points2 = kpts2[idx1].cpu().numpy()


        if len(points1) > 10 and len(points2) > 10:
            # Find homography
            self.H, inliers = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, self.ransac_thr, maxIters=700, confidence=0.995)
            inliers = inliers.flatten() > 0

            if inliers.sum() < self.min_inliers:
                self.H = None

            kp1 = [cv2.KeyPoint(p[0],p[1], 5) for p in points1[inliers]]
            kp2 = [cv2.KeyPoint(p[0],p[1], 5) for p in points2[inliers]]
            good_matches = [cv2.DMatch(i,i,0) for i in range(len(kp1))]

            # Draw matches
            matched_frame = cv2.drawMatches(ref_frame, kp1, current_frame, kp2, good_matches, None, matchColor=(0, 200, 0), flags=2)
            
        else:
            matched_frame = np.hstack([ref_frame, current_frame])

        color = (240, 89, 169)

        # Add a colored rectangle to separate from the top frame
        cv2.rectangle(matched_frame, (2, 2), (self.width*2-2, self.height-2), color, 5)

        # Adding captions on the top frame canvas
        self.putText(canvas=matched_frame, text="XFeat Matches: %d"%(len(good_matches)), org=(10, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)
        
        # Adding captions on the top frame canvas
        self.putText(canvas=matched_frame, text="FPS (registration): {:.1f}".format(self.FPS), org=(650, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)

        return matched_frame

    def main_loop(self):
        # self.current_frame = self.frame_grabber.get_last_frame()
        imgNum = self.grabber.imgCount
        self.current_frame = self.grabber.getFrame(1)
        self.ref_frame = self.current_frame.copy()
        t0 = time()
        self.ref_precomp = self.method.descriptor.detectAndCompute(self.ref_frame, None) #Cache ref features
        self.total_t = time() - t0
        for count in range(2, imgNum):
            if self.current_frame is None:
                break

            t0 = time()
            self.process()

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.ref_frame = self.current_frame.copy()  # Update reference frame
                self.ref_precomp = self.method.descriptor.detectAndCompute(self.ref_frame, None) #Cache ref features

            # self.ref_frame = self.current_frame.copy()
            self.current_frame = self.grabber.getFrame(count)

            self.total_t += time() - t0
            # print('total frame: {}, total time: {:.3f}s'.format(count, self.total_t))
            #Measure avg. FPS
            self.time_list.append(time()-t0)
            if len(self.time_list) > self.max_cnt:
                self.time_list.pop(0)
            self.FPS = 1.0 / np.array(self.time_list).mean()
        
        self.cleanup()

    def cleanup(self):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Running demo...")
    demo = MatchingDemo(args = argparser())
    demo.main_loop()
