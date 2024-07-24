from time import sleep, time
import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import torch
import argparse
from modules.xfeat import XFeat
import os
# from modules.model import *

# os.environ['CUDA_VISIBLE_DEVICES'] = ''
kpt = 200
width = 640
height = 480


def argparser():
    parser = argparse.ArgumentParser(description="Configurations for the real-time matching demo.")
    parser.add_argument('--width', type=int, default=width, help='Width of the video capture stream.')
    parser.add_argument('--height', type=int, default=height, help='Height of the video capture stream.')
    parser.add_argument('--max_kpts', type=int, default=kpt, help='Maximum number of keypoints.')
    parser.add_argument('--method', type=str, choices=['ORB', 'SIFT', 'XFeat'], default='XFeat', help='Local feature detection method to use.')
    return parser.parse_args()

class FrameGrabber():
    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        # self.config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 15)
        # self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 15)
        self.profile = self.pipeline.start(self.config)
        self.device = self.profile.get_device()
        depth_sensor = self.device.query_sensors()[0]
        if depth_sensor.supports(rs.option.emitter_enabled):
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
        self.frame = None

    def run(self):
        self.running = True
        while self.running:
            # print("Starting frame grabber")
            frames = self.pipeline.wait_for_frames()

            # 灰度图像获取
            # self.frame = frames.get_infrared_frame(1)

            # 彩色图像获取
            self.frame = frames.get_color_frame()
            # sleep(0.03)

    def stop(self):
        self.running = False
        self.pipeline.stop()
    
    def get_last_frame(self):
        # img = np.asanyarray(self.frame.get_data())
        # img = img[None]
        return np.asanyarray(self.frame.get_data())

def show_frame(f):
    while True:
        current_frame = f.get_last_frame()
        cv2.imshow('frame', current_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            f.stop()
            cv2.destroyAllWindows()
            break

def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)

class CVWrapper():
    def __init__(self, mtd):
        self.mtd = mtd
    def detectAndCompute(self, x, mask=None):
        # return self.mtd.detectAndCompute(torch.tensor(x).permute(2,0,1).float()[None])[0]
        return self.mtd.detectAndCompute(numpy_image_to_torch(x)[None])[0]

class Method:
    def __init__(self, descriptor, matcher):
        self.descriptor = descriptor
        self.matcher = matcher

def init_method(method, max_kpts):
        Method(descriptor=CVWrapper(XFeat(top_k = max_kpts)), matcher=XFeat())

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
        self.frame_grabber = FrameGrabber()

        #Homography params
        self.min_inliers = 50
        self.ransac_thr = 4.0

        #FPS check
        self.FPS = 0
        self.time_list = []
        self.max_cnt = 15 #avg FPS over this number of frames

        #Set local feature method here -- we expect cv2 or Kornia convention
        self.method = Method(descriptor=CVWrapper(XFeat(top_k = kpt)), matcher=XFeat())
        
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
        if top_frame.ndim == 3:
            top_frame_canvas[0:self.height, 0:self.width*2] = top_frame
        elif top_frame.ndim == 2:
            top_frame_canvas[0:self.height, 0:self.width*2, 0] = top_frame
            top_frame_canvas[0:self.height, 0:self.width*2, 1] = top_frame
            top_frame_canvas[0:self.height, 0:self.width*2, 2] = top_frame
        
        # Adding captions on the top frame canvas
        self.putText(canvas=top_frame_canvas, text="Reference Frame:", org=(10, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)

        self.putText(canvas=top_frame_canvas, text="Target Frame:", org=(650, 30), fontFace=self.font, 
                    fontScale=self.font_scale,  textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)
        
        self.draw_quad(top_frame_canvas, self.corners)
        
        return top_frame_canvas

    def process(self):
        # top_frame_canvas = np.zeros((480, 1280, 3), dtype=np.uint8)
        # kpts0, kpts1 = self.ref_precomp['keypoints'], self.current['keypoints']
        # for kp in kpts0:
        #     x, y = kp
        #     cv2.circle(self.ref_frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        # for kp in kpts1:
        #     x, y = kp
        #     cv2.circle(self.current_frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        # self.draw_quad(top_frame_canvas, self.corners)
        # top_frame = np.hstack((self.ref_frame, self.current_frame))
        # top_frame_canvas[0:self.height, 0:self.width*2] = top_frame
        # # Match features and draw matches on the bottom frame
        bottom_frame = self.match_and_draw(self.ref_frame, self.current_frame)
        # canvas = np.vstack((top_frame_canvas, bottom_frame))
        canvas = bottom_frame
        cv2.imshow(self.window_name, canvas)

        # self.current = self.method.descriptor.detectAndCompute(self.current_frame)
        # # kpts1, descs1 = self.ref_precomp['keypoints'], self.ref_precomp['descriptors']
        # kpts2, descs2 = self.current['keypoints'], self.current['descriptors']
        # for kp in kpts2:
        #     x, y = kp
        #     cv2.circle(self.current_frame, (int(x), int(y)), 2, (0, 255, 0), -1)
        # # plt.figure(figsize=(20, 20))
        # cv2.imshow("img", self.current_frame)

    def match_and_draw(self, ref_frame, current_frame):

        matches, good_matches = [], []
        kp1, kp2 = [], []
        points1, points2 = [], []

        # Detect and compute features
        if self.args.method in ['SIFT', 'ORB']:
            kp1, des1 = self.ref_precomp
            kp2, des2 = self.method.descriptor.detectAndCompute(current_frame, None)
        else:
            self.current = self.method.descriptor.detectAndCompute(current_frame)
            kpts1, descs1 = self.ref_precomp['keypoints'], self.ref_precomp['descriptors']
            kpts2, descs2 = self.current['keypoints'], self.current['descriptors']
            idx0, idx1 = self.method.matcher.match(descs1, descs2, 0.82)
            points1 = kpts1[idx0].cpu().numpy()
            points2 = kpts2[idx1].cpu().numpy()

            # im_set2 = self.method.matcher.parse_input(current_frame)
            # #Compute coarse feats
            # self.current = self.method.matcher.detectAndComputeDense(im_set2, top_k=4800)
            # #Match batches of pairs
            # idxs_list = self.method.matcher.batch_match(self.ref_precomp['descriptors'], self.current['descriptors'] )
            # B = len(im_set2)
            # # print(B)
            # #Refine coarse matches
            # #this part is harder to batch, currently iterate
            # matches = []
            # for b in range(B):
            #     matches.append(self.method.matcher.refine_matches(self.ref_precomp, self.current, matches = idxs_list, batch_idx=b))
            # if B <= 1:
            #     # matches = (matches[0][:, :2].cpu().numpy(), matches[0][:, 2:].cpu().numpy())
            #     matches = (matches[0][:, :2].cpu().detach().numpy(), matches[0][:, 2:].cpu().detach().numpy())
            # points1 = matches[0]
            # points2 = matches[1]

        if len(kp1) > 10 and len(kp2) > 10 and self.args.method in ['SIFT', 'ORB']:
            # Match descriptors
            matches = self.method.matcher.match(des1, des2)

            if len(matches) > 10:
                points1 = np.zeros((len(matches), 2), dtype=np.float32)
                points2 = np.zeros((len(matches), 2), dtype=np.float32)

                for i, match in enumerate(matches):
                    points1[i, :] = kp1[match.queryIdx].pt
                    points2[i, :] = kp2[match.trainIdx].pt

        # cv2.drawKeypoints(current_frame, kp2, current_frame, color=(0, 200, 0))
        if len(points1) > 10 and len(points2) > 10:
            # Find homography
            self.H, inliers = cv2.findHomography(points1, points2, cv2.USAC_MAGSAC, self.ransac_thr, maxIters=700, confidence=0.995)
            inliers = inliers.flatten() > 0

            if inliers.sum() < self.min_inliers:
                self.H = None

            if self.args.method in ["SIFT", "ORB"]:
                good_matches = [m for i,m in enumerate(matches) if inliers[i]]
            else:
                kp1 = [cv2.KeyPoint(p[0],p[1], 5) for p in points1[inliers]]
                kp2 = [cv2.KeyPoint(p[0],p[1], 5) for p in points2[inliers]]
                good_matches = [cv2.DMatch(i,i,0) for i in range(len(kp1))]

            # Draw matches
            matched_frame = cv2.drawMatches(ref_frame, kp1, current_frame, kp2, good_matches, None, matchColor=(0, 200, 0), flags=2)
            
        else:
            matched_frame = np.hstack([ref_frame, current_frame])

        color = (240, 89, 169)
        # # Add a colored rectangle to separate from the top frame
        cv2.rectangle(matched_frame, (2, 2), (self.width*2-2, self.height-2), color, 5)

        # Adding captions on the top frame canvas
        self.putText(canvas=matched_frame, text="%s Matches: %d"%(self.args.method, len(good_matches)), org=(10, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)
        
                # Adding captions on the top frame canvas
        self.putText(canvas=matched_frame, text="FPS (registration): {:.1f}".format(self.FPS), org=(650, 30), fontFace=self.font, 
            fontScale=self.font_scale, textColor=(0,0,0), borderColor=color, thickness=1, lineType=self.line_type)

        return matched_frame

    def main_loop(self):
        self.current_frame = self.frame_grabber.get_last_frame()
        # print(self.current_frame.shape)
        self.ref_frame = self.current_frame.copy()
        self.current = self.method.descriptor.detectAndCompute(self.current_frame, None)
        self.ref_precomp = self.current.copy() #Cache ref features
        # im_set1 = self.method.matcher.parse_input(self.ref_frame)
        # self.ref_precomp = self.method.matcher.detectAndComputeDense(im_set1, top_k=4800)
        self.process()
        while True:
            if self.current_frame is None:
                break

            t0 = time()
            self.process()
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.frame_grabber.stop()
                break
            elif key == ord('s'):
                self.ref_frame = self.current_frame.copy()  # Update reference frame
                self.ref_precomp = self.method.descriptor.detectAndCompute(self.ref_frame, None) #Cache ref features

            self.ref_frame = self.current_frame.copy()  # Update reference frame
            self.ref_precomp = self.current.copy() #Cache ref features
            self.current_frame = self.frame_grabber.get_last_frame()

            #Measure avg. FPS
            self.time_list.append(time()-t0)
            if len(self.time_list) > self.max_cnt:
                self.time_list.pop(0)
            self.FPS = 1.0 / np.array(self.time_list).mean()
            # print(self.FPS)
        self.cleanup()

    def cleanup(self):
        self.frame_grabber.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    demo = MatchingDemo(args = argparser())
    # demo.main_loop()
    thread1 = threading.Thread(target=demo.frame_grabber.run)
    # thread2 = threading.Thread(target=demo.main_loop)
    thread1.start()
    sleep(1)
    demo.main_loop()
    # thread2.start()
    # thread1.join()
    # thread2.join()



# if __name__ == "__main__":
#     f = FrameGrabber()
#     thread1 = threading.Thread(target=f.run)
#     thread2 = threading.Thread(target=show_frame, args=(f,))
#     thread1.start()
#     sleep(0.5)
#     thread2.start()

#     thread1.join()
#     thread2.join()
