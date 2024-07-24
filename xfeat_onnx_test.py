import os
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import onnxruntime
import torch

def parse_input(x):
    if len(x.shape) == 3:
        x = x[None, ...]

    if isinstance(x, np.ndarray):
        x = torch.tensor(x/255.0, dtype=torch.float).permute(0,3,1,2)
    return x


def main():
    # Setting variables
    dense = False  # Dense keypoints extraction
    multiscale = False  # Dense mode: enable multiscale

    # Get image and load
    fname_img = "assets/ref.png"
    img = cv2.imread(fname_img)
    # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create model
    fname_model = "weights/xfeatSim.onnx"
    match_model = 'weights/xfeat_match_480_640.onnx'

    session = onnxruntime.InferenceSession(fname_model)
    input_names = session.get_inputs()[0].name
    output_names = [node.name for node in session.get_outputs()]

    # Parse numpy array to tensor
    # img_tensor = np.array([imgRGB.transpose(2, 0, 1)], dtype=np.float32) / 255.0
    img_tensor = parse_input(img).numpy()

    # Run model
    results = session.run(output_names, {"images": img_tensor})
    # img = draw_points(img, results[0])
    # kpts, descr, scores = results[0], results[1], results[2]
    kpts, descs = results[0], results[1]
    # print(kpts.type, descs.type)
    print(kpts.shape, descs.shape)

    # Show
    for kp in kpts:
        x, y = kp
        cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1)
    # plt.figure(figsize=(20, 20))
    cv2.imshow("img", img)
    cv2.waitKey(0)



if __name__ == "__main__":
    main()