import numpy as np
import cv2
import joblib
import pandas as pd


def my_draw_flow(img, flow, step, boxes, nums, mode):
    data = pd.DataFrame()
    if(mode == 3):
        model = joblib.load("finalized_model.sav")
        scaler = joblib.load("scaler.sav")
    wh = np.flip(img.shape[0:2])
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    X = []
    for i in range(nums):
        x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
        x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
        y, x = np.mgrid[x1y1[1]:x2y2[1]:step, x1y1[0]:x2y2[0]:step].reshape(2, -1).astype(int)
        fx, fy = flow[y, x].T
        if(mode == 1):  # simple optical_flow
            lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
            lines = np.int32(lines + 0.5)
            cv2.polylines(vis, lines, 0, (0, 0, 255))
        else:  # aggregatede optical flow or other options
            fx = abs(x1y1[0]-x2y2[0])*np.mean(fx)/4
            fy = abs(x1y1[1]-x2y2[1])*np.mean(fy)/4
            x = (x1y1[0]+x2y2[0])/2
            y = (x1y1[1] + x2y2[1]) / 2
            temp = [fx, fy, x, y, 0]  # left = 0
            data = data.append([temp])
            if(mode == 3):
                X.append([fx, fy, x, y])
            cv2.arrowedLine(vis, (int(x), int(y)), (int(x+fx),
                                                    int(y+fy)), (0, 0, 255, 0), thickness=2)
    if(mode == 3):
        result = []
        if(len(X) != 0):
            X = scaler.transform(X)
            result = model.predict(X)
        return vis, result, data  # result ; 0 = unsafe; 1 = safe
    else:
        return vis, data


def opticalFlow(image1, image2):
    prevgray = image1
    gray = image2
    flow = cv2.calcOpticalFlowFarneback(
        prevgray, gray, None, 0.5, 5, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    return(-flow)


if __name__ == '__main__':
    try:
        print("This module is not callable as main instance")
    except SystemExit:
        pass
