import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
import ultralytics

from pathlib import Path
import json

def check():
    model_path = "/home/mihaib08/Desktop/_research_2025/drone_lab/auto-follow/models/yolo11n-seg_car_sim_simple.pt"
    detector = ultralytics.YOLO(model_path)

    frame_path = "/home/mihaib08/Desktop/_research_2025/drone_lab/auto-follow/assets/reference/images/frame_drone_sim_10m_center.png"
    frame = cv2.imread(frame_path)

    results = detector.predict(frame, stream=False, verbose=False)
    results = results[0]

    if results.boxes and results.boxes.conf is not None and len(results.boxes.conf) > 0:
        best_conf_idx = results.boxes.conf.argmax()
        best_conf = results.boxes.conf[best_conf_idx].item() # Get as float
        # You might want a confidence threshold here too
        # if best_conf > SOME_THRESHOLD:
        target_lost_this_frame = False
        coords = results.boxes.xyxy[best_conf_idx].cpu().numpy()
        
        xy_seg = results.masks.xy[best_conf_idx]
        xy_seg = [list(xy) for xy in xy_seg]
        xy_seg = np.array(xy_seg).astype(np.int32)
        # print(len(xy_seg), type(xy_seg))
        # print(xy_seg)

        x1, y1, x2, y2 = map(int, coords)
        best_target = {
            'conf': best_conf,
            'center': ((x1 + x2) // 2, (y1 + y2) // 2),
            'size': (x2 - x1, y2 - y1),
            'box': (x1, y1, x2, y2),
            "seg": xy_seg
        }

        plotted_frame = frame.copy()
        cv2.fillPoly(plotted_frame, pts=[xy_seg], color=(0, 255, 0))

        ## ellipse
        ## ------------------------------------------

        car_frame = np.zeros(frame.shape[:2], dtype=np.uint8)
        # print(car_frame.shape)

        poly = cv2.fillPoly(car_frame, pts=[xy_seg], color=255)
        # print(poly)
        cv2.imwrite("car.png", car_frame)

        contours, hierarchy = cv2.findContours(car_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cont = contours[0]

        (x,y), (MA,ma), angle = cv2.fitEllipse(cont)
        # ellipse = cv2.fitEllipse(cont)
        # print(ellipse)
        # print(f"{(x,y)}, {(MA,ma)}, {angle}")

        ## ------------------------------------------

        cv2.ellipse(img=plotted_frame,
                    center=(int(x),int(y)),
                    axes=(int(MA/2), int(ma/2)),
                    angle=angle,
                    startAngle=0,
                    endAngle=360,
                    color=(0,0,255))
        # cv2.ellipse(plotted_frame, ellipse, (0, 0, 255))

        cv2.imwrite("check.png", plotted_frame)

def compute_ellipse_axis_keypoints(ellipse):
        (xc_ellipse,yc_ellipse), (MA,ma), angle = ellipse

        kp_center = np.array([xc_ellipse, yc_ellipse])

        MA, ma = MA / 2, ma / 2
        angle_rad = np.radians(angle)
        
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        M_dx, M_dy = MA * cos_a, MA * sin_a
        m_dx, m_dy = ma * -sin_a, ma * cos_a

        kp_MA1 = kp_center + np.array([M_dx, M_dy])
        kp_MA2 = kp_center - np.array([M_dx, M_dy])
        kp_m1 = kp_center + np.array([m_dx, m_dy])
        kp_m2 = kp_center - np.array([m_dx, m_dy])

        return kp_MA1.astype(int), \
               kp_MA2.astype(int), \
               kp_m1.astype(int), \
               kp_m2.astype(int)

def retrieve_keypoints(model_path: Path, frame_path: Path):
    detector = ultralytics.YOLO(model_path)
    
    frame = cv2.imread(str(frame_path))
    print(frame.shape)

    print(frame_path.parent)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_color = (0,0,0)

    results = detector.predict(frame, stream=False, verbose=False)
    results = results[0]

    ellipse_points = []
    bbox_points = []
    if results.boxes and results.boxes.conf is not None and len(results.boxes.conf) > 0:
        best_conf_idx = results.boxes.conf.argmax()
        best_conf = results.boxes.conf[best_conf_idx].item() # Get as float

        coords = results.boxes.xyxy[best_conf_idx].cpu().numpy()
        xl, yl, xr, yr = map(int, coords)
        bbox_points = [(xl, yl), (xl, yr), (xr, yl), (xr, yr)]
        
        xy_seg = results.masks.xy[best_conf_idx]
        xy_seg = [list(xy) for xy in xy_seg]
        xy_seg = np.array(xy_seg).astype(np.int32)

        ## plotted frame
        ## ------------------------------------------
        plotted_frame = frame.copy()
        cv2.fillPoly(plotted_frame, pts=[xy_seg], color=(0, 255, 0))
        ## ------------------------------------------


        ## ellipse
        ## ------------------------------------------

        car_frame = np.zeros(frame.shape[:2], dtype=np.uint8)
        # print(car_frame.shape)

        poly = cv2.fillPoly(car_frame, pts=[xy_seg], color=255)
        # print(poly)
        cv2.imwrite("car.png", car_frame)

        contours, hierarchy = cv2.findContours(car_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cont = contours[0]

        ellipse = cv2.fitEllipse(cont)
        (x,y), (MA,ma), angle = ellipse
        ## ------------------------------------------

        cv2.ellipse(img=plotted_frame,
                    center=(int(x),int(y)),
                    axes=(int(MA/2), int(ma/2)),
                    angle=angle,
                    startAngle=0,
                    endAngle=360,
                    color=(0,0,255))
        # cv2.ellipse(plotted_frame, ellipse, (0, 0, 255))

        kp_M1, kp_M2, kp_m1, kp_m2 = compute_ellipse_axis_keypoints(ellipse)

        cv2.line(plotted_frame, tuple(kp_m1), tuple(kp_m2), (255, 255, 0), 1)

        text = "0"
        text_x, text_y = kp_m1[0], kp_m1[1] - 15
        cv2.putText(plotted_frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.circle(plotted_frame, tuple(kp_m1), 5, (255,0,0), -1)
        ellipse_points.append(tuple(map(int, kp_m1)))
        
        text = "1"
        text_x, text_y = kp_m2[0], kp_m2[1] + 15
        cv2.putText(plotted_frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.circle(plotted_frame, tuple(kp_m2), 5, (0,255,0), -1)
        ellipse_points.append(tuple(map(int, kp_m2)))

        cv2.line(plotted_frame, tuple(kp_M1), tuple(kp_M2), (255, 255, 0), 1)
        
        text = "2"
        text_x, text_y = kp_M1[0] - 15, kp_M1[1]
        cv2.putText(plotted_frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.circle(plotted_frame, tuple(kp_M1), 5, (0,0,255), -1)
        ellipse_points.append(tuple(map(int, kp_M1)))
        
        text = "3"
        text_x, text_y = kp_M2[0] + 15, kp_M2[1]
        cv2.putText(plotted_frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.circle(plotted_frame, tuple(kp_M2), 5, (255, 255, 255), -1)
        ellipse_points.append(tuple(map(int, kp_M2)))

        xyxy = results.boxes.xyxy[best_conf_idx].cpu().numpy()
        xl, yl, xr, yr = map(int, xyxy)
        xc = (xl + xr) // 2
        yc = (yl + yr) // 2
        object_center = (xc, yc)
        
        text = "4"
        text_x, text_y = object_center[0], object_center[1] - 15
        cv2.putText(plotted_frame, text, (text_x, text_y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.circle(plotted_frame, object_center, 5, (51,87,255), -1)
        ellipse_points.append(object_center)

        cv2.imwrite("check_ellipse.png", plotted_frame)

    print(ellipse_points)
    return bbox_points, ellipse_points



if __name__ == "__main__":
    # check()

    model_path = Path("/home/mihaib08/Desktop/_research_2025/drone_lab/auto-follow/models/yolo11n-seg_car_sim_simple.pt")
    frame_path = Path("/home/mihaib08/Desktop/_research_2025/drone_lab/auto-follow/assets/reference/images/frame_drone_sim_10m_center.png")

    bbox_points, ellipse_points = retrieve_keypoints(model_path, frame_path)
    data = {
        "ellipse_points": ellipse_points,
        "bbox_points": bbox_points
    }

    with open(f"{frame_path.stem}.json", "w") as f:
        json.dump(data, f, indent=4)
