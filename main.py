import torch
import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd
import os

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

def euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def resize_frame(frame, max_length):
    """Resize the frame while maintaining aspect ratio."""
    height, width = frame.shape[:2]
    if max(height, width) > max_length:
        if height > width:
            new_height = max_length
            new_width = int((new_height / height) * width)
        else:
            new_width = max_length
            new_height = int((new_width / width) * height)
        frame = cv2.resize(frame, (new_width, new_height))
    return frame

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Motion Detection")
        self.configure(bg='#1f1f1f')
        
        self.logo_img = Image.open('logo.png')
        self.logo_img = self.logo_img.resize((150, 49), Image.LANCZOS)  # 로고 크기 고정
        self.logo_photo = ImageTk.PhotoImage(self.logo_img)
        
        self.logo_label = tk.Label(self, image=self.logo_photo, bg='#1f1f1f')
        self.logo_label.place(x=10, y=10)
        
        self.video_label = tk.Label(self, bg='#1f1f1f')
        self.video_label.pack(pady=10)
        
        self.btn_frame = tk.Frame(self, bg='#1f1f1f')
        self.btn_frame.pack(pady=10)
        
        self.btn_start = tk.Button(self.btn_frame, text="Start Webcam", command=self.start_webcam, bg='#0078d4', fg='#ffffff')
        self.btn_start.pack(side=tk.LEFT, padx=10)
        
        self.btn_stop = tk.Button(self.btn_frame, text="Stop Webcam", command=self.stop_webcam, bg='#0078d4', fg='#ffffff')
        self.btn_stop.pack(side=tk.LEFT, padx=10)
        self.btn_stop.config(state=tk.DISABLED)
        
        self.console = scrolledtext.ScrolledText(self, width=80, height=20, bg='#000000', fg='#e9710d', font=("Courier", 8))
        self.console.pack(pady=10)
        
        self.cap = None
        self.running = False
        self.motion_counter = 0
        self.static_frames = 0
        self.positions = deque(maxlen=2)
        self.threshold = 5
        self.start_time = None
        self.detected_objects = []
        
        self.font_path = "NanumGothic.ttf"  # 한글 폰트 파일 경로 지정
        self.font_size = 45  # 폰트 크기 지정
        
    def log(self, message):
        self.console.insert(tk.END, message + '\n')
        self.console.see(tk.END)
        
    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.log("Error: Cannot open webcam")
            return
        
        self.running = True
        self.start_time = time.time()
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.log("Webcam started")
        self.process_webcam()
        
    def stop_webcam(self):
        self.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        
        if self.cap:
            self.cap.release()
        
        total_time = time.time() - self.start_time
        total_frames = self.motion_counter + self.static_frames
        motion_percentage = (self.motion_counter / total_frames) * 100 if total_frames > 0 else 0
        activity_level = self.determine_activity_level(motion_percentage)
        
        self.log(f"Total Time: {total_time:.2f} seconds")
        self.log(f"Motion Percentage: {motion_percentage:.2f}%")
        self.log(f"Activity Level: {activity_level}")
        
        self.display_summary(motion_percentage, activity_level)
        self.save_results_to_excel(motion_percentage, activity_level)
        
        self.motion_counter = 0
        self.static_frames = 0
        self.positions.clear()
        self.detected_objects.clear()
        
    def determine_activity_level(self, motion_percentage):
        if motion_percentage == 0:
            return "매우 작은 움직임"
        elif 0 < motion_percentage <= 20:
            return "작은 움직임"
        elif 20 < motion_percentage <= 40:
            return "평균적 움직임"
        elif 40 < motion_percentage <= 60:
            return "활동적"
        else:
            return "매우 활동적"
        
    def display_summary(self, motion_percentage, activity_level):
        if not self.cap:
            return

        self.cap = cv2.VideoCapture(0)
        ret, frame = self.cap.read()
        if not ret:
            self.log("Error: Cannot read frame from webcam")
            return

        frame = resize_frame(frame, 640)
        # Add activity level and motion percentage to frame
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        font = ImageFont.truetype(self.font_path, self.font_size)
        
        draw.text((frame_pil.width // 2 - 200, frame_pil.height // 2 - 50), f"활동 레벨: {activity_level}", font=font, fill=(0, 255, 0))
        draw.text((frame_pil.width // 2 - 200, frame_pil.height // 2 + 50), f"활동 점수: {motion_percentage:.2f}%", font=font, fill=(0, 255, 0))
        
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        self.cap.release()
        
    def save_results_to_excel(self, motion_percentage, activity_level):
        # Ensure the result directory exists
        if not os.path.exists('result'):
            os.makedirs('result')
        
        # Get the current time for the filename
        current_time = time.strftime("%Y%m%d%H%M%S")
        filename = f"result/activity_summary_{current_time}.xlsx"
        
        # Create a DataFrame with the results
        data = {
            "활동 레벨": [activity_level],
            "활동 점수": [motion_percentage],
            "디텍팅된 대상": [', '.join(self.detected_objects)],
            "작성된 날짜와 시간": [current_time]
        }
        df = pd.DataFrame(data)
        
        # Save the DataFrame to an Excel file
        df.to_excel(filename, index=False)
        self.log(f"Results saved to {filename}")
        
    def process_webcam(self):
        if not self.running:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.log("Error: Cannot read frame from webcam")
            self.stop_webcam()
            return
        
        frame = resize_frame(frame, 640)
        results = model(frame)
        detections = results.pandas().xyxy[0]
        
        for _, det in detections.iterrows():
            label = det['name']
            if label in ['person', 'cell phone']:
                x_center = (det['xmin'] + det['xmax']) / 2
                y_center = (det['ymin'] + det['ymax']) / 2
                current_position = (x_center, y_center)
                
                if label == 'person':
                    self.positions.append(current_position)
                    if len(self.positions) == 2:
                        distance = euclidean_distance(self.positions[0], self.positions[1])
                        if distance > self.threshold:
                            self.motion_counter += 1
                        else:
                            self.static_frames += 1

                if label not in self.detected_objects:
                    self.detected_objects.append(label)
                
                cv2.rectangle(frame, (int(det['xmin']), int(det['ymin'])), 
                              (int(det['xmax']), int(det['ymax'])), (255, 0, 0), 2)
                cv2.putText(frame, label, (int(det['xmin']), int(det['ymin']) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                cv2.circle(frame, (int(x_center), int(y_center)), 5, (0, 255, 0), -1)
        
        motion_percentage = (self.motion_counter / (self.motion_counter + self.static_frames)) * 100 if (self.motion_counter + self.static_frames) > 0 else 0
        activity_level = self.determine_activity_level(motion_percentage)
        
        # Log activity level and motion percentage to console
        self.log(f"Motion Percentage: {motion_percentage:.2f}%")
        self.log(f"Activity Level: {activity_level}")
        
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_pil.paste(self.logo_img, (10, 10), self.logo_img)
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
        if self.running:
            self.after(10, self.process_webcam)

if __name__ == "__main__":
    app = Application()
    app.mainloop()
