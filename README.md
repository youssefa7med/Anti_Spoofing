# 🛡️ Anti-Spoofing Face Detection with YOLOv8

This project aims to detect whether a face presented to the camera is **real (live)** or **fake (e.g., printed or on a screen)** using a YOLOv8 model trained on anti-spoofing datasets. It combines `ultralytics` YOLOv8 with `cvzone`'s `FaceDetector` for precise facial region analysis.

---

## 📸 Demo

![Anti-Spoofing Demo]([https://media.giphy.com/media/3o7btY60hzzp0hYb5u/giphy.gif](https://www.hackread.com/wp-content/uploads/2016/01/bypassing-lastpasss-security-a-phishing-attack-would-serve-just-right.gif))

---

## 🔍 Features

- 🚀 Real-time face anti-spoofing detection.
- 🎯 High accuracy using a custom-trained YOLOv8 model.
- 🧠 Detects whether the detected face is **real** or **fake**.
- 📦 Easy to integrate and extend.

---

## 🧠 Model

- Framework: **YOLOv8** from [Ultralytics](https://github.com/ultralytics/ultralytics)
- Classes: `["fake", "real"]`
- Trained on a custom anti-spoofing dataset (e.g., CASIA, CelebA-Spoof, etc.).

---

## 🛠️ Installation

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/anti-spoofing-yolo.git
cd anti-spoofing-yolo
