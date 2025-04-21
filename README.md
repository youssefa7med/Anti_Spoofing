# 🛡️ Anti-Spoofing Face Detection with YOLOv8

![Anti-Spoofing Demo](https://www.hackread.com/wp-content/uploads/2016/01/bypassing-lastpasss-security-a-phishing-attack-would-serve-just-right.gif)

This project focuses on detecting whether a detected face is **real (live)** or **fake (e.g., photo or video spoof)** using a YOLOv8 model. The application is developed in Python with real-time webcam processing and visual feedback using OpenCV and cvzone.

---

## 🔍 Overview

The Anti-Spoofing system uses deep learning techniques to enhance security in facial recognition systems. It helps prevent unauthorized access through printed photos, mobile screen images, or deepfake videos by identifying spoofed inputs in real-time.

---

## 💡 Key Features

- **Real-Time Detection**  
  Fast and responsive webcam-based detection with YOLOv8.

- **Classification**  
  Differentiates between `REAL` and `FAKE` faces using a custom-trained model.

- **Visual Feedback**  
  Uses colored bounding boxes and labels for easy identification:
  - 🟢 Green box for REAL
  - 🔴 Red box for FAKE

- **FPS Monitoring**  
  Displays frames-per-second for performance tracking.

---

## 🧠 Model Details

- Framework: **Ultralytics YOLOv8**
- Classes: `["fake", "real"]`
- Format: `.pt` (PyTorch trained weights)

---

## 🛠️ Technologies Used

- **Python**: Core language
- **OpenCV**: For video and image processing
- **cvzone**: For visual overlays and enhancements
- **Ultralytics**: For YOLOv8 object detection
- **Torch**: Deep learning framework

---

## 🖥️ Live Demo (Offline)

Run the app locally to experience real-time anti-spoofing.

---

## 🚀 Getting Started

### Prerequisites

Ensure Python 3.8+ is installed. Install the following dependencies:

```bash
pip install -r requirements.txt
```

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/youssefa7med/anti-spoofing-yolo.git
   cd anti-spoofing-yolo
   ```

2. **Add Your YOLOv8 Model**  
   Place your `best.pt` file in the project directory.

3. **Run the Script**:
   ```bash
   python detect.py
   ```

---

## 📂 Project Structure

```
.
├── best.pt               # Trained YOLOv8 model
├── detect.py             # Main real-time detection script
├── README.md             # Project documentation
├── requirements.txt      # Required libraries
```

---

## 📸 Output Example

- 🟢 `REAL 98%`: Detected face is a real person.
- 🔴 `FAKE 95%`: Detected spoofed face (photo, screen, etc.).

---

## 📈 Future Enhancements

- 🔐 Integration with authentication systems
- 📱 Mobile or Jetson Nano deployment
- 🔊 Audio alerts on detection
- 📊 Logging detected faces and timestamps

---

## 🧑‍💻 Contributing

Contributions are welcome! Fork the repo, make your changes, and submit a pull request.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

Thanks to the open-source community and Ultralytics for the YOLOv8 framework, and the contributors behind datasets used in training.

---

> Made with 💻 by [Youssef Ahmed](https://github.com/youssefa7med)

