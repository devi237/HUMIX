# HUMIX 🎯
### Smart Human Density & Movement Monitoring System

![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green?style=flat-square&logo=opencv)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=flat-square&logo=flask)
![YOLO](https://img.shields.io/badge/YOLOv12-Detection-red?style=flat-square)
![Arduino](https://img.shields.io/badge/Arduino-LED%20Control-00979D?style=flat-square&logo=arduino)
![Platform](https://img.shields.io/badge/Platform-Windows%2011-0078D6?style=flat-square&logo=windows)

---

##  Overview

**HUMIX** is a real-time surveillance system designed to monitor crowd density and human movement in public spaces such as airports, hospitals, railway stations, and event venues. The system uses computer vision techniques to detect and count people from live camera feeds, estimate crowd density, and automatically generate alerts when predefined thresholds are exceeded — enabling authorities to respond proactively without requiring continuous manual supervision.

---

##  Features

- 🎥 **Real-Time Video Processing** — Captures and processes live video from webcam or connected camera
- 🧍 **Human Detection** — Detects individuals in video frames using the YOLOv12 algorithm
- 📊 **Crowd Density Estimation** — Counts detected people and estimates crowd density in real time
- 🚨 **Threshold-Based Alerts** — Automatically triggers alerts on dashboard and via Arduino-controlled LED indicators
- 📋 **Event Management** — Admin and event-level dashboards for organized monitoring and reporting
- 🖥️ **Live Dashboard** — Web-based monitoring interface showing crowd count, status, and alert notifications
- 🔒 **Offline & Secure** — Operates entirely locally; no data is transmitted to external servers

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| Primary Language | Python 3.13 |
| Computer Vision | OpenCV |
| Human Detection | YOLOv12 (`yolo12n.pt`) |
| Backend Framework | Flask |
| Frontend | HTML, CSS, JavaScript |
| Hardware Control | Arduino (LED alert system) |
| Database | SQLite (`humix.db`) |
| IDE | Visual Studio Code |
| Operating System | Windows 11 |

---

## 📁 Project Structure

```
HUMIXANTIGRAVITY/
├── humix/
│   ├── arduino/
│   │   └── led_control.ino         # Arduino sketch for LED alert control
│   ├── database/
│   │   └── humix.db                # SQLite database for crowd records
│   ├── static/
│   │   ├── captured_images/        # Snapshots captured during monitoring
│   │   ├── css/                    # Dashboard stylesheets
│   │   └── js/                     # Frontend JavaScript scripts
│   ├── templates/
│   │   ├── admin_dashboard.html    # Admin monitoring dashboard
│   │   ├── admin_events.html       # Admin event management page
│   │   ├── admin_login.html        # Admin login page
│   │   ├── event_dashboard.html    # Event-level monitoring dashboard
│   │   ├── event_login.html        # Event login page
│   │   ├── landing.html            # Landing/home page
│   │   └── register.html           # Registration page
│   ├── app.py                      # Main Flask application
│   ├── testard.py                  # Testing script
│   └── yolo12n.pt                  # YOLOv12 model weights
├── static/
│   └── captured_images/            # Top-level captured images storage
├── database/
│   └── humix.db                    # Top-level database
├── create_project.py               # Project setup script
├── requirements.txt                # Python dependencies
├── yolo12n.pt                      # YOLOv12 model weights (root copy)
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.13+
- A connected webcam or camera device
- Arduino board with LED (for hardware alerts)
- Windows 11
- Visual Studio Code (recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/humixantigravity.git
cd humixantigravity
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Upload Arduino Sketch
Open `humix/arduino/led_control.ino` in the Arduino IDE and upload it to your Arduino board.

### 4. Run the Application
```bash
python humix/app.py
```

### 5. Open the Dashboard
Navigate to the following address in your browser:
```
http://localhost:5000
```

---

## 👥 User Roles

### 🛡️ System Administrator
- Log in via the Admin Login page
- Configure camera devices and crowd density thresholds
- Manage events through the Admin Events dashboard
- Monitor crowd data and review system activity

### 👁️ Event Monitoring Authority
- Log in via the Event Login page
- View live video feed with real-time human detection
- Monitor crowd count and density status
- Receive instant alert notifications (dashboard + LED)

---

## 📷 Hardware Requirements

| Component | Description |
|---|---|
| Camera Device | Webcam or IP camera for live video input |
| Arduino Board | Controls LED indicators for physical alerts |
| LED Indicators | Visual alert output triggered via Arduino |
| Computing System | Standard PC capable of real-time video processing |
| Power Supply | Stable power source for uninterrupted operation |

> ⚠️ **Note:** Internet connectivity is **not required**. All processing is performed locally.

---

## ⚠️ System Limitations

- Detection accuracy depends on camera quality and lighting conditions
- Performance may decrease in extremely crowded or occluded environments
- Requires a stable power supply for uninterrupted real-time monitoring
- System performance varies based on the host machine's hardware capabilities

---

## 🔐 Privacy & Security

- HUMIX does **not** collect, store, or process any personal or identifiable information
- Video input is used solely for real-time crowd detection and is not retained after processing
- System configuration access is restricted to authorized administrators only
- All operations are performed entirely offline on the local machine

---

## 📜 License

All software components, algorithms, and documentation related to HUMIX are the intellectual property of the project team. Unauthorized copying, distribution, or modification is prohibited. Third-party libraries are used in accordance with their respective licenses.

---

## 👩‍💻 Team

| Name | Role |
|---|---|
| Adheena Basil | Developer |
| Devi Parvathy K | Developer |
| Sethulakshmi P R | Developer |

**Group 15 — Dept. of Computer Science & Engineering, VJCET**

**Guide:** Ms. Judy Ann Joy, Asst. Prof., VJCET

---

## 📚 References

1. *Crowd Scene Analysis using Deep Learning Techniques* — Muhammad Junaid Asif
2. *A Real-Time Crowd Detection and Monitoring System using Machine Learning* — Vakula Rani Jakka & Pooja Shrivastav
3. OpenCV Official Documentation — opencv.org
4. Ultralytics YOLOv12 Documentation — ultralytics.com

---

<p align="center">Made by Group 15 | VJCET</p>
