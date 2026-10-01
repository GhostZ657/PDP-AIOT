import cv2
import time
import numpy as np
from ultralytics import YOLO
from gpiozero import LED  # Ganti dengan gpiozero

# === KONFIGURASI GPIO UNTUK LAMPU ===
LED_PIN = 17  # Ganti sesuai pin GPIO yang kamu gunakan
led = LED(LED_PIN)  # Inisialisasi LED
#led.off()  # Matikan lampu di awal

print("Loading YOLOv8 model...")
model = YOLO('human_model.pt')

print("Opening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    for i in [1, 2, 10]:
        print(f"Mencoba indeks kamera {i}...")
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Kamera ditemukan di indeks {i}")
            break
        cap.release()

if not cap.isOpened():
    print("ERROR: Tidak dapat membuka kamera!")
    import subprocess
    subprocess.run(["v4l2-ctl", "--list-devices"])
    exit(1)

# OPTIMASI untuk Raspberry Pi
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Starting human detection...")
print("Q=Quit  S=Save  R=Res  +/-=Conf  L=Toggle Light")

prev_time = time.time()
fps_buffer = []
confidence = 0.4
inference_size = 160
light_enabled = True  # Fitur lampu aktif secara default
human_detected = False  # Status deteksi manusia

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal mengambil frame")
        break
    
    # Resize frame untuk inferensi lebih cepat
    results = model(frame, 
                    conf=confidence,
                    imgsz=inference_size,
                    iou=0.3,
                    max_det=3,
                    classes=[0],
                    verbose=False)
    
    # Hitung FPS
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time
    fps_buffer.append(fps)
    if len(fps_buffer) > 10:
        fps_buffer.pop(0)
    avg_fps = np.mean(fps_buffer)
    
    # Gambar hasil deteksi
    human_count = 0
    if results[0].boxes is not None:
        for box in results[0].boxes:
            human_count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Bounding box tipis warna hijau
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            # Label hanya jika confidence tinggi
            if conf > 0.7:
                cv2.putText(frame, f"{conf:.0%}", (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    
    # === KONTROL LAMPU ===
    # Cek apakah ada manusia terdeteksi
    if human_count > 0:
        human_detected = False
        if light_enabled:
            led.off()  # Nyalakan lampu (gpiozero)
            cv2.putText(frame, "LIGHT: ON", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    else:
        human_detected = True
        if light_enabled:
            led.on()  # Matikan lampu (gpiozero)
            cv2.putText(frame, "LIGHT: OFF", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    
    # Jika fitur lampu dimatikan (toggle)
    if not light_enabled:
        led.on()  # Pastikan lampu mati
        cv2.putText(frame, "LIGHT: DISABLED", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    # === OVERLAY MINIMALIS ===
    # Indikator recording kecil di pojok kiri atas
    cv2.circle(frame, (15, 15), 5, (0, 0, 255), -1)
    cv2.putText(frame, "REC", (25, 19),
               cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
    
    # === TULISAN FPS DAN HUMAN SAJA DI POJOK KANAN ATAS ===
    text_bg_height = 40
    text_bg_width = 80
    cv2.rectangle(frame, 
                  (frame.shape[1] - text_bg_width - 5, 5), 
                  (frame.shape[1] - 5, 5 + text_bg_height), 
                  (0, 0, 0), -1)
    cv2.addWeighted(frame, 0.7, frame, 0.3, 0, frame)
    
    # Tampilkan FPS dan Human saja
    cv2.putText(frame, f"FPS:{avg_fps:.0f}", 
                (frame.shape[1] - 75, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(frame, f"HUM:{human_count}", 
                (frame.shape[1] - 75, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Tampilkan frame
    cv2.imshow('Human Detection', frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('s'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
    elif key == ord('r'):
        inference_size = 160 if inference_size == 320 else 320
        print(f"Resolusi: {inference_size}")
    elif key == ord('+'):
        confidence = min(0.9, confidence + 0.05)
        print(f"Conf: {confidence:.2f}")
    elif key == ord('-'):
        confidence = max(0.1, confidence - 0.05)
        print(f"Conf: {confidence:.2f}")
    elif key == ord('l'):
        light_enabled = not light_enabled
        status = "ON" if light_enabled else "OFF"
        print(f"Light feature: {status}")
        if not light_enabled:
            led.off()

# Cleanup
led.off()
cap.release()
cv2.destroyAllWindows()
print(f"\nStopped. Avg FPS: {avg_fps:.1f}")
