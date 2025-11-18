"""
Simple test script to verify camera access
"""
import cv2
import time

print("Testing camera access...")

# Test camera 0
print("\nTesting Camera 0 with AVFoundation backend...")
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if cap.isOpened():
    print("✓ Camera 0 opened with AVFoundation")
    ret, frame = cap.read()
    if ret:
        print(f"✓ Successfully read frame: {frame.shape}")
    else:
        print("✗ Failed to read frame")
    cap.release()
else:
    print("✗ Failed to open camera 0 with AVFoundation")

print("\nTesting Camera 0 with default backend...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✓ Camera 0 opened with default backend")
    # Wait a bit for camera to initialize
    time.sleep(0.5)
    ret, frame = cap.read()
    if ret:
        print(f"✓ Successfully read frame: {frame.shape}")
    else:
        print("✗ Failed to read frame")
    cap.release()
else:
    print("✗ Failed to open camera 0 with default backend")

print("\nDone!")

