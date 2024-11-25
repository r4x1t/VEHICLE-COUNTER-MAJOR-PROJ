import cv2
import cv2.bgsegm
import numpy as np
import streamlit as st

# Streamlit configuration
st.title("Vehicle Detection and Counting")
st.subheader("Upload a video to detect and count vehicles")

# Video file upload option in Streamlit
video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

# Parameters for bounding boxes and counting
min_width_rect = 80
min_height_rect = 80
min_area = 300
count_line_position = 550

# Background subtractor
algo = cv2.createBackgroundSubtractorMOG2()


# Function to calculate the center of bounding boxes
def center_handle(x, y, w, h):
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy

# Processing the video
if video_file is not None:
    # Load video
    file_bytes = np.asarray(bytearray(video_file.read()), dtype=np.uint8)
    cap = cv2.VideoCapture(cv2.imdecode(file_bytes, cv2.IMREAD_COLOR))
    detect = []
    offset = 6
    counter = 0
    
    # Streamlit video display loop
    stframe = st.empty()
    while cap.isOpened():
        ret, frame1 = cap.read()
        if not ret:
            break

        grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(grey, (3, 3), 5)

        # Background subtraction and morphology
        img_sub = algo.apply(blur)
        dilat = cv2.dilate(img_sub, np.ones((5, 5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
        dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

        counterShape, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

        for (i, c) in enumerate(counterShape):
            (x, y, w, h) = cv2.boundingRect(c)
            if w * h < min_area or w < min_width_rect or h < min_height_rect:
                continue

            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center = center_handle(x, y, w, h)
            detect.append(center)
            cv2.circle(frame1, center, 4, (0, 0, 255), -1)

        for (x, y) in detect:
            if y < (count_line_position + offset) and y > (count_line_position - offset):
                counter += 1
                cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
                detect.remove((x, y))

        # Display the counter on the frame
        cv2.putText(frame1, "VEHICLE COUNTER: " + str(counter), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        stframe.image(frame1, channels="BGR")

    cap.release()
else:
    st.write("Please upload a video file to proceed.")
