import cv2
import numpy as np


# Start video capture from webcam
video_cap = cv2.VideoCapture(0)

frame_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_cap.get(cv2.CAP_PROP_FPS))

# Read the first frame as the base for comparison
ret, frame1 = video_cap.read()
frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
frame1_gray = cv2.GaussianBlur(frame1_gray, (21, 21), 0)


def draw_banner_text(frame, text, banner_height_percent=0.08, font_scale=0.8, text_color=(0, 255, 0),
                   font_thickness=2):
    banner_height = int(banner_height_percent * frame.shape[0])
    cv2.rectangle(frame, (0, 0), (frame.shape[1], banner_height), (0, 0, 0), thickness=-1)
    left_offset = 20
    location = (left_offset, int(10 + (banner_height_percent * frame.shape[0]) / 2))
    cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color,
                font_thickness, cv2.LINE_AA)


bg_sub = cv2.createBackgroundSubtractorKNN(history=200)
ksize = (5, 5)
red = (0, 0, 255)
yellow = (0, 255, 255)

while True:
    ret, frame = video_cap.read()
    if not ret or frame is None:
        break

    frame_erode = frame.copy()

    # Stage 1: Motion area based on foreground mask.
    fg_mask = bg_sub.apply(frame)
    motion_area = cv2.findNonZero(fg_mask)
    if motion_area is not None:
        x, y, w, h = cv2.boundingRect(motion_area)
        cv2.rectangle(frame, (x, y), (x + w, y + h), red, thickness=6)

    # Stage 2: Eroded mask
    fg_mask_erode = cv2.erode(fg_mask, np.ones(ksize, np.uint8))
    motion_area_erode = cv2.findNonZero(fg_mask_erode)
    if motion_area_erode is not None:
        xe, ye, we, he = cv2.boundingRect(motion_area_erode)
        cv2.rectangle(frame_erode, (xe, ye), (xe + we, ye + he), red, thickness=6)

    # Convert to color for annotation
    frame_fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
    frame_fg_mask_erode = cv2.cvtColor(fg_mask_erode, cv2.COLOR_GRAY2BGR)

    # Annotate frames
    draw_banner_text(frame_fg_mask, 'Foreground Mask')
    draw_banner_text(frame_fg_mask_erode, 'Foreground Mask Eroded')

    # Build quad view
    frame_top = np.hstack([frame_fg_mask, frame])
    frame_bot = np.hstack([frame_fg_mask_erode, frame_erode])
    frame_composite = np.vstack([frame_top, frame_bot])

    # Draw cross line
    fc_h, fc_w, _ = frame_composite.shape
    cv2.line(frame_composite, (0, fc_h // 2), (fc_w, fc_h // 2), yellow, thickness=1, lineType=cv2.LINE_AA)

    # Resize for display
    frame_composite = cv2.resize(frame_composite, None, fx=0.5, fy=0.5)

    # Show output
    cv2.imshow("Motion Detection - Quad View", frame_composite)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()
