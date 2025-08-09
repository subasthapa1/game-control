import cv2
import numpy as np

# Open webcam
video_cap = cv2.VideoCapture(0)
if not video_cap.isOpened():
    print('Unable to open the webcam')
    exit()

# Get frame properties
frame_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video_cap.get(cv2.CAP_PROP_FPS)) or 30  # Fallback FPS if 0

# Quad view size = 2x width, 2x height
size_quad = (frame_w * 2, frame_h * 2)

# Video writer
video_out_quad = cv2.VideoWriter('video_out_quad.mp4',
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps,
                                 size_quad)

# Draw banner text
def drawBannerText(frame, text, banner_height_percent=0.08,
                   font_scale=0.8, text_color=(0, 255, 0), font_thickness=2):
    banner_height = int(banner_height_percent * frame.shape[0])
    cv2.rectangle(frame, (0, 0), (frame.shape[1], banner_height), (0, 0, 0), thickness=-1)

    left_offset = 20
    baseline = int((banner_height / 2) + (font_scale * 10))
    location = (left_offset, baseline)
    cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, text_color, font_thickness, cv2.LINE_AA)

# Background subtractor
bg_sub = cv2.createBackgroundSubtractorKNN(history=200)

# Parameters
ksize = (5, 5)
max_contours = 3
frame_count = 0
frame_start = 5
red = (0, 0, 255)
yellow = (0, 255, 255)
green = (0, 255, 0)

while True:
    ret, frame = video_cap.read()
    if not ret:
        break

    frame_count += 1
    frame_erode_c = frame.copy()

    # Foreground mask
    fg_mask = bg_sub.apply(frame)

    if frame_count > frame_start:
        # Stage 1: Motion detection bounding box
        motion_area = cv2.findNonZero(fg_mask)
        if motion_area is not None:
            x, y, w, h = cv2.boundingRect(motion_area)
            cv2.rectangle(frame, (x, y), (x + w, y + h), red, 2)
            drawBannerText(frame, 'Intrusion Alert', text_color=red)

        # Stage 2: Erosion
        fg_mask_erode_c = cv2.erode(fg_mask, np.ones(ksize, np.uint8))
        motion_area_erode = cv2.findNonZero(fg_mask_erode_c)
        if motion_area_erode is not None:
            xe, ye, we, he = cv2.boundingRect(motion_area_erode)
            cv2.rectangle(frame_erode_c, (xe, ye), (xe + we, ye + he), red, 2)
            drawBannerText(frame_erode_c, 'Intrusion Alert', text_color=red)

        # Convert masks to color
        frame_fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        frame_fg_mask_erode_c = cv2.cvtColor(fg_mask_erode_c, cv2.COLOR_GRAY2BGR)

        # Stage 3: Contour detection
        contours_erode, _ = cv2.findContours(fg_mask_erode_c, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours_erode) > 0:
            cv2.drawContours(frame_fg_mask_erode_c, contours_erode, -1, green, 2)
            contours_sorted = sorted(contours_erode, key=cv2.contourArea, reverse=True)

            for idx in range(min(max_contours, len(contours_sorted))):
                xc, yc, wc, hc = cv2.boundingRect(contours_sorted[idx])
                if idx == 0:
                    x1, y1, x2, y2 = xc, yc, xc + wc, yc + hc
                else:
                    x1 = min(x1, xc)
                    y1 = min(y1, yc)
                    x2 = max(x2, xc + wc)
                    y2 = max(y2, yc + hc)

            cv2.rectangle(frame_erode_c, (x1, y1), (x2, y2), yellow, 2)
            drawBannerText(frame_erode_c, 'Intrusion Alert', text_color=red)

        # Labels for masks
        drawBannerText(frame_fg_mask, 'Foreground Mask')
        drawBannerText(frame_fg_mask_erode_c, 'Foreground Mask (Eroded + Contours)')

        # Build quad view
        frame_top = np.hstack([frame_fg_mask, frame])
        frame_bot = np.hstack([frame_fg_mask_erode_c, frame_erode_c])
        frame_composite = np.vstack([frame_top, frame_bot])

        # Divider lines
        fc_h, fc_w, _ = frame_composite.shape
        cv2.line(frame_composite, (fc_w // 2, 0), (fc_w // 2, fc_h), yellow, 3, cv2.LINE_AA)
        cv2.line(frame_composite, (0, fc_h // 2), (fc_w, fc_h // 2), yellow, 3, cv2.LINE_AA)

        # Show and save
        cv2.imshow('Quad View', frame_composite)
        video_out_quad.write(frame_composite)

    # Exit on Q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video_cap.release()
video_out_quad.release()
cv2.destroyAllWindows()
