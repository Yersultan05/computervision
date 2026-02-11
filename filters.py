import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

methods = [
    "Canny", "Sobel", "Laplacian", "Otsu",
    "Prewitt", "Roberts", "Scharr",
    "HSV", "Anisotropic Diffusion",
    "Erosion", "Dilation", "Optical Flow"
]

method = 0
prev_gray = None

button_height = 40
panel_width = 260

def anisotropic_diffusion(img, num_iter=10, kappa=30, gamma=0.1):
    img = img.astype(np.float32)
    for _ in range(num_iter):
        north = np.roll(img, -1, axis=0) - img
        south = np.roll(img, 1, axis=0) - img
        east = np.roll(img, -1, axis=1) - img
        west = np.roll(img, 1, axis=1) - img

        cN = np.exp(-(north/kappa)**2)
        cS = np.exp(-(south/kappa)**2)
        cE = np.exp(-(east/kappa)**2)
        cW = np.exp(-(west/kappa)**2)

        img = img + gamma * (cN*north + cS*south + cE*east + cW*west)

    return np.clip(img, 0, 255).astype(np.uint8)

def mouse_event(event, x, y, flags, param):
    global method
    if event == cv2.EVENT_LBUTTONDOWN:
        if x > frame_width*2:
            idx = y // button_height
            if idx < len(methods):
                method = idx

cv2.namedWindow("Interface")

ret, frame = cap.read()
frame_height, frame_width = frame.shape[:2]

cv2.setMouseCallback("Interface", mouse_event)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    processed = frame.copy()

    if method == 0:  # Canny
        edges = cv2.Canny(gray, 100, 200)
        processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    elif method == 1:  # Sobel
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        processed = cv2.cvtColor(cv2.convertScaleAbs(sx + sy), cv2.COLOR_GRAY2BGR)

    elif method == 2:  # Laplacian
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        processed = cv2.cvtColor(cv2.convertScaleAbs(lap), cv2.COLOR_GRAY2BGR)

    elif method == 3:  # Otsu
        _, th = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        processed = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)

    elif method == 4:  # Prewitt
        kx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
        ky = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
        px = cv2.filter2D(gray, -1, kx)
        py = cv2.filter2D(gray, -1, ky)
        processed = cv2.cvtColor(cv2.convertScaleAbs(px + py), cv2.COLOR_GRAY2BGR)

    elif method == 5:  # Roberts
        kx = np.array([[1,0],[0,-1]])
        ky = np.array([[0,1],[-1,0]])
        rx = cv2.filter2D(gray, -1, kx)
        ry = cv2.filter2D(gray, -1, ky)
        processed = cv2.cvtColor(cv2.convertScaleAbs(rx + ry), cv2.COLOR_GRAY2BGR)

    elif method == 6:  # Scharr
        sx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        sy = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        processed = cv2.cvtColor(cv2.convertScaleAbs(sx + sy), cv2.COLOR_GRAY2BGR)


    elif method == 7:  # HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        processed = hsv


    elif method == 8:  # Anisotropic Diffusion
        diff = anisotropic_diffusion(gray)
        processed = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)

    elif method == 9:  # Erosion
        kernel = np.ones((5,5), np.uint8)
        er = cv2.erode(gray, kernel, iterations=1)
        processed = cv2.cvtColor(er, cv2.COLOR_GRAY2BGR)

    elif method == 10:  # Dilation
        kernel = np.ones((5,5), np.uint8)
        dil = cv2.dilate(gray, kernel, iterations=1)
        processed = cv2.cvtColor(dil, cv2.COLOR_GRAY2BGR)


    elif method == 11:  # Optical Flow
        if prev_gray is None:
            prev_gray = gray
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                             None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv = np.zeros_like(frame)
        hsv[...,1] = 255
        hsv[...,0] = ang * 180 / np.pi / 2
        hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        processed = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        prev_gray = gray

    # -------- Панель кнопок --------
    panel = np.zeros((frame_height, panel_width, 3), dtype=np.uint8)

    for i, name in enumerate(methods):
        y1 = i * button_height
        y2 = y1 + button_height
        color = (60,60,60)
        if i == method:
            color = (0,100,255)
        cv2.rectangle(panel, (0,y1), (panel_width,y2), color, -1)
        cv2.putText(panel, name, (10,y1+28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    combined = np.hstack((frame, processed, panel))
    cv2.imshow("Interface", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
