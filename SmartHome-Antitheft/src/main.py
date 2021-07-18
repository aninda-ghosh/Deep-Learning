import time
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

print(cv.__version__)

capture = cv.VideoCapture('http://192.168.0.117/mjpeg')

color = 'gray'
bins = 16
resizeWidth = 800

# Initialize plot.
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

if color == 'rgb':
    ax1.set_title('Histogram (RGB)')
elif color == 'lab':
    ax1.set_title('Histogram (L*a*b*)')
else:
    ax1.set_title('Histogram (grayscale)')
ax1.set_xlabel('Bin')
ax1.set_ylabel('Frequency')

# Initialize plot line object(s). Turn on interactive plotting and show plot.
lw = 3
alpha = 0.5

if color == 'rgb':
    lineR, = ax1.plot(np.arange(bins), np.zeros((bins,)), c='r', lw=lw, alpha=alpha, label='Red')
    lineG, = ax1.plot(np.arange(bins), np.zeros((bins,)), c='g', lw=lw, alpha=alpha, label='Green')
    lineB, = ax1.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha, label='Blue')
elif color == 'lab':
    lineL, = ax1.plot(np.arange(bins), np.zeros((bins,)), c='k', lw=lw, alpha=alpha, label='L*')
    lineA, = ax1.plot(np.arange(bins), np.zeros((bins,)), c='b', lw=lw, alpha=alpha, label='a*')
    lineB, = ax1.plot(np.arange(bins), np.zeros((bins,)), c='y', lw=lw, alpha=alpha, label='b*')
else:
    lineGray, = ax1.plot(np.arange(bins), np.zeros((bins,1)), c='k', lw=lw, label='intensity')
    equalizedlineGray, = ax2.plot(np.arange(bins), np.zeros((bins, 1)), c='k', lw=lw, label='equalized intensity')

ax1.set_xlim(0, bins-1)
ax1.set_ylim(0, 1)
ax1.legend()
ax2.set_xlim(0, bins-1)
ax2.set_ylim(0, 1)
ax2.legend()
fig.canvas.draw()
plt.show(block=False)

# Grab, process, and display video frames. Update plot line object(s).
while True:
    (grabbed, frame) = capture.read()

    if not grabbed:
        break

    # Resize frame to width, if specified.
    if resizeWidth > 0:
        (height, width) = frame.shape[:2]
        resizeHeight = int(float(resizeWidth / width) * height)
        frame = cv.resize(frame, (resizeWidth, resizeHeight), interpolation=cv.INTER_AREA)

    # Normalize histograms based on number of pixels per frame.
    numPixels = np.prod(frame.shape[:2])

    cv.imshow('RGB', frame)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if color == 'rgb':
        cv.imshow('RGB', frame)
        (b, g, r) = cv.split(frame)
        histogramR = cv.calcHist([r], [0], None, [bins], [0, 255]) / numPixels
        histogramG = cv.calcHist([g], [0], None, [bins], [0, 255]) / numPixels
        histogramB = cv.calcHist([b], [0], None, [bins], [0, 255]) / numPixels
        lineR.set_ydata(histogramR)
        lineG.set_ydata(histogramG)
        lineB.set_ydata(histogramB)
    elif color == 'lab':
        cv.imshow('L*a*b*', frame)
        lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
        (l, a, b) = cv.split(lab)
        histogramL = cv.calcHist([l], [0], None, [bins], [0, 255]) / numPixels
        histogramA = cv.calcHist([a], [0], None, [bins], [0, 255]) / numPixels
        histogramB = cv.calcHist([b], [0], None, [bins], [0, 255]) / numPixels
        lineL.set_ydata(histogramL)
        lineA.set_ydata(histogramA)
        lineB.set_ydata(histogramB)
    else:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        cv.imshow('Grayscale', gray)
        cv.imshow('Equalized Grayscale', equalized)
        histogram = cv.calcHist([gray], [0], None, [bins], [0, 255]) / numPixels
        equalizedhistogram = cv.calcHist([equalized], [0], None, [bins], [0, 255]) / numPixels
        lineGray.set_ydata(histogram)
        equalizedlineGray.set_ydata(equalizedhistogram)

    print("Trying to plot the RGB")

    fig.canvas.draw()
    plt.pause(0.0001)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()