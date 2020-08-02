import cv2, io
from matplotlib import pyplot as plt
import numpy as np

def displayFrames(f):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', np.mean(f, axis=2))
    cv2.waitKey(1)

def displayMetric(arr, name):
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(range(len(arr)), arr)
    fig.canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    plt.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(1)
