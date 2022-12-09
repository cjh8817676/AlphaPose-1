import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

win = pg.GraphicsLayoutWidget(show=True)
win.setWindowTitle('pyqtgraph example: Scrolling Plots')


p2 = win.addPlot()
data1 = np.random.normal(size=300)

curve2 = p2.plot(data1,pen=pg.mkPen('r', width=3))
ptr1 = 0


def update1():
    global data1, ptr1
    data1[:-1] = data1[1:]  # shift data in the array one sample left

    data1[-1] = np.random.normal()

    ptr1 += 1
    curve2.setData(data1)
    curve2.setPos(ptr1, 0)


timer = pg.QtCore.QTimer()
timer.timeout.connect(update1)
timer.start(50)


if __name__ == '__main__':
    import sys

    #if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    QtGui.QApplication.instance().exec_()