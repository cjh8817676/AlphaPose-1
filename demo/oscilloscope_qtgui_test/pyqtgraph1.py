import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

win = pg.GraphicsLayoutWidget(show=True)
win.setWindowTitle('Scrolling Plots Mode 1')

p1 = win.addPlot()
data1 = np.random.normal(size=300)

curve1 = p1.plot(data1)


def update1():
    global data1, ptr1
    data1[:-1] = data1[1:]  # shift data in the array one sample left
    # (see also: np.roll)
    data1[-1] = np.random.normal()
    curve1.setData(data1)


timer = pg.QtCore.QTimer()
timer.timeout.connect(update1)
timer.start(50)


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()