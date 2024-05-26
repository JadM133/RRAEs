from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
import jax.numpy as jnp
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from PyQt5.QtCore import QTimer
import os
import dill
import jax.random as jr
import equinox as eqx
from RRAEs.training_classes import Trainor_class

def load_eqx_nn(filename, creat_func):
    models = []
    hps = []
    with open(filename, "rb") as f:
        while True:
            try:
                hyperparams = dill.load(f)
                hyper = {k: hyperparams[k] for k in set(list(hyperparams.keys())) - set(["seed"])}
                model = creat_func(key=jr.PRNGKey(hyperparams["seed"]), **hyper)
                models.append(eqx.tree_deserialise_leaves(f, model))
                hps.append(hyperparams)
            except EOFError:
                break
    return models, hps

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=100, xlabel=None, ylabel=None):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        if xlabel is not None:
            self.axes.set_xlabel(xlabel)
        if ylabel is not None:
            self.axes.set_ylabel(ylabel)
        super(MplCanvas, self).__init__(fig)
        self.setParent(parent)

def find_mn_mx(arr):
    mn = jnp.min(arr)-(jnp.max(arr)-jnp.min(arr))/10
    mx = jnp.max(arr)+(jnp.max(arr)-jnp.min(arr))/10
    return mn, mx

class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, trainor_path):
        super().__init__()
        self.setObjectName("MainWindow")
        self.resize(1400, 1000)
        self.centralwidget = QtWidgets.QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.path = trainor_path
        trainor = Trainor_class()
        trainor.load(self.path)
        self.y_train = trainor.y_train
        self.idx1 = QtWidgets.QSpinBox(self.centralwidget)
        self.idx1.setGeometry(QtCore.QRect(1090, 105, 181, 51))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.idx1.setFont(font)
        self.idx1.setProperty("value", 0)
        self.idx1_val = 0
        self.idx1.setObjectName("spinBox")
        self.idx1.setMinimum(0)
        self.idx1.setMaximum(self.y_train.shape[-1]-1)
        self.idx1.valueChanged.connect(self.change_idx1)

        self.idx1_lab = QtWidgets.QLabel(self.centralwidget)
        self.idx1_lab.setGeometry(QtCore.QRect(1120, 57, 161, 45))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.idx1_lab.setFont(font)
        self.idx1_lab.setObjectName("label")
        
        self.idx2_lab = QtWidgets.QLabel(self.centralwidget)
        self.idx2_lab.setGeometry(QtCore.QRect(1100, 185, 181, 51))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.idx2_lab.setFont(font)
        self.idx2_lab.setObjectName("label_2")

        self.idx2 = QtWidgets.QSpinBox(self.centralwidget)
        self.idx2.setGeometry(QtCore.QRect(1090, 233, 181, 51))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.idx2.setFont(font)
        self.idx2.setProperty("value", 0)
        self.idx2_val = 0
        self.idx2.setObjectName("spinBox_2")
        self.idx2.setMinimum(0)
        self.idx2.setMaximum(self.y_train.shape[-1]-1)
        self.idx2.valueChanged.connect(self.change_idx2)

        self.ip_lab = QtWidgets.QLabel(self.centralwidget)
        self.ip_lab.setGeometry(QtCore.QRect(1100, 331, 181, 51))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.ip_lab.setFont(font)
        self.ip_lab.setObjectName("label_3")

        self.ip = QtWidgets.QSpinBox(self.centralwidget)
        self.ip.setGeometry(QtCore.QRect(1090, 381, 181, 51))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.ip.setFont(font)
        self.ip.setProperty("value", 20)
        self.ip_val = 50
        self.ip.setObjectName("spinBox_3")
        self.ip.setMinimum(50)
        self.ip.setMaximum(500)
        self.ip.valueChanged.connect(self.change_idx2)

        self.run_simul = QtWidgets.QPushButton(self.centralwidget)
        self.run_simul.setGeometry(QtCore.QRect(1090, 500, 181, 91))
        self.run_simul.setObjectName("run_simul")
        font = QtGui.QFont()
        font.setBold(True)
        font.setPointSize(10)
        self.run_simul.setCheckable(True)
        self.run_simul.setFont(font)
        self.run_simul.setStyleSheet("background-color : lightpink")
        self.run_simul.clicked.connect(self.run)
    
        self.run_simul_2 = QtWidgets.QPushButton(self.centralwidget)
        self.run_simul_2.setGeometry(QtCore.QRect(1090, 620, 181, 91))
        self.run_simul_2.setAutoFillBackground(True)
        self.run_simul_2.setObjectName("run_simul_2")
        font.setBold(True)
        font.setPointSize(10)
        self.run_simul_2.setFont(font)
        self.run_simul_2.setCheckable(True)
        self.run_simul_2.setStyleSheet("background-color : skyblue")
        self.run_simul_2.clicked.connect(self.interpolate)

        self.plot_above = MplCanvas(self, width=5, height=4, dpi=100, xlabel="Position")
        self.plot_above.axes.plot([], [])
        self.plot_above.setGeometry(QtCore.QRect(0, 0, 1000, 500))
        self.plot_above.mpl_connect("button_press_event", self.onclick)

        self.plot_below = MplCanvas(self, width=5, height=4, dpi=100, xlabel="Position", ylabel="Shadow")
        self.plot_below.axes.plot([], [])
        self.plot_below.setGeometry(QtCore.QRect(0, 500, 1000, 500))

        # self.widgetf = QtWidgets.QLabel(self.centralwidget)
        # self.widgetf.setText("Select variables...")
        # self.widgetf.setFont(QtGui.QFont('Arial', 10))
        # self.widgetf.setGeometry(QtCore.QRect(1070, 875, 221, 91))
        # self.widgetf.setStyleSheet("border: 1px solid black;")

        self.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 776, 22))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

        self.retranslateUi(self)
        self.computed = False
        QtCore.QMetaObject.connectSlotsByName(self)

    def onclick(self, event):
        self.sp_eval = event.xdata
        if self.sp_eval > jnp.max(self.spx):
            self.sp_eval = jnp.max(self.spx)
        if self.sp_eval < jnp.min(self.spx):
            self.sp_eval = jnp.min(self.spx)
        self.time_val.setValue(self.sp_eval)

    def change_idx1(self):
        self.idx1_val = self.idx1.value()
        self.computed = False

    def change_idx2(self):
        self.idx2_val = self.idx2.value()
        self.computed = False

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.idx1_lab.setText(_translate("MainWindow", "First Index"))
        self.idx2_lab.setText(_translate("MainWindow", "Second Index"))
        self.ip_lab.setText(_translate("MainWindow", "Interp. Points"))
        self.run_simul.setText(_translate("MainWindow", "Plot solutions"))
        self.run_simul_2.setText(_translate("MainWindow", "Interpolate"))

    def change_type(self):
        self.type_sim_val = self.type_sim.currentText()
    
    def update_graph(self):
        self.count += 1
        i = self.count
        self.plot_above.axes.cla()
        self.plot_above.axes.plot([], [])
        self.line.set_ydata(self.interp_res[:, i])
        self.plot_above.axes.plot(self.line.get_ydata(), linewidth=3)
        self.plot_above.axes.set_ylim([self.min_v, self.max_v])
        self.plot_above.axes.plot(self.interp_res[:, 0], color="blue", linewidth=3)
        self.plot_above.axes.plot(self.interp_res[:, -1], color="red", linewidth=3)
        self.plot_above.draw()
        return self.line
    
    def init_graph(self, curves):
        ax = self.plot_above.axes
        maxx = -1*jnp.inf
        minn = jnp.inf
        max_v = jnp.max(curves)
        min_v = jnp.min(curves)
        self.max_v = max_v
        self.min_v = min_v
        ax.set_ylim(self.min_v, self.max_v)
        self.plot_above.axes.plot(curves[:, 0], color="blue", linewidth=3)
        self.plot_above.axes.plot(curves[:, -1], color="red", linewidth=3)

    def run(self):
        self.plot_below.axes.cla()
        self.plot_below.draw()
        self.run_simul.repaint()
        trainor = Trainor_class()
        trainor.load(self.path)
        self.plot_below.axes.plot(trainor.y_train[:, self.idx1_val], color="blue", linewidth=3)
        self.plot_below.axes.plot(trainor.y_train[:, self.idx2_val], color="red", linewidth=3)
        self.plot_below.draw()
        self.run_simul.setChecked(False)


    def _interpolate(self):
        self.plot_above.draw()
        self.run_simul.repaint()
        if not self.computed:
            trainor = Trainor_class()
            trainor.load(self.path)
            lat_1 = trainor.model.latent(trainor.y_train[:, self.idx1_val:self.idx1_val+1])
            lat_2 = trainor.model.latent(trainor.y_train[:, self.idx2_val:self.idx2_val+1])
            points = self.ip_val
            prop_left = jnp.linspace(0, 1, points+2)[1:-1]
            latents = (jnp.squeeze(lat_1) + (prop_left[:, None] * jnp.squeeze(lat_2 - lat_1))).T
            self.interp_res = trainor.model.decode(latents)
            self.computed = True

        self.init_graph(self.interp_res)
        self.plot_above.draw()
        self.line = Line2D(self.interp_res[:, 0:1], self.interp_res[:, 0:1], color="black", linewidth=3)
        self.plot_above.axes.plot([], [])
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.count = 0
        self.timer.timeout.connect(self.update_graph)
        self.timer.start()
        self.run_simul_2.show()
        self.run_simul.setChecked(False)
        self.run_simul.setStyleSheet("background-color : red")

    def interpolate(self):
        self._interpolate()

def run_animation(trainor_path):
    import sys
    import os
    path = os.path.join(os.getcwd(), trainor_path)
    app = QtWidgets.QApplication(sys.argv)
    window = Ui_MainWindow(path)
    window.showNormal()
    app.exec()

if __name__ == "__main__":
    method = "Strong"
    problem = "shift"
    folder=f"{problem}/{method}_{problem}/"
    file=f"{method}_{problem}"
    run_animation(os.path.join(folder, file))
