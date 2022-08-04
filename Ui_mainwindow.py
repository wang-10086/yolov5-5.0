# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Ui_mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1080, 720)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 1080, 720))
        self.tabWidget.setFocusPolicy(QtCore.Qt.TabFocus)
        self.tabWidget.setAutoFillBackground(False)
        self.tabWidget.setStyleSheet("background-color: rgb(127, 139, 147);")
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.widget = QtWidgets.QWidget(self.tab)
        self.widget.setGeometry(QtCore.QRect(839, 30, 221, 541))
        self.widget.setStyleSheet("background-color: rgb(152, 184, 172);")
        self.widget.setObjectName("widget")
        self.label = QtWidgets.QLabel(self.widget)
        self.label.setGeometry(QtCore.QRect(20, 40, 80, 40))
        font = QtGui.QFont()
        font.setFamily("AcadEref")
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setGeometry(QtCore.QRect(120, 40, 70, 40))
        font = QtGui.QFont()
        font.setFamily("AcadEref")
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setGeometry(QtCore.QRect(20, 90, 80, 40))
        font = QtGui.QFont()
        font.setFamily("AcadEref")
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setGeometry(QtCore.QRect(20, 300, 80, 40))
        font = QtGui.QFont()
        font.setFamily("AcadEref")
        font.setPointSize(10)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.pushButton = QtWidgets.QPushButton(self.widget)
        self.pushButton.setGeometry(QtCore.QRect(60, 490, 93, 28))
        self.pushButton.setObjectName("pushButton")
        self.label_6 = QtWidgets.QLabel(self.widget)
        self.label_6.setGeometry(QtCore.QRect(30, 150, 201, 131))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(9)
        self.label_6.setFont(font)
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.widget)
        self.label_7.setGeometry(QtCore.QRect(110, 305, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(9)
        self.label_7.setFont(font)
        self.label_7.setText("")
        self.label_7.setObjectName("label_7")
        self.layoutWidget = QtWidgets.QWidget(self.widget)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 370, 167, 53))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_9 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(9)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 0, 0, 1, 1)
        self.doubleSpinBox = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.doubleSpinBox.setMaximum(1.0)
        self.doubleSpinBox.setSingleStep(0.01)
        self.doubleSpinBox.setProperty("value", 0.5)
        self.doubleSpinBox.setObjectName("doubleSpinBox")
        self.gridLayout.addWidget(self.doubleSpinBox, 0, 1, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(9)
        self.label_10.setFont(font)
        self.label_10.setObjectName("label_10")
        self.gridLayout.addWidget(self.label_10, 1, 0, 1, 1)
        self.doubleSpinBox_2 = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.doubleSpinBox_2.setMaximum(1.0)
        self.doubleSpinBox_2.setSingleStep(0.01)
        self.doubleSpinBox_2.setProperty("value", 0.45)
        self.doubleSpinBox_2.setObjectName("doubleSpinBox_2")
        self.gridLayout.addWidget(self.doubleSpinBox_2, 1, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.tab)
        self.label_5.setGeometry(QtCore.QRect(10, 30, 810, 540))
        self.label_5.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_5.setText("")
        self.label_5.setObjectName("label_5")
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.pushButton_2 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_2.setGeometry(QtCore.QRect(790, 430, 93, 28))
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_8 = QtWidgets.QLabel(self.tab_2)
        self.label_8.setGeometry(QtCore.QRect(20, 40, 720, 480))
        self.label_8.setStyleSheet("background-color: rgb(249, 255, 243);")
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")
        self.layoutWidget1 = QtWidgets.QWidget(self.tab_2)
        self.layoutWidget1.setGeometry(QtCore.QRect(780, 260, 221, 111))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget1)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_11 = QtWidgets.QLabel(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(9)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 0, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 0, 1, 1, 1)
        self.doubleSpinBox_3 = QtWidgets.QDoubleSpinBox(self.layoutWidget1)
        self.doubleSpinBox_3.setMaximum(1.0)
        self.doubleSpinBox_3.setSingleStep(0.01)
        self.doubleSpinBox_3.setProperty("value", 0.5)
        self.doubleSpinBox_3.setObjectName("doubleSpinBox_3")
        self.gridLayout_2.addWidget(self.doubleSpinBox_3, 0, 2, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(9)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.gridLayout_2.addWidget(self.label_12, 1, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 1, 1, 1, 1)
        self.doubleSpinBox_4 = QtWidgets.QDoubleSpinBox(self.layoutWidget1)
        self.doubleSpinBox_4.setMaximum(1.0)
        self.doubleSpinBox_4.setSingleStep(0.01)
        self.doubleSpinBox_4.setProperty("value", 0.45)
        self.doubleSpinBox_4.setObjectName("doubleSpinBox_4")
        self.gridLayout_2.addWidget(self.doubleSpinBox_4, 1, 2, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(9)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.gridLayout_2.addWidget(self.label_13, 2, 0, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem2, 2, 1, 1, 1)
        self.spinBox = QtWidgets.QSpinBox(self.layoutWidget1)
        self.spinBox.setMinimum(1)
        self.spinBox.setMaximum(60)
        self.spinBox.setProperty("value", 30)
        self.spinBox.setObjectName("spinBox")
        self.gridLayout_2.addWidget(self.spinBox, 2, 2, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.layoutWidget1)
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei")
        font.setPointSize(9)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.gridLayout_2.addWidget(self.label_14, 3, 0, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem3, 3, 1, 1, 1)
        self.spinBox_2 = QtWidgets.QSpinBox(self.layoutWidget1)
        self.spinBox_2.setMinimum(1)
        self.spinBox_2.setMaximum(50)
        self.spinBox_2.setProperty("value", 5)
        self.spinBox_2.setObjectName("spinBox_2")
        self.gridLayout_2.addWidget(self.spinBox_2, 3, 2, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_3.setGeometry(QtCore.QRect(920, 430, 93, 28))
        self.pushButton_3.setObjectName("pushButton_3")
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.tabWidget.addTab(self.tab_3, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1080, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        self.menu_4 = QtWidgets.QMenu(self.menubar)
        self.menu_4.setObjectName("menu_4")
        self.menu_5 = QtWidgets.QMenu(self.menubar)
        self.menu_5.setObjectName("menu_5")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionImport = QtWidgets.QAction(MainWindow)
        self.actionImport.setObjectName("actionImport")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionSettings = QtWidgets.QAction(MainWindow)
        self.actionSettings.setObjectName("actionSettings")
        self.actionImage = QtWidgets.QAction(MainWindow)
        self.actionImage.setCheckable(True)
        self.actionImage.setObjectName("actionImage")
        self.actionVideo = QtWidgets.QAction(MainWindow)
        self.actionVideo.setCheckable(True)
        self.actionVideo.setChecked(False)
        self.actionVideo.setObjectName("actionVideo")
        self.actionReal_time = QtWidgets.QAction(MainWindow)
        self.actionReal_time.setCheckable(True)
        self.actionReal_time.setObjectName("actionReal_time")
        self.actionfps_tets = QtWidgets.QAction(MainWindow)
        self.actionfps_tets.setCheckable(True)
        self.actionfps_tets.setObjectName("actionfps_tets")
        self.actionWeghts = QtWidgets.QAction(MainWindow)
        self.actionWeghts.setObjectName("actionWeghts")
        self.actiondevice = QtWidgets.QAction(MainWindow)
        self.actiondevice.setObjectName("actiondevice")
        self.actionconf_thres = QtWidgets.QAction(MainWindow)
        self.actionconf_thres.setObjectName("actionconf_thres")
        self.actiondevice_2 = QtWidgets.QAction(MainWindow)
        self.actiondevice_2.setObjectName("actiondevice_2")
        self.actionsignal = QtWidgets.QAction(MainWindow)
        self.actionsignal.setObjectName("actionsignal")
        self.actionspeed_curve = QtWidgets.QAction(MainWindow)
        self.actionspeed_curve.setObjectName("actionspeed_curve")
        self.actionhandbook = QtWidgets.QAction(MainWindow)
        self.actionhandbook.setObjectName("actionhandbook")
        self.actioncontact_us = QtWidgets.QAction(MainWindow)
        self.actioncontact_us.setObjectName("actioncontact_us")
        self.actionIs_ROI = QtWidgets.QAction(MainWindow)
        self.actionIs_ROI.setCheckable(True)
        self.actionIs_ROI.setObjectName("actionIs_ROI")
        self.actionROI_Settings = QtWidgets.QAction(MainWindow)
        self.actionROI_Settings.setObjectName("actionROI_Settings")
        self.menu.addAction(self.actionImport)
        self.menu.addAction(self.actionSave)
        self.menu.addAction(self.actionSettings)
        self.menu_2.addAction(self.actionIs_ROI)
        self.menu_2.addAction(self.actionROI_Settings)
        self.menu_3.addAction(self.actionWeghts)
        self.menu_3.addAction(self.actiondevice)
        self.menu_3.addAction(self.actionconf_thres)
        self.menu_3.addAction(self.actiondevice_2)
        self.menu_4.addAction(self.actionsignal)
        self.menu_4.addAction(self.actionspeed_curve)
        self.menu_5.addAction(self.actionhandbook)
        self.menu_5.addAction(self.actioncontact_us)
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())
        self.menubar.addAction(self.menu_4.menuAction())
        self.menubar.addAction(self.menu_5.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "检测对象："))
        self.label_2.setText(_translate("MainWindow", "File"))
        self.label_3.setText(_translate("MainWindow", "检测结果："))
        self.label_4.setText(_translate("MainWindow", "检测时长："))
        self.pushButton.setText(_translate("MainWindow", "开始检测"))
        self.label_9.setText(_translate("MainWindow", "置信度阈值："))
        self.label_10.setText(_translate("MainWindow", "IOU阈值："))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "图片检测"))
        self.pushButton_2.setText(_translate("MainWindow", "开始检测"))
        self.label_11.setText(_translate("MainWindow", "置信度阈值："))
        self.label_12.setText(_translate("MainWindow", "IOU阈值："))
        self.label_13.setText(_translate("MainWindow", "期望输出帧率："))
        self.label_14.setText(_translate("MainWindow", "检测频率："))
        self.pushButton_3.setText(_translate("MainWindow", "退出检测"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "视频检测"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "实时检测"))
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_2.setTitle(_translate("MainWindow", "ROI"))
        self.menu_3.setTitle(_translate("MainWindow", "参数配置"))
        self.menu_4.setTitle(_translate("MainWindow", "窗口"))
        self.menu_5.setTitle(_translate("MainWindow", "帮助"))
        self.actionImport.setText(_translate("MainWindow", "Import"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionSettings.setText(_translate("MainWindow", "Settings"))
        self.actionImage.setText(_translate("MainWindow", "Image"))
        self.actionVideo.setText(_translate("MainWindow", "Video"))
        self.actionReal_time.setText(_translate("MainWindow", "Real-time"))
        self.actionfps_tets.setText(_translate("MainWindow", "fps only"))
        self.actionWeghts.setText(_translate("MainWindow", "weights"))
        self.actiondevice.setText(_translate("MainWindow", "conf-thres"))
        self.actionconf_thres.setText(_translate("MainWindow", "iou-thres"))
        self.actiondevice_2.setText(_translate("MainWindow", "device"))
        self.actionsignal.setText(_translate("MainWindow", "signal"))
        self.actionspeed_curve.setText(_translate("MainWindow", "speed curve"))
        self.actionhandbook.setText(_translate("MainWindow", "handbook"))
        self.actioncontact_us.setText(_translate("MainWindow", "contact us"))
        self.actionIs_ROI.setText(_translate("MainWindow", "Is ROI"))
        self.actionROI_Settings.setText(_translate("MainWindow", "ROI Settings"))
