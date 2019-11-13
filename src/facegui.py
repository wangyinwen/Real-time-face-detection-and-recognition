# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'facegui.ui'
#
# Created by: PyQt5 UI code generator 5.12
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.opencamera = QtWidgets.QPushButton(self.centralwidget)
        self.opencamera.setGeometry(QtCore.QRect(460, 60, 111, 41))
        self.opencamera.setObjectName("opencamera")
        self.showcamera = QtWidgets.QLabel(self.centralwidget)
        self.showcamera.setGeometry(QtCore.QRect(10, 0, 431, 291))
        self.showcamera.setObjectName("showcamera")
        self.closecamera = QtWidgets.QPushButton(self.centralwidget)
        self.closecamera.setGeometry(QtCore.QRect(460, 120, 101, 41))
        self.closecamera.setObjectName("closecamera")
        self.addface = QtWidgets.QPushButton(self.centralwidget)
        self.addface.setGeometry(QtCore.QRect(460, 460, 91, 51))
        self.addface.setObjectName("addface")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(460, 280, 81, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.captureface = QtWidgets.QPushButton(self.centralwidget)
        self.captureface.setGeometry(QtCore.QRect(454, 322, 101, 51))
        self.captureface.setObjectName("captureface")
        self.saveface = QtWidgets.QPushButton(self.centralwidget)
        self.saveface.setGeometry(QtCore.QRect(460, 390, 101, 51))
        self.saveface.setObjectName("saveface")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.opencamera.setText(_translate("MainWindow", "OpenCamera"))
        self.showcamera.setText(_translate("MainWindow", "TextLabel"))
        self.closecamera.setText(_translate("MainWindow", "closecamera"))
        self.addface.setText(_translate("MainWindow", "建库"))
        self.captureface.setText(_translate("MainWindow", "采集样本"))
        self.saveface.setText(_translate("MainWindow", "保存人脸"))


