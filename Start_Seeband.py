import sys
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QVBoxLayout, QStackedLayout, QGridLayout, QHBoxLayout,
    QLabel, QToolBar, QAction, QPushButton, QWidget, QSlider, QFileDialog,
    QRadioButton, QComboBox, QLineEdit, QSizePolicy, QDesktopWidget, QSplashScreen
)
from PyQt5.QtGui import QIcon, QIntValidator, QDoubleValidator, QPainterPath, QRegion, QIcon, QFont, QPixmap, QMovie
from PyQt5.QtCore import Qt, QSize, QRectF, QPoint, QRect, QThread, pyqtSignal
import Fit_class_final
import pyqtgraph as pg
import numpy as np
import dataProcessing as dp
import ai

import os
#Special library
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


'''
Main window
'''
class MainWindow(QMainWindow):

    def __init__(self):
        '''
        Initializing some parameters
        '''
        
        self.T_list, self.S_list = [], []
        self.minT = 50
        self.maxT = 800
        self.interpolated_data = None
        self.threshold = 0.1
        self.max_iter = 20
        
        """
        Initiate the scatter-plots for the experimental data 
        """
        
        self.scatter_see_1PB_S = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter_see_1PB_Srho = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter_see_1PB_SrhoHall = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter_see_2PB_S = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter_see_2PB_Srho = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter_see_2PB_SrhoHall = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        
        self.scatter_res_1PB_S = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter_res_1PB_Srho = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter_res_1PB_SrhoHall = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter_res_2PB_S = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter_res_2PB_Srho = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter_res_2PB_SrhoHall = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        
        self.scatter_hall_1PB_S = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter_hall_1PB_Srho = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter_hall_1PB_SrhoHall = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter_hall_2PB_S = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter_hall_2PB_Srho = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter_hall_2PB_SrhoHall = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        
        self.experimental_data_scatter = {"see": {"1PB_S": self.scatter_see_1PB_S,
                                     "1PB_Srho": self.scatter_see_1PB_Srho,
                                     "1PB_SrhoHall": self.scatter_see_1PB_SrhoHall,
                                     "2PB_S": self.scatter_see_2PB_S,
                                     "2PB_Srho": self.scatter_see_2PB_Srho,
                                     "2PB_SrhoHall": self.scatter_see_2PB_SrhoHall},
                                     "res": {"1PB_S": self.scatter_res_1PB_S,
                                    "1PB_Srho": self.scatter_res_1PB_Srho,
                                    "1PB_SrhoHall": self.scatter_res_1PB_SrhoHall,
                                    "2PB_S": self.scatter_res_2PB_S,
                                    "2PB_Srho": self.scatter_res_2PB_Srho,
                                    "2PB_SrhoHall": self.scatter_res_2PB_SrhoHall},
                                      "hall": {"1PB_S": self.scatter_hall_1PB_S,
                                     "1PB_Srho": self.scatter_hall_1PB_Srho,
                                     "1PB_SrhoHall": self.scatter_hall_1PB_SrhoHall,
                                     "2PB_S": self.scatter_hall_2PB_S,
                                     "2PB_Srho": self.scatter_hall_2PB_Srho,
                                     "2PB_SrhoHall": self.scatter_hall_2PB_SrhoHall}             
                                         }
        
        '''
        Main window
        '''

        # Main window and property initialization
        # super(MainWindow, self).__init__()
        super().__init__()
        # self.resize(950, 950)
        self.resize(1440, 950)
        self.setWindowTitle("Seeband")
        icon = QIcon("Icons/cross-button.png")
        # Remove automatic os-dependent toolbar
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setWindowIcon(QIcon("Icons/seeband_logo_nobg.png"))
        # Make edges round
        radius = 10.0
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), radius, radius)
        mask = QRegion(path.toFillPolygon().toPolygon())
        self.setMask(mask)

        # Make window movable
        self.center()
        self.oldPos = self.pos()
        
        splash_pix = QPixmap("Icons/splash_image.jpg")  # Replace 'splash_image.png' with your image path
        splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
        splash.setMask(splash_pix.mask())
        splash.show()

        # Layouts for the main window, stacked layout and layout of the individual pages
        self.program_layout = QVBoxLayout()
        self.button_layout = QHBoxLayout()
        self.stacked_layout = QStackedLayout()

        # Initialize the different tabs
        self.page_input = QWidget()
        self.page_seebeckOnly_1PB = QWidget()
        self.page_seebeckRho_1PB = QWidget()
        self.page_seebeckRhoHall_1PB = QWidget()
        self.page_seebeckOnly_2PB = QWidget()
        self.page_seebeckRho_2PB = QWidget()
        self.page_seebeckRhoHall_2PB = QWidget()

        # Assign the respective Gridlayout to the respective pages
        self.page_input_layout = QGridLayout()
        self.page_seebeckOnly_1PB_layout = QGridLayout()
        self.page_seebeckRho_1PB_layout = QGridLayout()
        self.page_seebeckRhoHall_1PB_layout = QGridLayout()
        self.page_seebeckOnly_2PB_layout = QGridLayout()
        self.page_seebeckRho_2PB_layout = QGridLayout()
        self.page_seebeckRhoHall_2PB_layout = QGridLayout()

        # Adding the different pages of the tabs to the stacked_layout-main-widget
        self.stacked_layout.addWidget(self.page_input)
        self.stacked_layout.addWidget(self.page_seebeckOnly_1PB)
        self.stacked_layout.addWidget(self.page_seebeckRho_1PB)
        self.stacked_layout.addWidget(self.page_seebeckRhoHall_1PB)
        self.stacked_layout.addWidget(self.page_seebeckOnly_2PB)
        self.stacked_layout.addWidget(self.page_seebeckRho_2PB)
        self.stacked_layout.addWidget(self.page_seebeckRhoHall_2PB)

        # Asign the layouts to the respective pages
        self.page_input.setLayout(self.page_input_layout)
        self.page_seebeckOnly_1PB.setLayout(self.page_seebeckOnly_1PB_layout)
        self.page_seebeckRho_1PB.setLayout(self.page_seebeckRho_1PB_layout)
        self.page_seebeckRhoHall_1PB.setLayout(self.page_seebeckRhoHall_1PB_layout)
        self.page_seebeckOnly_2PB.setLayout(self.page_seebeckOnly_2PB_layout)
        self.page_seebeckRho_2PB.setLayout(self.page_seebeckRho_2PB_layout)
        self.page_seebeckRhoHall_2PB.setLayout(self.page_seebeckRhoHall_2PB_layout)

        #Set up the toolbar
        self.set_up_toolbar()
        
        '''
        Widget intialization for the different pages
        ----------------------------------------------------------------------------------------------------------------
        '''

        '''
        General style elements
        '''
        self.pen2 = pg.mkPen('r', width=2, style=Qt.SolidLine)
        self.pen3 = pg.mkPen("#00AA14", width=3, style=Qt.DashLine)
        self.tick_font = QFont()
        self.tick_font.setPixelSize(16)
        self.pen_axis = pg.mkPen("#AAAAAA", width=2, style=Qt.SolidLine)
        #Band structure plot
        self.pen_band1 = pg.mkPen((50, 198, 27), width=4, style=Qt.SolidLine)
        self.pen_band2 = pg.mkPen((255, 80, 0), width=4, style=Qt.SolidLine)
        self.pen_band3 = pg.mkPen((255, 24, 24), width=2.5, style=Qt.DashLine) 
        self.tick_font_3bands = QFont()
        self.tick_font_3bands.setPixelSize(14)
        self.pen_axis_3bands = pg.mkPen("#AAAAAA", width=1.5, style=Qt.SolidLine)

        # Data structure for different widgets
        # tested to initialize the widgets in a loop; it works, but I am not sure if it is worth it
        widget_data = [
            {
                
                'type': QLabel,
                'text': "Data management",
                'properties': {
                    'class': 'title'
                },
                'name': 'label_p1',
                'pos' : [0,0,1,1] ,
            },

            {
                'type': QPushButton,
                'text': "Import Seebeck data",
                'click_function': self.seeb_databutton_clicked,
                'name': 'seeb_databutton',
                'pos' : [1,0,1,2] ,
            },
            {
                'type': QPushButton,
                'text': "Clear Seeb data",
                'click_function': self.clear_seeb_data_button_clicked,
                'name': 'clear_seeb_data_button',
                'pos' : [2,0,1,2] ,
            },
            {
                'type': QPushButton,
                'text': "Interpolate Seeb",
                'click_function': self.int_seeb_data_button_clicked,
                'checkable': True,
                'checked': False,
                'name': 'int_seeb_data_button',
                'pos' : [3,0,1,2] ,
            },
            {
                'type': QLabel,
                'text': " ",
                'properties': {
                    'class': 'PCH'
                },
                'name': 'place_holder_page1_2',
                'pos' : [4,0,1,2] ,
            },
            {
                'type': QPushButton,
                'text': "Import Resistivity data",
                'click_function': self.res_databutton_clicked,
                'name': 'res_databutton',
                'pos' : [5,0,1,2] ,
            },
            {
                'type': QPushButton,
                'text': "Clear Res data",
                'click_function': self.clear_res_data_button_clicked,
                'name': 'clear_res_data_button',
                'pos' : [6,0,1,2] ,
            },
            {
                'type': QPushButton,
                'text': "Interpolated Res",
                'click_function': self.int_res_data_button_clicked,
                'checkable': True,
                'checked': False,
                'name': 'int_res_data_button',
                'pos' : [7,0,1,2] ,
            },
            {
                'type': QLabel,
                'text': " ",
                'properties': {
                    'class': 'PCH'
                },
                'name': 'place_holder_page1_4',
                'pos' : [8,0,1,2] ,
            },
            {
                'type': QPushButton,
                'text': "Import Hall data",
                'click_function': self.hall_databutton_clicked,
                'name': 'hall_databutton',
                'pos' : [9,0,1,2] ,
            },
            {
                'type': QPushButton,
                'text': "Clear Hall data",
                'click_function': self.clear_hall_data_button_clicked,
                'name': 'clear_hall_data_button',
                'pos' : [10,0,1,2] ,
            },
            {
                'type': QPushButton,
                'text': "Interpolate Hall",
                'click_function': self.int_hall_data_button_clicked,
                'checkable': True,
                'checked': False,
                'name': 'int_hall_data_button', 'pos' : [11,0,1,2] ,
            },
            {
                'type': QLabel,
                'text': " ",
                'properties': {
                    'class': 'PCH'
                },
                'name': 'place_holder_page1_6',
                'pos' : [12,0,1,2] ,
            },

            # Add more widgets as needed
        ]
        self.initialized_widgets = {}
        # Loop through the widget data and create widgets
        for widget_info in widget_data:
            widget_type = widget_info['type']
            widget = widget_type(widget_info.get('text', ''))

            name = widget_info.get('name')
            if name:
                widget.setObjectName(name)

            properties = widget_info.get('properties', {})
            for property_name, property_value in properties.items():
                widget.setProperty(property_name, property_value)

            click_function = widget_info.get('click_function')
            if click_function:
                widget.clicked.connect(click_function)

            checkable = widget_info.get('checkable')
            if checkable:
                widget.setCheckable(True)
                checked = widget_info.get('checked', False)
                widget.setChecked(checked)
            self.initialized_widgets[name] = widget

        #Seebeck_preview_plot
        self.graph_box = pg.GraphicsLayoutWidget()
        self.graph = self.graph_box.addPlot(2, 2)
        #self.graph_box_viewbox = self.graph.getViewBox()

        self.pen3 = pg.mkPen("#00AA14", width=3, style=Qt.DashLine)

        self.scatter = pg.ScatterPlotItem(
            size=5, brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter_int = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush(255, 255, 255, 120))

        self.graph.setDefaultPadding(padding=0.1)

        self.pen = pg.mkPen('w', width=1, style=Qt.SolidLine)
        self.border_pen = pg.mkPen((55, 55, 63), width=2, style=Qt.SolidLine)
        self.graph.setLabel(
            "bottom", '<p style="font-size:18px;color=white;font-style:bold">Temperature (T) </p>')
        self.graph.setLabel(
            "left", '<p style="font-size:18px;color=white;font-style:bold">Seebeck coefficient (\u03BCV/K) </p>')
        self.graph_box.setBackground((5, 5, 5))
        #self.graph_box.layout.setContentsMargins(20, 20, 20, 20)

        self.graph_box.setStyleSheet(
            "QFrame {border: 5px solid #252530;border-bottom: 5px solid #353540; border-top: 5px solid #151520}")
        
        
        #Resistivity preview plot
        self.preview_res = pg.GraphicsLayoutWidget()
        self.preview_res_graph = self.preview_res.addPlot(2, 2)

        self.pen3 = pg.mkPen("#00AA14", width=3, style=Qt.DashLine)

        self.scatter_res = pg.ScatterPlotItem(
            size=5, brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter_res_int = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush(255, 255, 255, 120))

        self.preview_res_graph.setDefaultPadding(padding=0.1)

        self.pen = pg.mkPen('w', width=1, style=Qt.SolidLine)
        self.border_pen = pg.mkPen((55, 55, 63), width=2, style=Qt.SolidLine)
        self.preview_res_graph.setLabel(
            "bottom", '<p style="font-size:18px;color=white;font-style:bold">Temperature (T) </p>')
        self.preview_res_graph.setLabel(
            "left", '<p style="font-size:18px;color=white;font-style:bold">Resistivity (\u03A9m) </p>')
        self.preview_res.setBackground((5, 5, 5))

        self.preview_res.setStyleSheet(
            "QFrame {border: 5px solid #252530;border-bottom: 5px solid #353540; border-top: 5px solid #151520}")
        
        
        #Hall preview plot
        self.preview_hall = pg.GraphicsLayoutWidget()
        self.preview_hall_graph = self.preview_hall.addPlot(2, 2)

        self.pen3 = pg.mkPen("#00AA14", width=3, style=Qt.DashLine)

        self.scatter_hall = pg.ScatterPlotItem(
            size=5, brush=pg.mkBrush(255, 255, 255, 120))
        self.scatter_hall_int = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush(255, 255, 255, 120))

        self.preview_hall_graph.setDefaultPadding(padding=0.1)

        self.pen = pg.mkPen('w', width=1, style=Qt.SolidLine)
        self.border_pen = pg.mkPen((55, 55, 63), width=2, style=Qt.SolidLine)
        self.preview_hall_graph.setLabel(
            "bottom", '<p style="font-size:18px;color=white;font-style:bold">Temperature (T) </p>')
        self.preview_hall_graph.setLabel(
            "left", '<p style="font-size:18px;color=white;font-style:bold">Hall coefficient [\u03A9m/T] </p>')
        self.preview_hall.setBackground((5, 5, 5))
        #self.graph_box.layout.setContentsMargins(20, 20, 20, 20)

        self.preview_hall.setStyleSheet(
            "QFrame {border: 5px solid #252530;border-bottom: 5px solid #353540; border-top: 5px solid #151520}")
        
        # Input Widgets for lower and upper temperature boundaries of the fit
        self.onlyInt = QIntValidator()
        self.onlyInt.setRange(-10000, 10000)
        self.onlyFloat = QDoubleValidator()
        self.onlyFloat.setRange(-100000.00, 100000.00)


        #Place_holders
        self.label_placeholder_page1 = QLabel(" ")
        self.label_placeholder_page1.setProperty("class","PCH")
        

        #Add the widget to the layout or parent widget
        for i in range(len(widget_data)):
            self.page_input_layout.addWidget(
                self.initialized_widgets[widget_data[i]["name"]], widget_data[i]["pos"][0], widget_data[i]["pos"][1], widget_data[i]["pos"][2], widget_data[i]["pos"][3])

        
        self.page_input_layout.addWidget(self.graph_box, 1, 2, 4, 5)
        self.page_input_layout.addWidget(self.preview_res, 5, 2, 4, 5)
        self.page_input_layout.addWidget(self.preview_hall, 9, 2, 4, 5)
        # for i in range(16):
        #     self.page1_layout.setRowStretch(i, 1)
        for i in range(8):
            self.page_input_layout.setColumnStretch(i, 1)


        # Creating the gif for highlighting the ongoing fitting process
        self.set_up_waiting_gifs()

        '''
        Set up all windows for the data visualization, manipulation and fit
        '''
        #Set up the window for 1PB Seebeck fit
        self.set_up_1PB_see_window()
        
        #Set up the window for 1PB Seebeck and resistivity fit
        self.set_up_1PB_res_window()
        
        #Set up the window for 1PB Seebeck, resistivity and Hall coefficient fit
        self.set_up_1PB_hall_window()
        
        #Set up the window for 2PB Seebeck fit
        self.set_up_2PB_see_window()
            
        #Set up the window for 2PB Seebeck and resistivity fit
        self.set_up_2PB_res_window()

        #Set up the window for 2PB Seebeck, resistivity and Hall coefficient fit
        self.set_up_2PB_hall_window()
        
        
        #Load all AI models
        self.model_100_400, self.model_200_500, self.model_300_600, self.model_300_800, self.model_400_700 = ai.load_models()
        
        '''
        Add all the pages and everything together
        '''

        widget = QWidget()
        widget.setLayout(self.program_layout)
        self.setCentralWidget(widget)
        self.program_layout.addLayout(self.button_layout)
        self.program_layout.addLayout(self.stacked_layout)
        
        # self.loading_label.hide()
        splash.hide()

    def set_up_toolbar(self):
        #Variable determining if the fitting windows are working with 1 or 2 parabolic bands; can be changed by pressing the 1PB/2PB button
        self.bands = 2
        # set toolbar
        self.toolbar = QToolBar("My main toolbar")
        self.toolbar.setIconSize(QSize(32, 32))
        self.toolbar.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self.toolbar.setMovable(False)
        self.addToolBar(self.toolbar)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # random button for fun
        self.logo_action = QAction(
            QIcon("Icons/seeband_logo_nobg.png"), "Siimply fitting", self)
        self.logo_action.setStatusTip("Siimply fitting")
        self.logo_action.triggered.connect(self.onMyToolBar_logo)
        self.logo_action.setCheckable(True)

        # Tab-button for the Fitting window
        self.importData_action = QAction(
            QIcon("Icons/categories.png"), "Switch to the Fitting window", self)
        self.importData_action.setStatusTip("Switch to Tryout")
        self.importData_action.triggered.connect(self.onMyToolBar_importData)
        self.importData_action.setCheckable(False)

        # Tab-button for the 1-band fitting window
        self.switch1PB_action = QAction(
            QIcon("Icons/1PB.png"), "Switch to 1PB", self)
        self.switch1PB_action.setStatusTip(
            "Set the number of considered bands to 1")
        self.switch1PB_action.triggered.connect(self.onMyToolBar_switch1PB)
        self.switch1PB_action.setCheckable(True)

        self.switch2PB_action = QAction(
            QIcon("Icons/2PB.png"), "Switch to 2PB", self)
        self.switch2PB_action.setStatusTip(
            "Set the number of considered bands to 2")
        self.switch2PB_action.triggered.connect(self.onMyToolBar_switch2PB)
        self.switch2PB_action.setCheckable(True)
        self.switch2PB_action.setChecked(True)


        # Tab-button for the 2-band fitting window
        self.seebeckOnly_action = QAction(
            QIcon("Icons/S.png"), "Switch to the Manipulate window for Seebeck only", self)
        self.seebeckOnly_action.setStatusTip(
            "Switch to the Manipulate window for Seebeck only")
        self.seebeckOnly_action.triggered.connect(self.onMyToolBar_seebeckOnly)
        self.seebeckOnly_action.setCheckable(False)

        #Tab-button for the 2-band fitting window
        self.seebeckRho_action = QAction(
            QIcon("Icons/Srho.png"), "Switch to the Manipulate window for Seebeck and resistivity", self)
        self.seebeckRho_action.setStatusTip(
            "Switch to the Manipulate window for Seebeck and resistivity")
        self.seebeckRho_action.triggered.connect(self.onMyToolBar_seebeckRho)
        self.seebeckRho_action.setCheckable(False)
        
        self.seebeckRhoHall_action = QAction(
            QIcon("Icons/SrhoRH.png"), "Switch to the Manipulate window for Seebeck, resistivity and Hall", self)
        self.seebeckRhoHall_action.setStatusTip(
            "Switch to the Manipulate window for Seebeck, resistivity and Hall")
        self.seebeckRhoHall_action.triggered.connect(self.onMyToolBar_seebeckRhoHall)
        self.seebeckRhoHall_action.setCheckable(False)
        
        # End the program button
        self.exit_action = QAction(
            QIcon("Icons/cross-button.png"), "Quit the program", self)
        self.exit_action.setStatusTip("Quit the program")
        self.exit_action.triggered.connect(self.onMyToolBar_exit)
        self.exit_action.setCheckable(False)

        # Asign the button-actions to the toolbar
        self.toolbar.addAction(self.logo_action)
        self.toolbar.addAction(self.importData_action)
        self.toolbar.addAction(self.switch1PB_action)
        self.toolbar.addAction(self.switch2PB_action)
        self.toolbar.addAction(self.seebeckOnly_action)
        self.toolbar.addAction(self.seebeckRho_action)
        self.toolbar.addAction(self.seebeckRhoHall_action)
        self.toolbar.addWidget(spacer)
        self.toolbar.addAction(self.exit_action)

    def set_up_1PB_see_window(self):
        # Descriptive label on top
        self.label_p2 = QLabel("1PB: <i>S</i>")
        self.label_p2.setProperty("class", "title")

        # Sliders for the 1PB-model parameters   
        self.mass_slider_1PB = QSlider(Qt.Horizontal)
        self.mass_slider_1PB.setMinimum(-300)
        self.mass_slider_1PB.setMaximum(300)
        self.mass_slider_1PB.setSingleStep(1)
        self.mass_slider_1PB.setValue(101)
        self.mass_slider_1PB.valueChanged.connect(self.manipulate_value_changed_1PB_see)
        
        self.mass_lineedit_1PB = QLineEdit("1.01")
        self.mass_lineedit_1PB.setProperty("class", "slideredit")
        self.mass_lineedit_1PB.returnPressed.connect(self.line_edit_changed_1PB)


        self.mass_SPB_label = QLabel('mass = ')
        self.mass_fit_SPB_label = QLabel('mass = ')
        self.mass_fit_SPB_label.setProperty("class", "fit")
        self.mass_fit_SPB_label.setVisible(False)
        self.mass_fit_SPB_label_PCH = QLabel()
        self.mass_fit_SPB_label_PCH.setProperty("class", "PCH")

        # Position of the Fermi level in Kelvin
        self.fermi_slider_1PB = QSlider(Qt.Horizontal)
        self.fermi_slider_1PB.setMinimum(-3000)
        self.fermi_slider_1PB.setMaximum(3000)
        self.fermi_slider_1PB.setSingleStep(1)
        self.fermi_slider_1PB.setValue(100)
        self.fermi_slider_1PB.valueChanged.connect(self.manipulate_value_changed_1PB_see)
        
        self.fermi_lineedit_1PB = QLineEdit("100")
        self.fermi_lineedit_1PB.setProperty("class", "slideredit")
        self.fermi_lineedit_1PB.returnPressed.connect(self.line_edit_changed_1PB)

        self.fermi_label_1PB = QLabel("<p style='font-size: 14px;'><i>E</i><sub>F</sub> =</p>")
        self.fermi_fit_label_1PB = QLabel("<p style='font-size: 14px;'><i>E</i><sub>F</sub> =</p>")
        self.fermi_fit_label_1PB.setProperty("class", "fit")
        self.fermi_fit_label_1PB.setVisible(False)
        self.fermi_fit_label_PCH_1PB = QLabel()
        self.fermi_fit_label_PCH_1PB.setProperty("class", "PCH")

        # Execute a fit using the starting parameters from the manipulate button
        self.manip_fit_button_1PB = QPushButton("Fit from manipulate")
        self.manip_fit_button_1PB.clicked.connect(self.manip_fit_button_clicked_1PB_see)

        # Activate or deactivate the plotting of the experimental data)
        self.experimental_data_1PB = QRadioButton("Experimental data")
        self.experimental_data_1PB.toggled.connect(self.experimental_data_toggled_1PB)
        self.experimental_data_1PB.setDisabled(True)

        # Manipulate plot (dependent on the 2PB-model parameters)

        self.graph_SPB = pg.PlotWidget()
        self.graph_SPB.setBackground((5, 5, 5))
        self.graph_SPB.setTitle(
            "<span style=\"color:white;font-size:12pt;font:Lato;font-weight:semi-bold\">Calculated Seebeck coefficient</span>")
        self.graph_SPB.setProperty("class", "seebeck")
        # Plot of the experimental data that can be imported
        self.scatter_SPB = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.is_scatter_SPB_there = False

        # Plot the resulting curve from the initial parameters
        x_val, y_val = TE.spb_see_calc(self.minT, self.maxT, 50, self.mass_slider_1PB.value()/100., self.fermi_slider_1PB.value())
       
        
        self.manipulate_plot_1PB = pg.PlotCurveItem()
        self.manipulate_plot_1PB.setData(x_val, y_val, pen=self.pen2, clear=True)

        # Plot for the resulting curve from the fit (is left empty here and filled in a function down below)
        self.manip_fit_plot_1PB = pg.PlotCurveItem()

        self.legend_graph_SPB = self.graph_SPB.addLegend()
        self.graph_SPB.addItem(self.manipulate_plot_1PB)
        self.graph_SPB.addItem(self.manip_fit_plot_1PB)

        self.graph_SPB_Xaxis = self.graph_SPB.getAxis("bottom")
        self.graph_SPB_Yaxis = self.graph_SPB.getAxis("left")
        self.graph_SPB_Xaxis.setStyle(tickTextOffset=10)
        self.graph_SPB_Yaxis.setStyle(tickTextOffset=10)
        self.graph_SPB_Xaxis.setHeight(h=60)
        self.graph_SPB_Yaxis.setWidth(w=60)
        self.graph_SPB_Xaxis.setLabel(
            '<p style="font-size:14px;color=white">Temperature [K] </p>')
        self.graph_SPB_Yaxis.setLabel(
            '<p style="font-size:14px;color=white">Seebeck coefficient [\u03BCV/K] </p>')

        self.legend_graph_SPB.addItem(self.manipulate_plot_1PB, "Manipulate")
        self.graph_SPB.setContentsMargins(50, 50, 50, 50)

        # Plot the resulting effective band structure
        self.graph_SPB_band = pg.PlotWidget()
        self.graph_SPB_band_item = pg.PlotCurveItem()
        self.graph_SPB_band.setYRange(-1000, 1000)

        self.graph_SPB_band.setLabel(
            "bottom", '<p style="font-size:14px;color=white">k space </p>')
        self.graph_SPB_band.setLabel(
            "left", '<p style="font-size:14px;color=white">E-E_F (K) </p>')

        self.graph_SPB_band_2 = pg.PlotWidget()
        self.band1_1PB = pg.PlotCurveItem()
        self.band3_1PB = pg.PlotCurveItem()

        self.band1_data_1PB = TE.get_parabolic_band(
            -20, 20, 100, self.mass_slider_1PB.value()/100., 0)
        self.band3_data_1PB = TE.get_parabolic_band(
            -20, 20, 100, 10000000000, self.fermi_slider_1PB.value())
        self.band1_1PB.setData(
            self.band1_data_1PB[0], self.band1_data_1PB[1], pen=self.pen_band1, clear=True)
        self.band3_1PB.setData(
            self.band3_data_1PB[0], self.band3_data_1PB[1], pen=self.pen_band3, clear=True)
        self.graph_SPB_band.addItem(self.band1_1PB)
        self.graph_SPB_band.addItem(self.band3_1PB)
        
        self.adv_opt_button_1PB = QPushButton("Advanced options") 
        self.adv_opt_button_1PB.clicked.connect(self.adv_opt_button_clicked_1PB)
        
        self.ind_cont_button_1PB = QPushButton("Additional graphs")
        self.ind_cont_button_1PB.clicked.connect(self.ind_cont_button_clicked_1PB)
        
        self.print_data_button_1PB = QPushButton("Data to file")
        self.print_data_button_1PB.setToolTip("Save all visible data to a single file")
        self.print_data_button_1PB.clicked.connect(self.print_data_button_clicked_see_1PB)
        
        self.print_params_button_1PB = QPushButton("Parameters to file")
        self.print_params_button_1PB.setToolTip("Save all parameters to a single file")
        self.print_params_button_1PB.clicked.connect(self.print_params_button_clicked_see_1PB)
    
        self.label_1PB_1 = QLabel('aaaaaa')
        self.label_1PB_1.setProperty("class", "PCH")
        
        #!!! Preliminary_degeneracy value <- not added to GUI yet
        self.Nv_value_1PB = QLineEdit('1')
        self.Nv_value_1PB.setProperty("class", "slideredit")
        self.Nv_value_1PB.setValidator(self.onlyInt)  # assuming onlyInt is a QValidator object defined elsewhere
        self.Nv_value_1PB.returnPressed.connect(self.manipulate_value_changed_1PB_see)

        # Adding the widgets to the window on their respective grid slots
        self.page_seebeckOnly_1PB_layout.addWidget(self.label_p2, 0, 0, 1, 3)
        self.page_seebeckOnly_1PB_layout.addWidget(self.experimental_data_1PB, 0, 3, 1, 3)
        self.page_seebeckOnly_1PB_layout.addWidget(self.manip_fit_button_1PB, 0, 6, 1, 3)
        self.page_seebeckOnly_1PB_layout.addWidget(self.gif_label_spb_see, 0, 7, 1, 1)
        self.page_seebeckOnly_1PB_layout.addWidget(self.label_1PB_1, 1, 0, 1, 3)
        self.page_seebeckOnly_1PB_layout.addWidget(self.mass_slider_1PB, 2, 0, 1, 3)
        self.page_seebeckOnly_1PB_layout.addWidget(self.mass_SPB_label, 3, 0, 1, 2)
        self.page_seebeckOnly_1PB_layout.addWidget(self.mass_lineedit_1PB, 3, 2, 1, 1)
        self.page_seebeckOnly_1PB_layout.addWidget(self.mass_fit_SPB_label, 4, 0, 1, 3)
        self.page_seebeckOnly_1PB_layout.addWidget(self.mass_fit_SPB_label_PCH, 4, 0, 1, 3)
        self.page_seebeckOnly_1PB_layout.addWidget(self.fermi_slider_1PB, 5, 0, 1, 3)
        self.page_seebeckOnly_1PB_layout.addWidget(self.fermi_label_1PB, 6, 0, 1, 2)
        self.page_seebeckOnly_1PB_layout.addWidget(self.fermi_lineedit_1PB, 6, 2, 1, 3)
        self.page_seebeckOnly_1PB_layout.addWidget(self.fermi_fit_label_1PB, 7, 0, 1, 3)
        self.page_seebeckOnly_1PB_layout.addWidget(self.fermi_fit_label_PCH_1PB, 7, 0, 1, 3)
        self.page_seebeckOnly_1PB_layout.addWidget(self.graph_SPB, 1, 4, 10, 5)
        self.page_seebeckOnly_1PB_layout.addWidget(self.graph_SPB_band, 1, 15, 8, 4)
        self.page_seebeckOnly_1PB_layout.addWidget(self.adv_opt_button_1PB, 9, 15, 1, 2)
        self.page_seebeckOnly_1PB_layout.addWidget(self.print_data_button_1PB, 10, 15, 1, 2)
        self.page_seebeckOnly_1PB_layout.addWidget(self.ind_cont_button_1PB, 9, 17, 1, 2)
        self.page_seebeckOnly_1PB_layout.addWidget(self.print_params_button_1PB, 10, 17, 1, 2)

    def set_up_1PB_res_window(self):
        self.para_A_lower_limit_1PB_res = 0
        self.para_A_upper_limit_1PB_res = 1e6
        self.para_C_lower_limit_1PB_res = 1e-3
        self.para_C_upper_limit_1PB_res = 1e3
        self.para_F_lower_limit_1PB_res = 0
        self.para_F_upper_limit_1PB_res = 1e5
        
        # Descriptive label on top
        self.label_p2_res = QLabel("1PB: <i>S</i> + <i>&rho;</i>")
        self.label_p2_res.setProperty("class", "title")

        # Sliders for the 1PB-model parameters   
        self.mass_slider_1PB_res = QSlider(Qt.Horizontal)
        self.mass_slider_1PB_res.setMinimum(-300)
        self.mass_slider_1PB_res.setMaximum(300)
        self.mass_slider_1PB_res.setSingleStep(1)
        self.mass_slider_1PB_res.setValue(-100)
        self.mass_slider_1PB_res.valueChanged.connect(self.manipulate_value_changed_1PB_res)

        self.mass_SPB_label_res = QLabel('mass =')
        
        self.mass_lineedit_1PB_res = QLineEdit("-1")
        self.mass_lineedit_1PB_res.setProperty("class", "slideredit")
        self.mass_lineedit_1PB_res.returnPressed.connect(self.line_edit_changed_1PB_res)  # Same slot as the original
        self.mass_fit_SPB_label_res = QLabel('mass = ')
        self.mass_fit_SPB_label_res.setProperty("class", "fit")
        self.mass_fit_SPB_label_res.setVisible(False)
        self.mass_fit_SPB_label_PCH_res = QLabel()
        self.mass_fit_SPB_label_PCH_res.setProperty("class", "PCH")

        # Position of the Fermi level in Kelvin
        self.fermi_slider_1PB_res = QSlider(Qt.Horizontal)
        self.fermi_slider_1PB_res.setMinimum(-3000)
        self.fermi_slider_1PB_res.setMaximum(3000)
        self.fermi_slider_1PB_res.setSingleStep(1)
        self.fermi_slider_1PB_res.setValue(-100)
        self.fermi_slider_1PB_res.valueChanged.connect(self.manipulate_value_changed_1PB_res)
        
        self.fermi_label_1PB_res = QLabel('Fermi energy =')
        
        self.fermi_lineedit_1PB_res = QLineEdit("-100")
        self.fermi_lineedit_1PB_res.setProperty("class", "slideredit")
        self.fermi_lineedit_1PB_res.returnPressed.connect(self.line_edit_changed_1PB_res)  # Same slot as the original
        
        self.fermi_fit_label_1PB_res = QLabel("<p style='font-size: 14px;'><i>E</i><sub>F</sub> =</p>")
        self.fermi_fit_label_1PB_res.setProperty("class", "fit")
        self.fermi_fit_label_1PB_res.setVisible(False)
        self.fermi_fit_label_PCH_1PB_res = QLabel()
        self.fermi_fit_label_PCH_1PB_res.setProperty("class", "PCH")
        
        self.para_A_slider_1PB_res = QSlider(Qt.Horizontal)
        self.para_A_slider_1PB_res.setMinimum(0)
        self.para_A_slider_1PB_res.setMaximum(int(1e6))
        self.para_A_slider_1PB_res.setSingleStep(1000)
        self.para_A_slider_1PB_res.setValue(40000)  # Initial value
        self.para_A_slider_1PB_res.valueChanged.connect(self.manipulate_value_changed_1PB_res)  # Assuming you want to use the same slot
        
        self.para_A_label_1PB_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,1</sub> =</p>")
        self.para_A_label_1PB_res.setProperty("class", "sliderlabel")
        
        self.para_A_lineedit_1PB_res = QLineEdit("40000")
        self.para_A_lineedit_1PB_res.setProperty("class", "slideredit")
        self.para_A_lineedit_1PB_res.returnPressed.connect(self.line_edit_changed_1PB_res)  # Same slot as the original
        
        self.para_A_fit_label_1PB_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,1</sub> = {}</p>".format(0))
        self.para_A_fit_label_1PB_res.setProperty("class", "fit")
        self.para_A_fit_label_1PB_res.setVisible(False)
        
        self.para_A_fit_label_PCH_1PB_res = QLabel()
        self.para_A_fit_label_PCH_1PB_res.setProperty("class", "PCH")
        
        self.para_C_slider_1PB_res = QSlider(Qt.Horizontal)
        self.para_C_slider_1PB_res.setMinimum(0)
        self.para_C_slider_1PB_res.setMaximum(int(300))
        self.para_C_slider_1PB_res.setSingleStep(1)
        self.para_C_slider_1PB_res.setValue(100)  # Initial value
        self.para_C_slider_1PB_res.valueChanged.connect(self.manipulate_value_changed_1PB_res)  # Assuming you want to use the same slot
        
        self.para_C_label_1PB_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,1</sub>/&tau;<sub>dis,1</sub> =</p>")
        self.para_C_label_1PB_res.setProperty("class", "sliderlabel")
        
        self.para_C_lineedit_1PB_res = QLineEdit("100")
        self.para_C_lineedit_1PB_res.setProperty("class", "slideredit")
        self.para_C_lineedit_1PB_res.returnPressed.connect(self.line_edit_changed_1PB_res)  # Same slot as the original
        
        self.para_C_fit_label_1PB_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,1</sub>/&tau;<sub>dis,1</sub> = {}</p>".format(0))
        self.para_C_fit_label_1PB_res.setProperty("class", "fit")
        self.para_C_fit_label_1PB_res.setVisible(False)
        
        self.para_C_fit_label_PCH_1PB_res = QLabel()
        self.para_C_fit_label_PCH_1PB_res.setProperty("class", "PCH")
        
        self.para_C_fit_label_1PB_res.setVisible(False)
        self.para_C_slider_1PB_res.setVisible(False)
        self.para_C_label_1PB_res.setVisible(False)
        self.para_C_lineedit_1PB_res.setVisible(False)
        
        self.para_F_slider_1PB_res = QSlider(Qt.Horizontal)
        self.para_F_slider_1PB_res.setMinimum(0)
        self.para_F_slider_1PB_res.setMaximum(int(1e5))
        self.para_F_slider_1PB_res.setSingleStep(100)
        self.para_F_slider_1PB_res.setValue(4000)  # Initial value
        self.para_F_slider_1PB_res.valueChanged.connect(self.manipulate_value_changed_1PB_res)  # Assuming you want to use the same slot
        
        self.para_F_label_1PB_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>dis,1</sub> =</p>")
        self.para_F_label_1PB_res.setProperty("class", "sliderlabel")
        self.para_F_lineedit_1PB_res = QLineEdit("4000")
        self.para_F_lineedit_1PB_res.setProperty("class", "slideredit")
        self.para_F_lineedit_1PB_res.returnPressed.connect(self.line_edit_changed_1PB_res)  # Same slot as the original
        
        self.para_F_fit_label_1PB_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>dis,1</sub> = {}</p>".format(0))
        self.para_F_fit_label_1PB_res.setProperty("class", "fit")
        self.para_F_fit_label_1PB_res.setVisible(False)
        self.para_F_slider_1PB_res.setVisible(False)
        self.para_F_label_1PB_res.setVisible(False)
        self.para_F_lineedit_1PB_res.setVisible(False)
        
        # Execute a fit using the starting parameters from the manipulate button
        self.manip_fit_button_1PB_res = QPushButton("Fit from manipulate")
        self.manip_fit_button_1PB_res.clicked.connect(self.manip_fit_button_clicked_1PB_res)

        #ComboBox to choose the dominant scattering mechanism  
        self.scattering_type_1PB_res = QComboBox()
        self.scattering_type_1PB_res.addItem("acPh")
        self.scattering_type_1PB_res.addItem("dis")
        self.scattering_type_1PB_res.addItem("acPhDis")
        self.scattering_type_1PB_res.setCurrentIndex(0)
        self.scattering_type_1PB_res.currentIndexChanged.connect(self.type_of_fitting_changed_1PB_res)


        #Calculate data for SPB_res window
        x_val_1PB, y_val_1PB, y_val_1PB_res = TE.spb_see_res_calc(self.minT, self.maxT, 50, self.mass_slider_1PB_res.value()/100., self.fermi_slider_1PB_res.value(), [
            self.para_A_slider_1PB_res.value(), self.para_C_slider_1PB_res.value()/100.,self.para_F_slider_1PB_res.value(),], f"{self.scattering_type_1PB_res.currentText()}")
        x_val_1PB_res = x_val_1PB

        #
        #Graph for Seebeck curve and fit
        #
        self.graph_box_1PB_res = pg.GraphicsLayoutWidget()
        self.graph_1PB_res = self.graph_box_1PB_res.addPlot(2, 3)
        self.graph_box_1PB_res.setStyleSheet("QFrame {border: 5px solid #252530;border-bottom: 5px solid #353540; border-top: 5px solid #151520}")
        self.legend_1PB_res = self.graph_1PB_res.addLegend()
        self.graph_box_1PB_res.setBackground((5, 5, 5))
        self.graph_1PB_res.setTitle(
            "<span style=\"color:#AAAAAA;font-size:16pt;font:Lato;font-weight:semi-bold\">Calculated Seebeck coefficient</span>")
        self.graph_1PB_res.setProperty("class", "seebeck")
        self.graph_Xaxis_1PB_res = self.graph_1PB_res.getAxis("bottom")
        self.graph_Yaxis_1PB_res = self.graph_1PB_res.getAxis("left")
        self.graph_Xaxis_1PB_res.setStyle(tickFont=self.tick_font)
        self.graph_Yaxis_1PB_res.setStyle(tickFont=self.tick_font)
        self.graph_Xaxis_1PB_res.setPen(self.pen_axis)
        self.graph_Yaxis_1PB_res.setPen(self.pen_axis)
        self.graph_Xaxis_1PB_res.setTickPen(self.pen_axis)
        self.graph_Yaxis_1PB_res.setTickPen(self.pen_axis)
        self.graph_Xaxis_1PB_res.setStyle(tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_Yaxis_1PB_res.setStyle(tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_Xaxis_1PB_res.setHeight(h=60)
        self.graph_Yaxis_1PB_res.setWidth(w=60)
        self.graph_Xaxis_1PB_res.setLabel('<p style="font-size:18px;color=white">Temperature [K] </p>')
        self.graph_Yaxis_1PB_res.setLabel('<p style="font-size:18px;color=white">Seebeck coefficient [\u03BCV/K] </p>')

        # Plot the resulting curve from the initial parameters
        self.manipulate_plot_1PB_res = pg.PlotCurveItem()
        self.manipulate_plot_1PB_res.setData(x_val_1PB, y_val_1PB, pen=self.pen2, clear=True)
        self.manip_fit_plot_1PB_res = pg.PlotCurveItem()
        # Plot for the resulting curve from the fit (is left empty here and filled in a function down below)
        self.legend_1PB_res = self.graph_1PB_res.addLegend()
        self.graph_1PB_res.addItem(self.manipulate_plot_1PB_res)
        self.graph_1PB_res.addItem(self.manip_fit_plot_1PB_res)
        self.legend_1PB_res.addItem(self.manipulate_plot_1PB_res, "Manipulate")
        
        #Graph for Resistivity curve and fit
        #
        #Resistivity plot for 2PB
        self.graph_box_1PB_rho_res = pg.GraphicsLayoutWidget()
        self.graph_1PB_rho_res = self.graph_box_1PB_rho_res.addPlot(2, 3)
        self.graph_box_1PB_rho_res.setStyleSheet("QFrame {border: 5px solid #252530;border-bottom: 5px solid #353540; border-top: 5px solid #151520}")
        self.legend_1PB_rho_res = self.graph_1PB_rho_res.addLegend()
        self.graph_box_1PB_rho_res.setBackground((5, 5, 5))
        self.graph_1PB_rho_res.setTitle(
            "<span style=\"color:#AAAAAA;font-size:16pt;font:Lato;font-weight:semi-bold\">Calculated Resistivity</span>")
        self.graph_1PB_rho_res.setProperty("class", "seebeck")
        self.graph_Xaxis_1PB_rho_res = self.graph_1PB_rho_res.getAxis("bottom")
        self.graph_Yaxis_1PB_rho_res = self.graph_1PB_rho_res.getAxis("left")
        self.graph_Xaxis_1PB_rho_res.setStyle(tickFont=self.tick_font)
        self.graph_Yaxis_1PB_rho_res.setStyle(tickFont=self.tick_font)
        self.graph_Xaxis_1PB_rho_res.setPen(self.pen_axis)
        self.graph_Yaxis_1PB_rho_res.setPen(self.pen_axis)
        self.graph_Xaxis_1PB_rho_res.setTickPen(self.pen_axis)
        self.graph_Yaxis_1PB_rho_res.setTickPen(self.pen_axis)
        self.graph_Xaxis_1PB_rho_res.setStyle(tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_Yaxis_1PB_rho_res.setStyle(tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_Xaxis_1PB_rho_res.setHeight(h=60)
        self.graph_Yaxis_1PB_rho_res.setWidth(w=60)
        self.graph_Xaxis_1PB_rho_res.setLabel('<p style="font-size:18px;color=white">Temperature [K] </p>')
        self.graph_Yaxis_1PB_rho_res.setLabel('<p style="font-size:18px;color=white">Resistivity [\u03BC\u03A9cm] </p>')

        # Plot the resulting curve from the initial parameters
        self.manipulate_plot_1PB_rho_res = pg.PlotCurveItem()
        self.manipulate_plot_1PB_rho_res.setData(x_val_1PB_res, y_val_1PB_res, pen=self.pen2, clear=True)
        # Plot for the resulting curve from the fit (is left empty here and filled in a function down below)
        self.manip_fit_plot_1PB_rho_res = pg.PlotCurveItem()

        self.legend_1PB_rho_res = self.graph_1PB_rho_res.addLegend()
        self.graph_1PB_rho_res.addItem(self.manipulate_plot_1PB_rho_res)
        self.graph_1PB_rho_res.addItem(self.manip_fit_plot_1PB_rho_res)
        self.legend_1PB_rho_res.addItem(self.manipulate_plot_1PB_rho_res, "Manipulate")
        

        # Plot the resulting effective band structure
        self.graph_box_bands_1PB_res = pg.GraphicsLayoutWidget()
        self.graph_bands_1PB_res = self.graph_box_bands_1PB_res.addPlot(2, 3)
        self.graph_box_bands_1PB_res.setStyleSheet(
            "QFrame {border: 5px solid #252530;border-bottom: 5px solid #353540; border-top: 5px solid #151520}")

        # self.graph_3bands = pg.PlotWidget()
        self.graph_bands_1PB_res_Xaxis = self.graph_bands_1PB_res.getAxis("bottom")
        self.graph_bands_1PB_res_Yaxis = self.graph_bands_1PB_res.getAxis("left")
        self.graph_bands_1PB_res_Xaxis.setStyle(tickFont=self.tick_font_3bands)
        self.graph_bands_1PB_res_Yaxis.setStyle(tickFont=self.tick_font_3bands)
        self.graph_bands_1PB_res_Xaxis.setPen(self.pen_axis)
        self.graph_bands_1PB_res_Yaxis.setPen(self.pen_axis)
        self.graph_bands_1PB_res_Xaxis.setTickPen(self.pen_axis_3bands)
        self.graph_bands_1PB_res_Yaxis.setTickPen(self.pen_axis_3bands)
        self.graph_bands_1PB_res_Xaxis.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_bands_1PB_res_Yaxis.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_bands_1PB_res_Xaxis.setHeight(h=60)
        self.graph_bands_1PB_res_Yaxis.setWidth(w=80)
        self.graph_bands_1PB_res_Xaxis.setLabel(
            '<p style="font-size:18px;color=white">k [1/m] </p>')
        self.graph_bands_1PB_res_Yaxis.setLabel(
            '<p style="font-size:18px;color=white">E-E_VB-edge [eV] </p>')

        self.graph_bands_1PB_res.setContentsMargins(2, 0, 5, 10)

        self.band1_1PB_res = pg.PlotCurveItem()
        self.band3_1PB_res = pg.PlotCurveItem()

        self.band1_data_1PB_res = TE.get_parabolic_band(-40, 40, 200, self.mass_slider_1PB_res.value()/100.,0)
        self.band3_data_1PB_res = TE.get_parabolic_band(-40, 40, 200, 100000000, self.fermi_slider_1PB_res.value())

        self.band1_1PB_res.setData(self.band1_data_1PB_res[0], self.band1_data_1PB_res[1], pen=self.pen_band1, clear=True)
        self.band3_1PB_res.setData(self.band3_data_1PB_res[0], self.band3_data_1PB_res[1], pen=self.pen_band3, clear=True)
        
        self.graph_bands_1PB_res.addItem(self.band1_1PB_res)
        self.graph_bands_1PB_res.addItem(self.band3_1PB_res)
        
        self.experimental_data_1PB_res = QRadioButton("Experimental data")
        self.experimental_data_1PB_res.toggled.connect(self.experimental_data_toggled_1PB_res)
        self.experimental_data_1PB_res.setDisabled(True)
        
        self.is_scatter2_there_1PB_res = False
        self.label1PB_res_1 = QLabel('aaaaaa')
        self.label1PB_res_1.setProperty("class", "PCH")
        
        self.adv_opt_button_1PB_res = QPushButton("Advanced options") 
        self.adv_opt_button_1PB_res.clicked.connect(self.adv_opt_button_clicked_1PB_res)
        
        self.ind_cont_button_1PB_res = QPushButton("Additional graphs")
        self.ind_cont_button_1PB_res.clicked.connect(self.ind_cont_button_clicked_1PB_res)
        
        self.print_data_button_1PB_res = QPushButton("Data to file")
        self.print_data_button_1PB_res.setToolTip("Save all visible data to a single file")
        self.print_data_button_1PB_res.clicked.connect(self.print_data_button_clicked_see_res_1PB)
        
        self.print_params_button_1PB_res = QPushButton("Parameters to file")
        self.print_params_button_1PB_res.setToolTip("Save all parameters to a single file")
        self.print_params_button_1PB_res.clicked.connect(self.print_params_button_clicked_see_res_1PB)
        
        #!!! Preliminary_degeneracy value <- not added to GUI yet
        self.Nv_value_1PB_res = QLineEdit('1')
        self.Nv_value_1PB_res.setProperty("class", "slideredit")
        self.Nv_value_1PB_res.setValidator(self.onlyInt)  # assuming onlyInt is a QValidator object defined elsewhere
        self.Nv_value_1PB_res.returnPressed.connect(self.manipulate_value_changed_1PB_res)
        
        # Adding the widgets to the window on their respective grid slots
        self.page_seebeckRho_1PB_layout.addWidget(self.label_p2_res, 0, 0, 1, 3)
        self.page_seebeckRho_1PB_layout.addWidget(self.experimental_data_1PB_res, 0, 3, 1, 3)
        self.page_seebeckRho_1PB_layout.addWidget(self.manip_fit_button_1PB_res, 0, 6, 1, 3)
        self.page_seebeckRho_1PB_layout.addWidget(self.gif_label_spb_res, 0, 7, 1, 1)
        self.page_seebeckRho_1PB_layout.addWidget(self.scattering_type_1PB_res, 0, 9,1,3)
        self.page_seebeckRho_1PB_layout.addWidget(self.label1PB_res_1, 1, 0, 1, 3)
        self.page_seebeckRho_1PB_layout.addWidget(self.mass_slider_1PB_res, 2, 0, 1, 3)
        self.page_seebeckRho_1PB_layout.addWidget(self.mass_SPB_label_res, 3, 0, 1, 2)
        self.page_seebeckRho_1PB_layout.addWidget(self.mass_lineedit_1PB_res, 3, 2, 1, 1)
        self.page_seebeckRho_1PB_layout.addWidget(self.mass_fit_SPB_label_res, 4, 0, 1, 3)
        self.page_seebeckRho_1PB_layout.addWidget(self.mass_fit_SPB_label_PCH_res, 4, 0, 1, 3)
        self.page_seebeckRho_1PB_layout.addWidget(self.fermi_slider_1PB_res, 5, 0, 1, 3)
        self.page_seebeckRho_1PB_layout.addWidget(self.fermi_label_1PB_res, 6, 0, 1, 2)
        self.page_seebeckRho_1PB_layout.addWidget(self.fermi_lineedit_1PB_res, 6, 2, 1, 1)    
        self.page_seebeckRho_1PB_layout.addWidget(self.fermi_fit_label_1PB_res, 7, 0, 1, 3)
        self.page_seebeckRho_1PB_layout.addWidget(self.fermi_fit_label_PCH_1PB_res, 7, 0, 1, 3)
        self.page_seebeckRho_1PB_layout.addWidget(self.para_A_slider_1PB_res, 8, 0, 1, 3)
        self.page_seebeckRho_1PB_layout.addWidget(self.para_A_label_1PB_res, 9, 0, 1, 2)
        self.page_seebeckRho_1PB_layout.addWidget(self.para_A_lineedit_1PB_res, 9, 2, 1, 1)
        self.page_seebeckRho_1PB_layout.addWidget(self.para_A_fit_label_1PB_res, 10, 0, 1, 3)
        self.page_seebeckRho_1PB_layout.addWidget(self.para_A_fit_label_PCH_1PB_res, 10, 4, 1, 3)
        self.page_seebeckRho_1PB_layout.addWidget(self.para_C_slider_1PB_res, 11, 0, 1, 3)
        self.page_seebeckRho_1PB_layout.addWidget(self.para_C_label_1PB_res, 12, 0, 1, 2)
        self.page_seebeckRho_1PB_layout.addWidget(self.para_C_lineedit_1PB_res, 12, 2, 1, 1)
        self.page_seebeckRho_1PB_layout.addWidget(self.para_C_fit_label_1PB_res, 13, 0, 1, 3)
        self.page_seebeckRho_1PB_layout.addWidget(self.para_C_fit_label_PCH_1PB_res, 13, 4, 1, 3)
        self.page_seebeckRho_1PB_layout.addWidget(self.para_F_slider_1PB_res, 8, 0, 1, 3)
        self.page_seebeckRho_1PB_layout.addWidget(self.para_F_label_1PB_res, 9, 0, 1, 2)
        self.page_seebeckRho_1PB_layout.addWidget(self.para_F_lineedit_1PB_res, 9, 2, 1, 1)
        self.page_seebeckRho_1PB_layout.addWidget(self.para_F_fit_label_1PB_res, 10, 0, 1, 3)
        self.page_seebeckRho_1PB_layout.addWidget(self.graph_box_1PB_res, 1, 4, 5, 5)
        self.page_seebeckRho_1PB_layout.addWidget(self.graph_box_1PB_rho_res, 6, 4, 5, 5)
        self.page_seebeckRho_1PB_layout.addWidget(self.graph_box_bands_1PB_res, 1, 9, 5, 5)
        self.page_seebeckRho_1PB_layout.addWidget(self.adv_opt_button_1PB_res, 6, 9, 1, 2)
        self.page_seebeckRho_1PB_layout.addWidget(self.ind_cont_button_1PB_res, 6, 11, 1, 2)      
        self.page_seebeckRho_1PB_layout.addWidget(self.print_data_button_1PB_res, 7, 9, 1, 2)
        self.page_seebeckRho_1PB_layout.addWidget(self.print_params_button_1PB_res, 7, 11, 1, 2)

    def set_up_1PB_hall_window(self):
        self.para_A_lower_limit_1PB_hall = 0
        self.para_A_upper_limit_1PB_hall = 1e6
        self.para_C_lower_limit_1PB_hall = 1e-3
        self.para_C_upper_limit_1PB_hall = 1e3
        self.para_F_lower_limit_1PB_hall = 0
        self.para_F_upper_limit_1PB_hall = 1e5
        self.para_E_lower_limit_1PB_hall = 1e-3
        self.para_E_upper_limit_1PB_hall = 1e3
        
        # Descriptive label on top
        self.label_p2_hall = QLabel("1PB: <i>S</i> + <i>&rho;</i> +  <i>Hall</i>")
        self.label_p2_hall.setProperty("class", "title")

        # Sliders for the 1PB-model parameters   
        self.mass_slider_1PB_hall = QSlider(Qt.Horizontal)
        self.mass_slider_1PB_hall.setMinimum(-300)
        self.mass_slider_1PB_hall.setMaximum(300)
        self.mass_slider_1PB_hall.setSingleStep(1)
        self.mass_slider_1PB_hall.setValue(100)
        self.mass_slider_1PB_hall.valueChanged.connect(self.manipulate_value_changed_1PB_hall)

        self.mass_SPB_label_hall = QLabel('mass =')
        
        self.mass_lineedit_1PB_hall = QLineEdit("1")
        self.mass_lineedit_1PB_hall.setProperty("class", "slideredit")
        self.mass_lineedit_1PB_hall.returnPressed.connect(self.line_edit_changed_1PB_hall)  # Same slot as the original
        self.mass_fit_SPB_label_hall = QLabel('mass = ')
        self.mass_fit_SPB_label_hall.setProperty("class", "fit")
        self.mass_fit_SPB_label_hall.setVisible(False)
        self.mass_fit_SPB_label_PCH_hall = QLabel()
        self.mass_fit_SPB_label_PCH_hall.setProperty("class", "PCH")

        # Position of the Fermi level in Kelvin
        self.fermi_slider_1PB_hall = QSlider(Qt.Horizontal)
        self.fermi_slider_1PB_hall.setMinimum(-3000)
        self.fermi_slider_1PB_hall.setMaximum(3000)
        self.fermi_slider_1PB_hall.setSingleStep(1)
        self.fermi_slider_1PB_hall.setValue(100)
        self.fermi_slider_1PB_hall.valueChanged.connect(self.manipulate_value_changed_1PB_hall)
        
        self.fermi_label_1PB_hall = QLabel('Fermi energy =')
        
        self.fermi_lineedit_1PB_hall = QLineEdit("100")
        self.fermi_lineedit_1PB_hall.setProperty("class", "slideredit")
        self.fermi_lineedit_1PB_hall.returnPressed.connect(self.line_edit_changed_1PB_hall)  # Same slot as the original
        
        self.fermi_fit_label_1PB_hall = QLabel("<p style='font-size: 14px;'><i>E</i><sub>F</sub> =</p>")
        self.fermi_fit_label_1PB_hall.setProperty("class", "fit")
        self.fermi_fit_label_1PB_hall.setVisible(False)
        self.fermi_fit_label_PCH_1PB_hall = QLabel()
        self.fermi_fit_label_PCH_1PB_hall.setProperty("class", "PCH")
        
        self.para_A_slider_1PB_hall = QSlider(Qt.Horizontal)
        self.para_A_slider_1PB_hall.setMinimum(0)
        self.para_A_slider_1PB_hall.setMaximum(int(1e6))
        self.para_A_slider_1PB_hall.setSingleStep(1000)
        self.para_A_slider_1PB_hall.setValue(40000)  # Initial value
        self.para_A_slider_1PB_hall.valueChanged.connect(self.manipulate_value_changed_1PB_hall)  # Assuming you want to use the same slot
        
        self.para_A_label_1PB_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,1</sub> =</p>")
        self.para_A_label_1PB_hall.setProperty("class", "sliderlabel")
        
        self.para_A_lineedit_1PB_hall = QLineEdit("40000")
        self.para_A_lineedit_1PB_hall.setProperty("class", "slideredit")
        self.para_A_lineedit_1PB_hall.returnPressed.connect(self.line_edit_changed_1PB_hall)  # Same slot as the original
        
        self.para_A_fit_label_1PB_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,1</sub> = {}</p>".format(0))
        self.para_A_fit_label_1PB_hall.setProperty("class", "fit")
        self.para_A_fit_label_1PB_hall.setVisible(False)
        
        self.para_A_fit_label_PCH_1PB_hall = QLabel()
        self.para_A_fit_label_PCH_1PB_hall.setProperty("class", "PCH")
        
        self.para_C_slider_1PB_hall = QSlider(Qt.Horizontal)
        self.para_C_slider_1PB_hall.setMinimum(0)
        self.para_C_slider_1PB_hall.setMaximum(int(300))
        self.para_C_slider_1PB_hall.setSingleStep(1)
        self.para_C_slider_1PB_hall.setValue(100)  # Initial value
        self.para_C_slider_1PB_hall.valueChanged.connect(self.manipulate_value_changed_1PB_hall)  # Assuming you want to use the same slot
        
        self.para_C_label_1PB_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,1</sub>/&tau;<sub>dis,1</sub> =</p>")
        self.para_C_label_1PB_hall.setProperty("class", "sliderlabel")
        
        self.para_C_lineedit_1PB_hall = QLineEdit("100")
        self.para_C_lineedit_1PB_hall.setProperty("class", "slideredit")
        self.para_C_lineedit_1PB_hall.returnPressed.connect(self.line_edit_changed_1PB_hall)  # Same slot as the original
        
        self.para_C_fit_label_1PB_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,1</sub>/&tau;<sub>dis,1</sub> = {}</p>".format(0))
        self.para_C_fit_label_1PB_hall.setProperty("class", "fit")
        self.para_C_fit_label_1PB_hall.setVisible(False)
        
        self.para_C_fit_label_PCH_1PB_hall = QLabel()
        self.para_C_fit_label_PCH_1PB_hall.setProperty("class", "PCH")
        
        self.para_C_fit_label_1PB_hall.setVisible(False)
        self.para_C_slider_1PB_hall.setVisible(False)
        self.para_C_label_1PB_hall.setVisible(False)
        self.para_C_lineedit_1PB_hall.setVisible(False)
        
        self.para_F_slider_1PB_hall = QSlider(Qt.Horizontal)
        self.para_F_slider_1PB_hall.setMinimum(0)
        self.para_F_slider_1PB_hall.setMaximum(int(1e5))
        self.para_F_slider_1PB_hall.setSingleStep(100)
        self.para_F_slider_1PB_hall.setValue(4000)  # Initial value
        self.para_F_slider_1PB_hall.valueChanged.connect(self.manipulate_value_changed_1PB_hall)  # Assuming you want to use the same slot
         
        self.para_F_label_1PB_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>dis,1</sub> =</p>")
        self.para_F_label_1PB_hall.setProperty("class", "sliderlabel")
        
        self.para_F_lineedit_1PB_hall = QLineEdit("4000")
        self.para_F_lineedit_1PB_hall.setProperty("class", "slideredit")
        self.para_F_lineedit_1PB_hall.returnPressed.connect(self.line_edit_changed_1PB_hall)  # Same slot as the original
        
        self.para_F_fit_label_1PB_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>dis,1</sub> = {}</p>".format(0))
        self.para_F_fit_label_1PB_hall.setProperty("class", "fit")

        self.para_F_fit_label_1PB_hall.setVisible(False)
        self.para_F_slider_1PB_hall.setVisible(False)
        self.para_F_label_1PB_hall.setVisible(False)
        self.para_F_lineedit_1PB_hall.setVisible(False)
        

        # Execute a fit using the starting parameters from the manipulate button
        self.manip_fit_button_1PB_hall = QPushButton("Fit from manipulate")
        self.manip_fit_button_1PB_hall.clicked.connect(self.manip_fit_button_clicked_1PB_hall)

        #ComboBox to choose the dominant scattering mechanism  
        self.scattering_type_1PB_hall = QComboBox()
        self.scattering_type_1PB_hall.addItem("acPh")
        self.scattering_type_1PB_hall.addItem("dis")
        self.scattering_type_1PB_hall.addItem("acPhDis")
        self.scattering_type_1PB_hall.setCurrentIndex(0)
        self.scattering_type_1PB_hall.currentIndexChanged.connect(self.type_of_fitting_changed_1PB_hall)

        #Calculate data for SPB_hall window
        x_val_1PB, y_val_1PB, y_val_1PB_res, y_val_1PB_hall = TE.spb_see_res_hall_calc(self.minT, self.maxT, 50, self.mass_slider_1PB_hall.value()/100., self.fermi_slider_1PB_hall.value(), [
            self.para_A_slider_1PB_hall.value(), self.para_C_slider_1PB_hall.value()/100.,self.para_F_slider_1PB_hall.value()], f"{self.scattering_type_1PB_hall.currentText()}")
        x_val_1PB_res, x_val_1PB_hall = x_val_1PB, x_val_1PB

        #
        #Graph for Seebeck curve and fit
        #
        self.graph_box_1PB_hall = pg.GraphicsLayoutWidget()
        self.graph_1PB_hall = self.graph_box_1PB_hall.addPlot(2, 3)
        self.graph_box_1PB_hall.setStyleSheet("QFrame {border: 5px solid #252530;border-bottom: 5px solid #353540; border-top: 5px solid #151520}")
        self.legend_1PB_hall = self.graph_1PB_hall.addLegend()
        self.graph_box_1PB_hall.setBackground((5, 5, 5))
        self.graph_1PB_hall.setTitle(
            "<span style=\"color:#AAAAAA;font-size:16pt;font:Lato;font-weight:semi-bold\">Calculated Seebeck coefficient</span>")
        self.graph_1PB_hall.setProperty("class", "seebeck")
        self.graph_Xaxis_1PB_hall = self.graph_1PB_hall.getAxis("bottom")
        self.graph_Yaxis_1PB_hall = self.graph_1PB_hall.getAxis("left")
        self.graph_Xaxis_1PB_hall.setStyle(tickFont=self.tick_font)
        self.graph_Yaxis_1PB_hall.setStyle(tickFont=self.tick_font)
        self.graph_Xaxis_1PB_hall.setPen(self.pen_axis)
        self.graph_Yaxis_1PB_hall.setPen(self.pen_axis)
        self.graph_Xaxis_1PB_hall.setTickPen(self.pen_axis)
        self.graph_Yaxis_1PB_hall.setTickPen(self.pen_axis)
        self.graph_Xaxis_1PB_hall.setStyle(tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_Yaxis_1PB_hall.setStyle(tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_Xaxis_1PB_hall.setHeight(h=60)
        self.graph_Yaxis_1PB_hall.setWidth(w=60)
        self.graph_Xaxis_1PB_hall.setLabel('<p style="font-size:18px;color=white">Temperature [K] </p>')
        self.graph_Yaxis_1PB_hall.setLabel('<p style="font-size:18px;color=white">Seebeck coefficient [\u03BCV/K] </p>')

        # Plot the resulting curve from the initial parameters
        self.manipulate_plot_1PB_hall = pg.PlotCurveItem()
        self.manipulate_plot_1PB_hall.setData(x_val_1PB, y_val_1PB, pen=self.pen2, clear=True)
        self.manip_fit_plot_1PB_hall = pg.PlotCurveItem()
        # Plot for the resulting curve from the fit (is left empty here and filled in a function down below)
        self.legend_1PB_hall = self.graph_1PB_hall.addLegend()
        self.graph_1PB_hall.addItem(self.manipulate_plot_1PB_hall)
        self.graph_1PB_hall.addItem(self.manip_fit_plot_1PB_hall)
        self.legend_1PB_hall.addItem(self.manipulate_plot_1PB_hall, "Manipulate")
        
        #
        #Graph for Resistivity curve and fit
        #
        self.graph_box_1PB_rho_hall = pg.GraphicsLayoutWidget()
        self.graph_1PB_rho_hall = self.graph_box_1PB_rho_hall.addPlot(2, 3)
        self.graph_box_1PB_rho_hall.setStyleSheet("QFrame {border: 5px solid #252530;border-bottom: 5px solid #353540; border-top: 5px solid #151520}")
        self.legend_1PB_rho_hall = self.graph_1PB_rho_hall.addLegend()
        self.graph_box_1PB_rho_hall.setBackground((5, 5, 5))
        self.graph_1PB_rho_hall.setTitle(
            "<span style=\"color:#AAAAAA;font-size:16pt;font:Lato;font-weight:semi-bold\">Calculated Resistivity</span>")
        self.graph_1PB_rho_hall.setProperty("class", "seebeck")
        self.graph_Xaxis_1PB_rho_hall = self.graph_1PB_rho_hall.getAxis("bottom")
        self.graph_Yaxis_1PB_rho_hall = self.graph_1PB_rho_hall.getAxis("left")
        self.graph_Xaxis_1PB_rho_hall.setStyle(tickFont=self.tick_font)
        self.graph_Yaxis_1PB_rho_hall.setStyle(tickFont=self.tick_font)
        self.graph_Xaxis_1PB_rho_hall.setPen(self.pen_axis)
        self.graph_Yaxis_1PB_rho_hall.setPen(self.pen_axis)
        self.graph_Xaxis_1PB_rho_hall.setTickPen(self.pen_axis)
        self.graph_Yaxis_1PB_rho_hall.setTickPen(self.pen_axis)
        self.graph_Xaxis_1PB_rho_hall.setStyle(tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_Yaxis_1PB_rho_hall.setStyle(tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_Xaxis_1PB_rho_hall.setHeight(h=60)
        self.graph_Yaxis_1PB_rho_hall.setWidth(w=60)
        self.graph_Xaxis_1PB_rho_hall.setLabel('<p style="font-size:18px;color=white">Temperature [K] </p>')
        self.graph_Yaxis_1PB_rho_hall.setLabel('<p style="font-size:18px;color=white">Resistivity [\u03BC\u03A9cm] </p>')

        # Plot the resulting curve from the initial parameters
        self.manipulate_plot_1PB_rho_hall = pg.PlotCurveItem()
        self.manipulate_plot_1PB_rho_hall.setData(x_val_1PB_res, y_val_1PB_res, pen=self.pen2, clear=True)
        # Plot for the resulting curve from the fit (is left empty here and filled in a function down below)
        self.manip_fit_plot_1PB_rho_hall = pg.PlotCurveItem()

        self.legend_1PB_rho_hall = self.graph_1PB_rho_hall.addLegend()
        self.graph_1PB_rho_hall.addItem(self.manipulate_plot_1PB_rho_hall)
        self.graph_1PB_rho_hall.addItem(self.manip_fit_plot_1PB_rho_hall)
        self.legend_1PB_rho_hall.addItem(self.manipulate_plot_1PB_rho_hall, "Manipulate")
        
        #
        #Graph for Hall curve and fit
        #
        self.graph_box_1PB_Hall_hall = pg.GraphicsLayoutWidget()
        self.graph_1PB_Hall_hall = self.graph_box_1PB_Hall_hall.addPlot(2, 3)
        self.graph_box_1PB_Hall_hall.setStyleSheet("QFrame {border: 5px solid #252530;border-bottom: 5px solid #353540; border-top: 5px solid #151520}")
        self.legend_1PB_Hall_hall = self.graph_1PB_Hall_hall.addLegend()
        self.graph_box_1PB_Hall_hall.setBackground((5, 5, 5))
        self.graph_1PB_Hall_hall.setTitle(
            "<span style=\"color:#AAAAAA;font-size:16pt;font:Lato;font-weight:semi-bold\">Calculated Hall coefficient</span>")
        self.graph_1PB_Hall_hall.setProperty("class", "seebeck")
        self.graph_Xaxis_1PB_Hall_hall = self.graph_1PB_Hall_hall.getAxis("bottom")
        self.graph_Yaxis_1PB_Hall_hall = self.graph_1PB_Hall_hall.getAxis("left")
        self.graph_Xaxis_1PB_Hall_hall.setStyle(tickFont=self.tick_font)
        self.graph_Yaxis_1PB_Hall_hall.setStyle(tickFont=self.tick_font)
        self.graph_Xaxis_1PB_Hall_hall.setPen(self.pen_axis)
        self.graph_Yaxis_1PB_Hall_hall.setPen(self.pen_axis)
        self.graph_Xaxis_1PB_Hall_hall.setTickPen(self.pen_axis)
        self.graph_Yaxis_1PB_Hall_hall.setTickPen(self.pen_axis)
        self.graph_Xaxis_1PB_Hall_hall.setStyle(tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_Yaxis_1PB_Hall_hall.setStyle(tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_Xaxis_1PB_Hall_hall.setHeight(h=60)
        self.graph_Yaxis_1PB_Hall_hall.setWidth(w=60)
        self.graph_Xaxis_1PB_Hall_hall.setLabel('<p style="font-size:18px;color=white">Temperature [K] </p>')
        self.graph_Yaxis_1PB_Hall_hall.setLabel('<p style="font-size:18px;color=white">Hall coefficient [m<sup>3</sup>/A<sup>-1</sup>s<sup>-1</sup>] </p>')

        # Plot the resulting curve from the initial parameters
        self.manipulate_plot_1PB_Hall_hall = pg.PlotCurveItem()
        self.manipulate_plot_1PB_Hall_hall.setData(x_val_1PB_hall, y_val_1PB_hall, pen=self.pen2, clear=True)
        # Plot for the resulting curve from the fit (is left empty here and filled in a function down below)
        self.manip_fit_plot_1PB_Hall_hall = pg.PlotCurveItem()

        self.legend_1PB_Hall_hall = self.graph_1PB_Hall_hall.addLegend()
        self.graph_1PB_Hall_hall.addItem(self.manipulate_plot_1PB_Hall_hall)
        self.graph_1PB_Hall_hall.addItem(self.manip_fit_plot_1PB_Hall_hall)
        self.legend_1PB_Hall_hall.addItem(self.manipulate_plot_1PB_Hall_hall, "Manipulate")
        

        # Plot the resulting effective band structure
        self.graph_box_bands_1PB_hall = pg.GraphicsLayoutWidget()
        self.graph_bands_1PB_hall = self.graph_box_bands_1PB_hall.addPlot(2, 3)
        self.graph_box_bands_1PB_hall.setStyleSheet(
            "QFrame {border: 5px solid #252530;border-bottom: 5px solid #353540; border-top: 5px solid #151520}")

        # self.graph_3bands = pg.PlotWidget()
        self.graph_bands_1PB_hall_Xaxis = self.graph_bands_1PB_hall.getAxis("bottom")
        self.graph_bands_1PB_hall_Yaxis = self.graph_bands_1PB_hall.getAxis("left")
        self.graph_bands_1PB_hall_Xaxis.setStyle(tickFont=self.tick_font_3bands)
        self.graph_bands_1PB_hall_Yaxis.setStyle(tickFont=self.tick_font_3bands)
        self.graph_bands_1PB_hall_Xaxis.setPen(self.pen_axis)
        self.graph_bands_1PB_hall_Yaxis.setPen(self.pen_axis)
        self.graph_bands_1PB_hall_Xaxis.setTickPen(self.pen_axis_3bands)
        self.graph_bands_1PB_hall_Yaxis.setTickPen(self.pen_axis_3bands)
        self.graph_bands_1PB_hall_Xaxis.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_bands_1PB_hall_Yaxis.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_bands_1PB_hall_Xaxis.setHeight(h=60)
        self.graph_bands_1PB_hall_Yaxis.setWidth(w=80)
        self.graph_bands_1PB_hall_Xaxis.setLabel(
            '<p style="font-size:18px;color=white">k [1/m] </p>')
        self.graph_bands_1PB_hall_Yaxis.setLabel(
            '<p style="font-size:18px;color=white">E-E_VB-edge [eV] </p>')

        self.graph_bands_1PB_hall.setContentsMargins(2, 0, 5, 10)

        # self.graph_bands = pg.PlotWidget()
        self.band1_1PB_hall = pg.PlotCurveItem()
        self.band3_1PB_hall = pg.PlotCurveItem()

        self.band1_data_1PB_hall = TE.get_parabolic_band(-40, 40, 200, self.mass_slider_1PB_hall.value()/100.,0)
        self.band3_data_1PB_hall = TE.get_parabolic_band(-40, 40, 200, 100000000, self.fermi_slider_1PB_hall.value())

        self.band1_1PB_hall.setData(self.band1_data_1PB_hall[0], self.band1_data_1PB_hall[1], pen=self.pen_band1, clear=True)
        self.band3_1PB_hall.setData(self.band3_data_1PB_hall[0], self.band3_data_1PB_hall[1], pen=self.pen_band3, clear=True)
        
        self.graph_bands_1PB_hall.addItem(self.band1_1PB_hall)
        self.graph_bands_1PB_hall.addItem(self.band3_1PB_hall)
        
        self.experimental_data_1PB_hall = QRadioButton("Experimental data")
        self.experimental_data_1PB_hall.toggled.connect(self.experimental_data_toggled_1PB_hall)
        self.experimental_data_1PB_hall.setDisabled(True)
        
        self.is_scatter2_there_1PB_hall = False
        
        
        self.label1PB_hall_1 = QLabel('aaaaaa')
        self.label1PB_hall_1.setProperty("class", "PCH")
        
        self.adv_opt_button_1PB_hall = QPushButton("Advanced options") 
        self.adv_opt_button_1PB_hall.clicked.connect(self.adv_opt_button_clicked_1PB_hall)
        
        self.ind_cont_button_1PB_hall = QPushButton("Additional graphs")
        self.ind_cont_button_1PB_hall.clicked.connect(self.ind_cont_button_clicked_1PB_hall)
        
        self.print_data_button_1PB_hall = QPushButton("Data to file")
        self.print_data_button_1PB_hall.setToolTip("Save all visible data to a single file")
        self.print_data_button_1PB_hall.clicked.connect(self.print_data_button_clicked_see_res_hall_1PB)
        
        self.print_params_button_1PB_hall = QPushButton("Parameters to file")
        self.print_params_button_1PB_hall.setToolTip("Save all parameters to a single file")
        self.print_params_button_1PB_hall.clicked.connect(self.print_params_button_clicked_see_res_hall_1PB)
        
        #!!! Preliminary_degeneracy value <- not added to GUI yet
        self.Nv_value_1PB_hall = QLineEdit('1')
        self.Nv_value_1PB_hall.setProperty("class", "slideredit")
        self.Nv_value_1PB_hall.setValidator(self.onlyInt)  # assuming onlyInt is a QValidator object defined elsewhere
        self.Nv_value_1PB_hall.returnPressed.connect(self.manipulate_value_changed_1PB_hall)

        # Adding the widgets to the window on their respective grid slots
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.label_p2_hall, 0, 0, 1, 3)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.experimental_data_1PB_hall, 0, 3, 1, 3)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.manip_fit_button_1PB_hall, 0, 6, 1, 3)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.gif_label_spb_hall, 0, 7, 1, 1)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.scattering_type_1PB_hall, 0, 9,1,3)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.label1PB_hall_1, 1, 0, 1, 3)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.mass_slider_1PB_hall, 2, 0, 1, 3)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.mass_SPB_label_hall, 3, 0, 1, 2)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.mass_lineedit_1PB_hall, 3, 2, 1, 1)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.mass_fit_SPB_label_hall, 4, 0, 1, 3)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.mass_fit_SPB_label_PCH_hall, 4, 0, 1, 3)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.fermi_slider_1PB_hall, 5, 0, 1, 3)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.fermi_label_1PB_hall, 6, 0, 1, 2)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.fermi_lineedit_1PB_hall, 6, 2, 1, 1)    
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.fermi_fit_label_1PB_hall, 7, 0, 1, 3)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.fermi_fit_label_PCH_1PB_hall, 7, 0, 1, 3)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.para_A_slider_1PB_hall, 8, 0, 1, 3)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.para_A_label_1PB_hall, 9, 0, 1, 2)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.para_A_lineedit_1PB_hall, 9, 2, 1, 1)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.para_A_fit_label_1PB_hall, 10, 0, 1, 3)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.para_A_fit_label_PCH_1PB_hall, 10, 4, 1, 3)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.para_C_slider_1PB_hall, 11, 0, 1, 3)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.para_C_label_1PB_hall, 12, 0, 1, 2)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.para_C_lineedit_1PB_hall, 12, 2, 1, 1)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.para_C_fit_label_1PB_hall, 13, 0, 1, 3)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.para_C_fit_label_PCH_1PB_hall, 13, 4, 1, 3)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.para_F_slider_1PB_hall, 8, 0, 1, 3)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.para_F_label_1PB_hall, 9, 0, 1, 2)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.para_F_lineedit_1PB_hall, 9, 2, 1, 1)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.para_F_fit_label_1PB_hall, 10, 0, 1, 3)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.graph_box_1PB_hall, 1, 4, 5, 5)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.graph_box_1PB_rho_hall, 6, 4, 5, 5)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.graph_box_1PB_Hall_hall, 6, 9, 5, 5)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.graph_box_bands_1PB_hall, 1, 9, 5, 5)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.adv_opt_button_1PB_hall, 12, 9, 1, 2)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.ind_cont_button_1PB_hall, 12, 11, 1, 2)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.print_data_button_1PB_hall, 13, 9, 1, 2)
        self.page_seebeckRhoHall_1PB_layout.addWidget(self.print_params_button_1PB_hall, 13, 11, 1, 2)

    def set_up_2PB_see_window(self):
        # Descriptive label on top
        self.label_p2 = QLabel("2PB: <i>S</i>")
        self.label_p2.setProperty("class", "title")

        # Sliders for the 2PB-model parameters
        self.label1 = QLabel('aaaaaa')
        self.label2 = QLabel('aaaaaa')
        self.label3 = QLabel('aaaaaa')

        self.label1.setProperty("class", "PCH")
        self.label2.setProperty("class", "PCH")
        self.label3.setProperty("class", "PCH")

        # band mass
        # Actually the relation of mass of band1 (m1) and (m2): m2/m1; not an absolute value!
        # However, m1 is set to 1 -> the value directly gives m2 with respect to m1
        self.mass_slider = QSlider(Qt.Horizontal)
        self.mass_slider.setMinimum(-300)
        self.mass_slider.setMaximum(300)
        self.mass_slider.setSingleStep(1)
        self.mass_slider.setValue(101)
        self.mass_slider.valueChanged.connect(self.manipulate_value_changed_2PB_see)

        self.mass_label = QLabel("<p style='font-size: 14px;'>m<sub>2</sub>/m<sub>1</sub> =</p>")
        self.mass_label.setProperty("class", "sliderlabel")
        self.mass_lineedit = QLineEdit("1.01")
        self.mass_lineedit.setProperty("class", "slideredit")
        self.mass_lineedit.returnPressed.connect(self.line_edit_changed) 

        self.mass_fit_label = QLabel("<p style='font-size: 14px;'>m<sub>2</sub>/m<sub>1</sub> = {}</p>".format(0))
        self.mass_fit_label.setProperty("class", "fit")
        self.mass_fit_label.setVisible(False)
        self.mass_fit_label_PCH = QLabel()
        self.mass_fit_label_PCH.setProperty("class", "PCH")

        # band gap
        # Actually the position of the second band with respect to the first band in Kelvin
        # However, the first bands position is set to 0, therefore this value directly gives the distance of the two bands -> band gap
        self.bandgap_slider = QSlider(Qt.Horizontal)
        self.bandgap_slider.setMinimum(-3000)
        self.bandgap_slider.setMaximum(3000)
        self.bandgap_slider.setSingleStep(10)
        self.bandgap_slider.setValue(1000)
        self.bandgap_slider.valueChanged.connect(self.manipulate_value_changed_2PB_see)

        self.bandgap_label = QLabel("gap =")
        self.bandgap_label.setProperty("class", "sliderlabel")
        self.bandgap_lineedit = QLineEdit("1000")
        self.bandgap_lineedit.setProperty("class", "slideredit")
        self.bandgap_lineedit.returnPressed.connect(self.line_edit_changed) 
        
        self.bandgap_fit_label = QLabel('gap = {}'.format(0))
        self.bandgap_fit_label.setProperty("class", "fit")
        self.bandgap_fit_label.setVisible(False)
        self.bandgap_fit_label_PCH = QLabel()
        self.bandgap_fit_label_PCH.setProperty("class", "PCH")

        # Position of the Fermi level in Kelvin
        self.fermi_slider = QSlider(Qt.Horizontal)
        self.fermi_slider.setMinimum(-3000)
        self.fermi_slider.setMaximum(3000)
        self.fermi_slider.setSingleStep(1)
        self.fermi_slider.setValue(-100)
        self.fermi_slider.valueChanged.connect(self.manipulate_value_changed_2PB_see)
        
        self.fermi_label = QLabel("<p style='font-size: 14px;'><i>E</i><sub>F</sub> =</p>")
        self.fermi_label.setProperty("class", "sliderlabel")
        self.fermi_lineedit = QLineEdit("-100")
        self.fermi_lineedit.setProperty("class", "slideredit")
        self.fermi_lineedit.returnPressed.connect(self.line_edit_changed) 
        
        self.fermi_fit_label = QLabel("<p style='font-size: 14px;'><i>E</i><sub>F</sub> = {}</p>".format(0))
        self.fermi_fit_label.setProperty("class", "fit")
        self.fermi_fit_label.setVisible(False)
        self.fermi_fit_label_PCH = QLabel()
        self.fermi_fit_label_PCH.setProperty("class", "PCH")

        # Jo labels for 2PB window
        # label for Nv1
        # creates a label and defines the displayed text
        self.Nv1_label = QLabel('deg 1')
        self.Nv1_label.setProperty("class", "sliderlabel")
        self.Nv1_label.setToolTip("Degeneracy of band 1")

        # textbox to enter Nv1
        self.Nv1_value = QLineEdit()  # creates and editable text window
        self.Nv1_value.setText('1')  # why does this not work?
        self.Nv1_value.setProperty("class", "slideredit")
        # makes sure that only integers can be entered
        self.Nv1_value.setValidator(self.onlyInt)
        # label for Nv2
        self.Nv2_label = QLabel('deg 2')
        self.Nv2_label.setProperty("class", "sliderlabel")
        self.Nv2_label.setToolTip("Degeneracy of band 2")
        # textbox to enter Nv2
        self.Nv2_value = QLineEdit()
        self.Nv2_value.setText('1')
        self.Nv2_value.setProperty("class", "slideredit")
        self.Nv1_value.returnPressed.connect(self.manipulate_value_changed_2PB_see)
        self.Nv2_value.returnPressed.connect(self.manipulate_value_changed_2PB_see)
        
        self.deg_PCH = QLabel()
        self.deg_PCH.setProperty("class", "PCH")

        # Smooth plot button to show a finer interpolation of the Manipulate plot (more points)
        self.plot_button = QPushButton("Smooth plot")
        self.plot_button.clicked.connect(self.plot_button_clicked)

        # Execute a fit using the starting parameters from the manipulate button
        self.manip_fit_button = QPushButton("Fit from manipulate")
        self.manip_fit_button.clicked.connect(self.manip_fit_button_clicked_2PB_see)

        # Activate or deactivate the plotting of the experimental data)
        self.experimental_data = QRadioButton("Experimental data")
        self.experimental_data.toggled.connect(self.experimental_data_toggled)
        self.experimental_data.setDisabled(True)

        # Manipulate plot (dependent on the 2PB-model parameters)

        self.graph_box2 = pg.GraphicsLayoutWidget()
        self.graph2 = self.graph_box2.addPlot(2, 3)
        self.graph_box2.setStyleSheet("QFrame {border: 5px solid #252530;border-bottom: 5px solid #353540; border-top: 5px solid #151520}")

        self.graph_box2.setBackground((5, 5, 5))
        self.graph2.setTitle("<span style=\"color:#AAAAAA;font-size:16pt;font:Lato;font-weight:semi-bold\">Calculated Seebeck coefficient</span>")
        self.graph2.setProperty("class", "seebeck")
        
        # Plot of the experimental data that can be imported
        self.is_scatter2_there = False

        # Plot the resulting curve from the initial parameters 
        x_val, y_val = TE.dpb_see_calc(self.minT, self.maxT, 50, [self.mass_slider.value(
        )/100., self.bandgap_slider.value(), self.fermi_slider.value()], float(self.Nv2_value.text())/float(self.Nv1_value.text()))
      
        self.manipulate_plot = pg.PlotCurveItem()
        self.manipulate_plot.setData(x_val, y_val, pen=self.pen2, clear=True)

        # Plot for the resulting curve from the fit (is left empty here and filled in a function down below)
        self.manip_fit_plot = pg.PlotCurveItem()

        self.legend_graph2 = self.graph2.addLegend()
        self.graph2.addItem(self.manipulate_plot)
        self.graph2.addItem(self.manip_fit_plot)

        self.tick_font = QFont()
        self.tick_font.setPixelSize(16)
        self.pen_axis = pg.mkPen("#AAAAAA", width=2, style=Qt.SolidLine)

        self.graph2_Xaxis = self.graph2.getAxis("bottom")
        self.graph2_Yaxis = self.graph2.getAxis("left")
        self.graph2_Xaxis.setStyle(tickFont=self.tick_font)
        self.graph2_Yaxis.setStyle(tickFont=self.tick_font)
        self.graph2_Xaxis.setPen(self.pen_axis)
        self.graph2_Yaxis.setPen(self.pen_axis)
        self.graph2_Xaxis.setTickPen(self.pen_axis)
        self.graph2_Yaxis.setTickPen(self.pen_axis)
        self.graph2_Xaxis.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph2_Yaxis.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph2_Xaxis.setHeight(h=60)
        self.graph2_Yaxis.setWidth(w=60)
        self.graph2_Xaxis.setLabel(
            '<p style="font-size:18px;color=white">Temperature [K] </p>')
        self.graph2_Yaxis.setLabel(
            '<p style="font-size:18px;color=white">Seebeck coefficient [\u03BCV/K] </p>')
        self.legend_graph2.addItem(self.manipulate_plot, '<p style="font-size:16px;color=white">Manipulate </p>')
        self.graph2.setContentsMargins(2, 0, 5, 10)

        # variable to check if label for "Fitting" result already exists; is used again in manip_fit_button_clicked()
        self.label_here = True
        self.label_here_res = True
        self.label_here_hall = True

        # Plot the resulting effective band structure
        self.graph_3bands_box = pg.GraphicsLayoutWidget()
        self.graph_3bands = self.graph_3bands_box.addPlot(2, 3)
        self.graph_3bands_box.setStyleSheet(
            "QFrame {border: 5px solid #252530;border-bottom: 5px solid #353540; border-top: 5px solid #151520}")

        # self.graph_3bands = pg.PlotWidget()
        self.graph_3bands_item = pg.PlotCurveItem()

        self.tick_font_3bands = QFont()
        self.tick_font_3bands.setPixelSize(14)
        self.pen_axis_3bands = pg.mkPen(
            "#AAAAAA", width=1.5, style=Qt.SolidLine)

        self.graph_3bands_Xaxis = self.graph_3bands.getAxis("bottom")
        self.graph_3bands_Yaxis = self.graph_3bands.getAxis("left")
        self.graph_3bands_Xaxis.setStyle(tickFont=self.tick_font_3bands)
        self.graph_3bands_Yaxis.setStyle(tickFont=self.tick_font_3bands)
        self.graph_3bands_Xaxis.setPen(self.pen_axis)
        self.graph_3bands_Yaxis.setPen(self.pen_axis)
        self.graph_3bands_Xaxis.setTickPen(self.pen_axis_3bands)
        self.graph_3bands_Yaxis.setTickPen(self.pen_axis_3bands)
        self.graph_3bands_Xaxis.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_3bands_Yaxis.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_3bands_Xaxis.setHeight(h=60)
        self.graph_3bands_Yaxis.setWidth(w=80)
        self.graph_3bands_Xaxis.setLabel(
            '<p style="font-size:18px;color=white">k [1/m] </p>')
        self.graph_3bands_Yaxis.setLabel(
            '<p style="font-size:18px;color=white">E-E_VB-edge [eV] </p>')
        self.graph_3bands.setContentsMargins(2, 0, 5, 10)

        # self.graph_bands = pg.PlotWidget()
        self.band1 = pg.PlotCurveItem()
        self.band2 = pg.PlotCurveItem()
        self.band3 = pg.PlotCurveItem()

        self.band1_data = TE.get_parabolic_band(-40, 40, 200, -1, 0)
        self.band2_data = TE.get_parabolic_band(
            -40, 40, 200, self.mass_slider.value()/100., self.bandgap_slider.value())
        self.band3_data = TE.get_parabolic_band(
            -40, 40, 200, 100000000, self.fermi_slider.value())

        self.band1.setData(
            self.band1_data[0], self.band1_data[1], pen=self.pen_band1, clear=True)
        self.band2.setData(
            self.band2_data[0], self.band2_data[1], pen=self.pen_band2, clear=True)
        self.band3.setData(
            self.band3_data[0], self.band3_data[1], pen=self.pen_band3, clear=True)
        self.graph_3bands.addItem(self.band1)
        self.graph_3bands.addItem(self.band2)
        self.graph_3bands.addItem(self.band3)

        self.fitpar_button = QPushButton("Adjust to fit")
        self.fitpar_button.clicked.connect(self.fitpar_button_clicked)
        self.fitpar_button.setEnabled(False)
        
        self.adv_opt_button = QPushButton("Advanced options") 
        self.adv_opt_button.clicked.connect(self.adv_opt_button_clicked)
        
        self.ind_cont_button = QPushButton("Additional graphs") 
        self.ind_cont_button.clicked.connect(self.ind_cont_button_clicked)
        
        self.print_data_button_2PB = QPushButton("Data to file")
        self.print_data_button_2PB.setToolTip("Save all visible data to a single file")
        self.print_data_button_2PB.clicked.connect(self.print_data_button_clicked_see_2PB)
        
        self.print_params_button_2PB = QPushButton("Parameters to file")
        self.print_params_button_2PB.setToolTip("Save all parameters to a single file")
        self.print_params_button_2PB.clicked.connect(self.print_params_button_clicked_see_2PB)
        
        # Adding the widgets to the window on their respective grid slots
        self.page_seebeckOnly_2PB_layout.addWidget(self.label_p2, 0, 0, 1 ,3)
        self.page_seebeckOnly_2PB_layout.addWidget(self.experimental_data, 0, 3, 1 ,3)
        self.page_seebeckOnly_2PB_layout.addWidget(self.manip_fit_button, 0, 6, 1 ,3)
        self.page_seebeckOnly_2PB_layout.addWidget(self.plot_button, 0, 9, 1 ,3)
        self.page_seebeckOnly_2PB_layout.addWidget(self.gif_label_dpb_see, 0, 12, 1 ,1)
        self.page_seebeckOnly_2PB_layout.addWidget(self.label2, 0, 13, 1 ,2)
        self.page_seebeckOnly_2PB_layout.addWidget(self.label3, 0, 15, 1 ,3)
        self.page_seebeckOnly_2PB_layout.addWidget(self.Nv1_label, 1, 0, 1, 2)
        self.page_seebeckOnly_2PB_layout.addWidget(self.Nv1_value, 1, 2, 1, 1)
        self.page_seebeckOnly_2PB_layout.addWidget(self.Nv2_label, 2, 0, 1, 2)
        self.page_seebeckOnly_2PB_layout.addWidget(self.Nv2_value, 2, 2, 1, 1)
        self.page_seebeckOnly_2PB_layout.addWidget(self.deg_PCH, 3, 0, 1 ,3)
        self.page_seebeckOnly_2PB_layout.addWidget(self.mass_slider, 4, 0, 1 ,3)
        self.page_seebeckOnly_2PB_layout.addWidget(self.mass_label, 5, 0, 1 ,2)
        self.page_seebeckOnly_2PB_layout.addWidget(self.mass_lineedit, 5, 2, 1 ,1)
        self.page_seebeckOnly_2PB_layout.addWidget(self.mass_fit_label, 6, 0, 1 ,3)
        self.page_seebeckOnly_2PB_layout.addWidget(self.mass_fit_label_PCH, 6, 0, 1 ,3)
        self.page_seebeckOnly_2PB_layout.addWidget(self.bandgap_slider, 7, 0, 1 ,3)
        self.page_seebeckOnly_2PB_layout.addWidget(self.bandgap_label, 8, 0, 1 ,2)
        self.page_seebeckOnly_2PB_layout.addWidget(self.bandgap_lineedit, 8, 2, 1 ,1)
        self.page_seebeckOnly_2PB_layout.addWidget(self.bandgap_fit_label, 9, 0, 1 ,3)
        self.page_seebeckOnly_2PB_layout.addWidget(self.bandgap_fit_label_PCH, 9, 0, 1 ,3)
        self.page_seebeckOnly_2PB_layout.addWidget(self.fermi_slider, 10, 0, 1 ,3)
        self.page_seebeckOnly_2PB_layout.addWidget(self.fermi_label, 11, 0, 1 ,2)
        self.page_seebeckOnly_2PB_layout.addWidget(self.fermi_lineedit, 11, 2, 1 ,1)
        self.page_seebeckOnly_2PB_layout.addWidget(self.fermi_fit_label, 12, 0, 1 ,3)
        self.page_seebeckOnly_2PB_layout.addWidget(self.fermi_fit_label_PCH, 12, 0, 1 ,3)
        # self.page_seebeckOnly_2PB_layout.addWidget(self.label3, 15, 15, 1 ,3)
        self.page_seebeckOnly_2PB_layout.addWidget(self.graph_box2, 1, 3, 13, 15)
        self.page_seebeckOnly_2PB_layout.addWidget(self.graph_3bands_box, 1, 18, 8, 5)
        self.page_seebeckOnly_2PB_layout.addWidget(self.fitpar_button, 13, 0, 1 ,3)
        self.page_seebeckOnly_2PB_layout.addWidget(self.adv_opt_button, 9, 18, 1, 2)
        self.page_seebeckOnly_2PB_layout.addWidget(self.ind_cont_button, 9, 20, 1, 2)
        self.page_seebeckOnly_2PB_layout.addWidget(self.print_data_button_2PB, 10, 18, 1, 2)
        self.page_seebeckOnly_2PB_layout.addWidget(self.print_params_button_2PB, 10, 20, 1, 2)

    def set_up_2PB_res_window(self):
        #predefined limits for parameters A-D
        self.para_A_lower_limit_res = 0
        self.para_B_lower_limit_res = 0
        self.para_C_lower_limit_res = 0
        self.para_D_lower_limit_res = 0
        self.para_F_lower_limit_res = 0
        self.para_G_lower_limit_res = 0
        
        self.para_A_upper_limit_res = 1e12
        self.para_B_upper_limit_res = 100
        self.para_C_upper_limit_res = 100
        self.para_D_upper_limit_res = 100
        self.para_F_upper_limit_res = 1e10
        self.para_G_upper_limit_res = 100
    
# Descriptive label on top
        self.label_p2_res = QLabel("2PB: <i>S</i> + <i>&rho;</i>")
        self.label_p2_res.setProperty("class", "title")

        
        # Labels
        self.label1_res = QLabel('aaaaaa')
        self.label2_res = QLabel('aaaaaa')
        self.label3_res = QLabel('aaaaaa')
        self.label1_res.setProperty("class", "PCH")
        self.label2_res.setProperty("class", "PCH")
        self.label3_res.setProperty("class", "PCH")
        
        self.label4_res = QLabel('aaaaaa')
        self.label5_res = QLabel('aaaaaa')
        self.label6_res = QLabel('aaaaaa')
        self.label4_res.setProperty("class", "PCH")
        self.label5_res.setProperty("class", "PCH")
        self.label6_res.setProperty("class", "PCH")
        
        # Band mass slider and related widgets
        self.mass_slider_res = QSlider(Qt.Horizontal)
        self.mass_slider_res.setMinimum(-300)
        self.mass_slider_res.setMaximum(300)
        self.mass_slider_res.setSingleStep(1)
        self.mass_slider_res.setValue(101)  # Initial value
        self.mass_slider_res.valueChanged.connect(self.manipulate_value_changed_res)  # Assuming you want to use the same slot
        
        self.mass_label_res = QLabel("<p style='font-size: 14px;'>m<sub>2</sub>/m<sub>1</sub> =</p>")
        self.mass_label_res.setProperty("class", "sliderlabel")
        
        self.mass_lineedit_res = QLineEdit("1.01")
        self.mass_lineedit_res.setProperty("class", "slideredit")
        self.mass_lineedit_res.returnPressed.connect(self.line_edit_changed_res)  # Same slot as the original
        
        self.mass_fit_label_res = QLabel("<p style='font-size: 14px;'>m<sub>2</sub>/m<sub>1</sub> = {}</p>".format(0))
        self.mass_fit_label_res.setProperty("class", "fit")
        self.mass_fit_label_res.setVisible(False)
        
        self.mass_fit_label_PCH_res = QLabel()
        self.mass_fit_label_PCH_res.setProperty("class", "PCH")
        
        # Band gap slider and related widgets
        self.bandgap_slider_res = QSlider(Qt.Horizontal)
        self.bandgap_slider_res.setMinimum(-3000)
        self.bandgap_slider_res.setMaximum(3000)
        self.bandgap_slider_res.setSingleStep(10)
        self.bandgap_slider_res.setValue(1000)
        self.bandgap_slider_res.valueChanged.connect(self.manipulate_value_changed_res)
        
        self.bandgap_label_res = QLabel("gap =")
        self.bandgap_label_res.setProperty("class", "sliderlabel")
        
        self.bandgap_lineedit_res = QLineEdit("1000")
        self.bandgap_lineedit_res.setProperty("class", "slideredit")
        self.bandgap_lineedit_res.returnPressed.connect(self.line_edit_changed_res)
        
        self.bandgap_fit_label_res = QLabel('gap = {}'.format(0))
        self.bandgap_fit_label_res.setProperty("class", "fit")
        self.bandgap_fit_label_res.setVisible(False)
        
        self.bandgap_fit_label_PCH_res = QLabel()
        self.bandgap_fit_label_PCH_res.setProperty("class", "PCH")
        
        # Fermi level slider and related widgets
        self.fermi_slider_res = QSlider(Qt.Horizontal)
        self.fermi_slider_res.setMinimum(-3000)
        self.fermi_slider_res.setMaximum(3000)
        self.fermi_slider_res.setSingleStep(1)
        self.fermi_slider_res.setValue(-100)
        self.fermi_slider_res.valueChanged.connect(self.manipulate_value_changed_res)
        
        self.fermi_label_res = QLabel("<p style='font-size: 14px;'><i>E</i><sub>F</sub> =</p>")
        self.fermi_label_res.setProperty("class", "sliderlabel")
        
        self.fermi_lineedit_res = QLineEdit("-100")
        self.fermi_lineedit_res.setProperty("class", "slideredit")
        self.fermi_lineedit_res.returnPressed.connect(self.line_edit_changed_res)
        
        self.fermi_fit_label_res = QLabel("<p style='font-size: 14px;'><i>E</i><sub>F</sub> = {}</p>".format(0))
        self.fermi_fit_label_res.setProperty("class", "fit")
        self.fermi_fit_label_res.setVisible(False)
        
        self.fermi_fit_label_PCH_res = QLabel()
        self.fermi_fit_label_PCH_res.setProperty("class", "PCH")
        
        self.para_A_slider_res = QSlider(Qt.Horizontal)
        self.para_A_slider_res.setMinimum(0)
        self.para_A_slider_res.setMaximum(int(1e6))
        self.para_A_slider_res.setSingleStep(1000)
        self.para_A_slider_res.setValue(40000)  # Initial value
        self.para_A_slider_res.valueChanged.connect(self.manipulate_value_changed_res)  # Assuming you want to use the same slot
        
        self.para_A_label_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,1</sub> =</p>")
        self.para_A_label_res.setProperty("class", "sliderlabel")
        
        self.para_A_lineedit_res = QLineEdit("40000")
        self.para_A_lineedit_res.setProperty("class", "slideredit")
        self.para_A_lineedit_res.returnPressed.connect(self.line_edit_changed_res)  # Same slot as the original
        
        self.para_A_fit_label_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,1</sub> = {}</p>".format(0))
        self.para_A_fit_label_res.setProperty("class", "fit")
        self.para_A_fit_label_res.setVisible(True)
        
        self.para_A_fit_label_PCH_res = QLabel()
        self.para_A_fit_label_PCH_res.setProperty("class", "PCH")
        
        self.para_B_slider_res = QSlider(Qt.Horizontal)
        self.para_B_slider_res.setMinimum(0)
        self.para_B_slider_res.setMaximum(3000)
        self.para_B_slider_res.setSingleStep(1)
        self.para_B_slider_res.setValue(100)  # Initial value
        self.para_B_slider_res.valueChanged.connect(self.manipulate_value_changed_res)  # Assuming you want to use the same slot
        
        self.para_B_label_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,2</sub>/&tau;<sub>ph,1</sub> =</p>")
        self.para_B_label_res.setProperty("class", "sliderlabel")
        
        self.para_B_lineedit_res = QLineEdit("1.0")
        self.para_B_lineedit_res.setProperty("class", "slideredit")
        self.para_B_lineedit_res.returnPressed.connect(self.line_edit_changed_res)  # Same slot as the original
        
        self.para_B_fit_label_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,2</sub>/&tau;<sub>ph,1</sub> = {}</p>".format(0))
        self.para_B_fit_label_res.setProperty("class", "fit")
        self.para_B_fit_label_res.setVisible(False)
        
        self.para_B_fit_label_PCH_res = QLabel()
        self.para_B_fit_label_PCH_res.setProperty("class", "PCH")
        
        self.para_C_slider_res = QSlider(Qt.Horizontal)
        self.para_C_slider_res.setMinimum(0)
        self.para_C_slider_res.setMaximum(300)
        self.para_C_slider_res.setSingleStep(1)
        self.para_C_slider_res.setValue(1)  # Initial value
        self.para_C_slider_res.valueChanged.connect(self.manipulate_value_changed_res)  # Assuming you want to use the same slot
        self.para_C_label_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,1</sub>/&tau;<sub>dis,1</sub> =</p>")
        self.para_C_label_res.setProperty("class", "sliderlabel")
        
        self.para_C_lineedit_res = QLineEdit("1.0")
        self.para_C_lineedit_res.setProperty("class", "slideredit")
        self.para_C_lineedit_res.returnPressed.connect(self.line_edit_changed_res)  # Same slot as the original
        
        self.para_C_fit_label_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,1</sub>/&tau;<sub>dis,1</sub> = {}</p>".format(0))
        self.para_C_fit_label_res.setProperty("class", "fit")
        self.para_C_fit_label_res.setVisible(False)
        
        self.para_C_fit_label_PCH_res = QLabel()
        self.para_C_fit_label_PCH_res.setProperty("class", "PCH")
        
        self.para_C_label_res.setVisible(False)
        self.para_C_slider_res.setVisible(False)
        self.para_C_lineedit_res.setVisible(False)
        
        self.para_D_slider_res = QSlider(Qt.Horizontal)
        self.para_D_slider_res.setMinimum(0)
        self.para_D_slider_res.setMaximum(300)
        self.para_D_slider_res.setSingleStep(1)
        self.para_D_slider_res.setValue(1)  # Initial value
        self.para_D_slider_res.valueChanged.connect(self.manipulate_value_changed_res)  # Assuming you want to use the same slot
        
        self.para_D_label_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,2</sub>/&tau;<sub>dis,2</sub> =</p>")
        self.para_D_label_res.setProperty("class", "sliderlabel")
        
        self.para_D_lineedit_res = QLineEdit("1.0")
        self.para_D_lineedit_res.setProperty("class", "slideredit")
        self.para_D_lineedit_res.returnPressed.connect(self.line_edit_changed_res)  # Same slot as the original
        
        self.para_D_fit_label_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,2</sub>/&tau;<sub>dis,2</sub> = {}</p>".format(0))
        self.para_D_fit_label_res.setProperty("class", "fit")
        self.para_D_fit_label_res.setVisible(False)
        
        self.para_D_fit_label_PCH_res = QLabel()
        self.para_D_fit_label_PCH_res.setProperty("class", "PCH")
        
        self.para_D_label_res.setVisible(False)
        self.para_D_slider_res.setVisible(False)
        self.para_D_lineedit_res.setVisible(False)
        
        self.para_F_slider_res = QSlider(Qt.Horizontal)
        self.para_F_slider_res.setMinimum(0)
        self.para_F_slider_res.setMaximum(int(1e5))
        self.para_F_slider_res.setSingleStep(100)
        self.para_F_slider_res.setValue(1000)  # Initial value
        self.para_F_slider_res.valueChanged.connect(self.manipulate_value_changed_res)  # Assuming you want to use the same slot
        
        self.para_F_label_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>dis,1</sub> =</p>")
        self.para_F_label_res.setProperty("class", "sliderlabel")
        
        self.para_F_lineedit_res = QLineEdit("1000")
        self.para_F_lineedit_res.setProperty("class", "slideredit")
        self.para_F_lineedit_res.returnPressed.connect(self.line_edit_changed_res)  # Same slot as the original
        
        self.para_F_fit_label_res = QLabel('para_F= {}'.format(0))
        self.para_F_fit_label_res.setProperty("class", "fit")
        self.para_F_fit_label_res.setVisible(False)
        
        self.para_F_fit_label_PCH_res = QLabel()
        self.para_F_fit_label_PCH_res.setProperty("class", "PCH")
        
        self.para_F_label_res.setVisible(False)
        self.para_F_slider_res.setVisible(False)
        self.para_F_lineedit_res.setVisible(False)
        
        self.para_G_slider_res = QSlider(Qt.Horizontal)
        self.para_G_slider_res.setMinimum(0)
        self.para_G_slider_res.setMaximum(300)
        self.para_G_slider_res.setSingleStep(1)
        self.para_G_slider_res.setValue(100)  # Initial value
        self.para_G_slider_res.valueChanged.connect(self.manipulate_value_changed_res)  # Assuming you want to use the same slot
        
        self.para_G_label_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>dis,2</sub>/ &tau;<sub>dis,1</sub> =</p>")
        self.para_G_label_res.setProperty("class", "sliderlabel")
        
        self.para_G_lineedit_res = QLineEdit("1.0")
        self.para_G_lineedit_res.setProperty("class", "slideredit")
        self.para_G_lineedit_res.returnPressed.connect(self.line_edit_changed_res)  # Same slot as the original
        
        self.para_G_fit_label_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>dis,2</sub>/ &tau;<sub>dis,2</sub> = {}</p>".format(0))
        self.para_G_fit_label_res.setProperty("class", "fit")
        self.para_G_fit_label_res.setVisible(False)
        
        self.para_G_fit_label_PCH_res = QLabel()
        self.para_G_fit_label_PCH_res.setProperty("class", "PCH")
        
        self.para_G_label_res.setVisible(False)
        self.para_G_slider_res.setVisible(False)
        self.para_G_lineedit_res.setVisible(False)
        
        #Combo-box for scattering types:
        self.scattering_type_res = QComboBox()

        self.scattering_type_res.addItem("acPh")
        self.scattering_type_res.addItem("dis")
        self.scattering_type_res.addItem("acPhDis")
        self.scattering_type_res.setCurrentIndex(0)
        self.scattering_type_res.currentIndexChanged.connect(self.type_of_fitting_changed_res)

        # Degeneracy labels and line edits
        self.Nv1_label_res = QLabel('deg 1')
        self.Nv1_label_res.setProperty("class", "sliderlabel")
        self.Nv1_label_res.setToolTip("Degeneracy of band 1")
        
        self.Nv1_value_res = QLineEdit('1')
        self.Nv1_value_res.setProperty("class", "slideredit")
        self.Nv1_value_res.setValidator(self.onlyInt)  # assuming onlyInt is a QValidator object defined elsewhere
        self.Nv1_value_res.returnPressed.connect(self.manipulate_value_changed_res)
        
        self.Nv2_label_res = QLabel('deg 2')
        self.Nv2_label_res.setProperty("class", "sliderlabel")
        self.Nv2_label_res.setToolTip("Degeneracy of band 2")
        
        self.Nv2_value_res = QLineEdit('2')
        self.Nv2_value_res.setProperty("class", "slideredit")
        self.Nv2_value_res.setValidator(self.onlyInt)  # assuming onlyInt is a QValidator object defined elsewhere
        self.Nv2_value_res.returnPressed.connect(self.manipulate_value_changed_res)
        
        # Additional buttons and their connections
        self.plot_button_res = QPushButton("Smooth plot")
        self.plot_button_res.clicked.connect(self.plot_button_clicked)
        
        self.manip_fit_button_res = QPushButton("Fit from manipulate")
        self.manip_fit_button_res.clicked.connect(self.manip_fit_button_clicked_2PB_res)
        
        self.experimental_data_res = QRadioButton("Experimental data")
        self.experimental_data_res.toggled.connect(self.experimental_data_toggled_res)
        self.experimental_data_res.setDisabled(True)
        
        # Advanced options, adjust to fit, and individual control buttons
        self.fitpar_button_res = QPushButton("Adjust to fit")
        self.fitpar_button_res.clicked.connect(self.fitpar_button_clicked_res)
        self.fitpar_button_res.setEnabled(False)
        
        self.adv_opt_button_res = QPushButton("Advanced options")
        self.adv_opt_button_res.clicked.connect(self.adv_opt_button_clicked_res)
        
        self.ind_cont_button_res = QPushButton("Additional graphs")
        self.ind_cont_button_res.clicked.connect(self.ind_cont_button_clicked_res)
        
        self.print_data_button_2PB_res = QPushButton("Data to file")
        self.print_data_button_2PB_res.setToolTip("Save all visible data to a single file")
        self.print_data_button_2PB_res.clicked.connect(self.print_data_button_clicked_see_res_2PB)
        
        self.print_params_button_2PB_res = QPushButton("Parameters to file")
        self.print_params_button_2PB_res.setToolTip("Save all parameters to a single file")
        self.print_params_button_2PB_res.clicked.connect(self.print_params_button_clicked_see_res_2PB)
        
        self.deg_PCH_res = QLabel()
        self.deg_PCH_res.setProperty("class", "PCH")
        # Graph and plot widgets for the manipulate window, continued
        # Assuming graph_box2, graph2, and related plot setup is similar and focusing on layout addition here
        self.graph_box2_res = pg.GraphicsLayoutWidget()
        self.graph2_res = self.graph_box2_res.addPlot(2, 3)
        self.graph_box2_res.setStyleSheet(
            "QFrame {border: 5px solid #252530;border-bottom: 5px solid #353540; border-top: 5px solid #151520}")
        self.legend_graph2_res = self.graph2_res.addLegend()
        self.graph_box2_res.setBackground((5, 5, 5))
        self.graph2_res.setTitle(
            "<span style=\"color:#AAAAAA;font-size:16pt;font:Lato;font-weight:semi-bold\">Calculated Seebeck coefficient</span>")
        self.graph2_res.setProperty("class", "seebeck")
        self.graph2_Xaxis_res = self.graph2_res.getAxis("bottom")
        self.graph2_Yaxis_res = self.graph2_res.getAxis("left")
        self.graph2_Xaxis_res.setStyle(tickFont=self.tick_font)
        self.graph2_Yaxis_res.setStyle(tickFont=self.tick_font)
        self.graph2_Xaxis_res.setPen(self.pen_axis)
        self.graph2_Yaxis_res.setPen(self.pen_axis)
        self.graph2_Xaxis_res.setTickPen(self.pen_axis)
        self.graph2_Yaxis_res.setTickPen(self.pen_axis)
        self.graph2_Xaxis_res.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph2_Yaxis_res.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph2_Xaxis_res.setHeight(h=60)
        self.graph2_Yaxis_res.setWidth(w=60)
        self.graph2_Xaxis_res.setLabel(
            '<p style="font-size:18px;color=white">Temperature [K] </p>')
        self.graph2_Yaxis_res.setLabel(
            '<p style="font-size:18px;color=white">Seebeck coefficient [\u03BCV/K] </p>')

        self.legend_graph2_res.addItem(self.manipulate_plot, '<p style="font-size:16px;color=white">Manipulate </p>')
        self.graph2_res.setContentsMargins(2, 0, 5, 10)

        #Resistivity plot for 2PB
        self.graph_box_res = pg.GraphicsLayoutWidget()
        self.graph_res = self.graph_box_res.addPlot(2, 3)
        self.graph_box_res.setStyleSheet(
            "QFrame {border: 5px solid #252530;border-bottom: 5px solid #353540; border-top: 5px solid #151520}")
        self.legend_res = self.graph_res.addLegend()
        self.graph_box_res.setBackground((5, 5, 5))
        self.graph_res.setTitle(
            "<span style=\"color:#AAAAAA;font-size:16pt;font:Lato;font-weight:semi-bold\">Calculated Resistivity</span>")
        self.graph_res.setProperty("class", "seebeck")
        self.graph_Xaxis_res = self.graph_res.getAxis("bottom")
        self.graph_Yaxis_res = self.graph_res.getAxis("left")
        self.graph_Xaxis_res.setStyle(tickFont=self.tick_font)
        self.graph_Yaxis_res.setStyle(tickFont=self.tick_font)
        self.graph_Xaxis_res.setPen(self.pen_axis)
        self.graph_Yaxis_res.setPen(self.pen_axis)
        self.graph_Xaxis_res.setTickPen(self.pen_axis)
        self.graph_Yaxis_res.setTickPen(self.pen_axis)
        self.graph_Xaxis_res.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_Yaxis_res.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_Xaxis_res.setHeight(h=60)
        self.graph_Yaxis_res.setWidth(w=60)
        self.graph_Xaxis_res.setLabel(
            '<p style="font-size:18px;color=white">Temperature [K] </p>')
        self.graph_Yaxis_res.setLabel(
            '<p style="font-size:18px;color=white">Resistivity [\u03BC\u03A9cm] </p>')
        
        #Calculate data for 2PB_see_res window
        paras_see = [self.mass_slider_res.value()/100, self.bandgap_slider_res.value(), self.fermi_slider_res.value()]
        paras_res = [self.para_A_slider_res.value(), self.para_B_slider_res.value()/100, self.para_C_slider_res.value(), self.para_D_slider_res.value(), self.para_F_slider_res.value(), self.para_G_slider_res.value()/100]
        
        x_vals, y1_vals, y2_vals = TE.dpb_see_res_calc(self.minT, self.maxT, 50, paras_see, paras_res, int(self.Nv2_value_res.text())/int(self.Nv1_value_res.text()), scatter_type =f"{self.scattering_type_res.currentText()}")
        
        
        self.manipulate_plot_res_res = pg.PlotCurveItem()
        self.manipulate_plot_res_res.setData(x_vals, y2_vals, pen=self.pen2, clear=True)
        
        # Plot the resulting curve from the initial parameters
        self.manipulate_plot_res = pg.PlotCurveItem()
        self.manipulate_plot_res.setData(x_vals, y1_vals, pen=self.pen2, clear=True)

        self.scatter2_res = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.is_scatter2_there_res = False

        # Plot for the resulting curve from the fit (is left empty here and filled in a function down below)
        self.manip_fit_plot_res = pg.PlotCurveItem()
        self.manip_fit_plot_res_res = pg.PlotCurveItem()
        
        self.legend_graph2_res = self.graph2.addLegend()
        self.graph2_res.addItem(self.manipulate_plot_res)
        self.graph2_res.addItem(self.manip_fit_plot_res)
        self.graph_res.addItem(self.manipulate_plot_res_res)
        self.graph_res.addItem(self.manip_fit_plot_res_res)

        # Plot the resulting effective band structure
        self.graph_3bands_box_res = pg.GraphicsLayoutWidget()
        self.graph_3bands_res = self.graph_3bands_box_res.addPlot(2, 3)
        self.graph_3bands_box_res.setStyleSheet(
            "QFrame {border: 5px solid #252530;border-bottom: 5px solid #353540; border-top: 5px solid #151520}")

        # self.graph_3bands = pg.PlotWidget()
        self.graph_3bands_item_res = pg.PlotCurveItem()
        self.graph_3bands_Xaxis_res = self.graph_3bands_res.getAxis("bottom")
        self.graph_3bands_Yaxis_res = self.graph_3bands_res.getAxis("left")
        self.graph_3bands_Xaxis_res.setStyle(tickFont=self.tick_font_3bands)
        self.graph_3bands_Yaxis_res.setStyle(tickFont=self.tick_font_3bands)
        self.graph_3bands_Xaxis_res.setPen(self.pen_axis)
        self.graph_3bands_Yaxis_res.setPen(self.pen_axis)
        self.graph_3bands_Xaxis_res.setTickPen(self.pen_axis_3bands)
        self.graph_3bands_Yaxis_res.setTickPen(self.pen_axis_3bands)
        self.graph_3bands_Xaxis_res.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_3bands_Yaxis_res.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_3bands_Xaxis_res.setHeight(h=60)
        self.graph_3bands_Yaxis_res.setWidth(w=80)
        self.graph_3bands_Xaxis_res.setLabel(
            '<p style="font-size:18px;color=white">k [1/m] </p>')
        self.graph_3bands_Yaxis_res.setLabel(
            '<p style="font-size:18px;color=white">E-E_VB-edge [eV] </p>')

        self.graph_3bands_res.setContentsMargins(2, 0, 5, 10)

        self.band1_res = pg.PlotCurveItem()
        self.band2_res = pg.PlotCurveItem()
        self.band3_res = pg.PlotCurveItem()

        self.band1_res.setData(
            self.band1_data[0], self.band1_data[1], pen=self.pen_band1, clear=True)
        self.band2_res.setData(
            self.band2_data[0], self.band2_data[1], pen=self.pen_band2, clear=True)
        self.band3_res.setData(
            self.band3_data[0], self.band3_data[1], pen=self.pen_band3, clear=True)

        self.graph_3bands_res.addItem(self.band1_res)
        self.graph_3bands_res.addItem(self.band2_res)
        self.graph_3bands_res.addItem(self.band3_res)

# Adding the widgets to the window on their respective grid slots with updated names for page31_layout
        self.page_seebeckRho_2PB_layout.addWidget(self.label_p2_res, 0, 0, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.label_p2_res, 0, 0, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.experimental_data_res, 0, 3, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.manip_fit_button_res, 0, 6, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.plot_button_res, 0, 9, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.scattering_type_res, 0, 12, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.gif_label_dpb_res, 0, 15, 1, 1)
        self.page_seebeckRho_2PB_layout.addWidget(self.label3_res, 0, 16, 1, 2)
        self.page_seebeckRho_2PB_layout.addWidget(self.label4_res, 0, 18, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.label5_res, 0, 21, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.label5_res, 0, 24, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.Nv1_label_res, 1, 0, 1, 2)
        self.page_seebeckRho_2PB_layout.addWidget(self.Nv1_value_res, 1, 2, 1, 1)
        self.page_seebeckRho_2PB_layout.addWidget(self.Nv2_label_res, 2, 0, 1, 2)
        self.page_seebeckRho_2PB_layout.addWidget(self.Nv2_value_res, 2, 2, 1, 1)
        self.page_seebeckRho_2PB_layout.addWidget(self.deg_PCH_res, 3, 0, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.mass_slider_res, 4, 0, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.mass_label_res, 5, 0, 1, 2)
        self.page_seebeckRho_2PB_layout.addWidget(self.mass_lineedit_res, 5, 2, 1, 1)
        self.page_seebeckRho_2PB_layout.addWidget(self.mass_fit_label_res, 6, 0, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.mass_fit_label_PCH_res, 6, 0, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.bandgap_slider_res, 7, 0, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.bandgap_label_res, 8, 0, 1, 2)
        self.page_seebeckRho_2PB_layout.addWidget(self.bandgap_lineedit_res, 8, 2, 1, 1)
        self.page_seebeckRho_2PB_layout.addWidget(self.bandgap_fit_label_res, 9, 0, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.bandgap_fit_label_PCH_res, 9, 0, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.fermi_slider_res, 10, 0, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.fermi_label_res, 11, 0, 1, 2)
        self.page_seebeckRho_2PB_layout.addWidget(self.fermi_lineedit_res, 11, 2, 1, 1)
        self.page_seebeckRho_2PB_layout.addWidget(self.fermi_fit_label_res, 12, 0, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.fermi_fit_label_PCH_res, 12, 0, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_A_slider_res, 1, 4, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_A_label_res, 2, 4, 1, 2)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_A_lineedit_res, 2, 6, 1, 1)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_A_fit_label_res, 3, 4, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_A_fit_label_PCH_res, 3, 4, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_B_slider_res, 4, 4, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_B_label_res, 5, 4, 1, 2)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_B_lineedit_res, 5, 6, 1, 1)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_B_fit_label_res, 6, 4, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_B_fit_label_PCH_res, 6, 4, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_F_slider_res, 1, 4, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_F_label_res, 2, 4, 1, 2)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_F_lineedit_res, 2, 6, 1, 1)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_F_fit_label_res, 3, 4, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_F_fit_label_PCH_res, 3, 4, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_G_slider_res, 4, 4, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_G_label_res, 5, 4, 1, 2)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_G_lineedit_res, 5, 6, 1, 1)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_G_fit_label_res, 6, 4, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_G_fit_label_PCH_res, 6, 4, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_C_slider_res, 7, 4, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_C_label_res, 8, 4, 1, 2)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_C_lineedit_res, 8, 6, 1, 1)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_C_fit_label_res, 9, 4, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_C_fit_label_PCH_res, 9, 4, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_D_slider_res, 10, 4, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_D_label_res, 11, 4, 1, 2)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_D_lineedit_res, 11, 6, 1, 1)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_D_fit_label_res, 12, 4, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.para_D_fit_label_PCH_res, 12, 4, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.graph_box2_res, 1, 7, 6, 15)
        self.page_seebeckRho_2PB_layout.addWidget(self.graph_box_res, 7, 7, 6, 15)
        self.page_seebeckRho_2PB_layout.addWidget(self.graph_3bands_box_res, 1, 22, 6, 8)
        self.page_seebeckRho_2PB_layout.addWidget(self.fitpar_button_res, 13, 0, 1, 3)
        self.page_seebeckRho_2PB_layout.addWidget(self.adv_opt_button_res, 7, 22, 1, 2)
        self.page_seebeckRho_2PB_layout.addWidget(self.ind_cont_button_res, 7, 24, 1, 2)
        self.page_seebeckRho_2PB_layout.addWidget(self.print_data_button_2PB_res, 8, 22, 1, 2)
        self.page_seebeckRho_2PB_layout.addWidget(self.print_params_button_2PB_res, 8, 24, 1, 2)

    def set_up_2PB_hall_window(self):
        #!!!Working on this
        self.para_A_lower_limit_hall = 0
        self.para_B_lower_limit_hall = 1e-2
        self.para_C_lower_limit_hall = 1e-4
        self.para_D_lower_limit_hall = 1e-4
        self.para_F_lower_limit_hall = 0
        self.para_G_lower_limit_hall = 0
        self.para_E_lower_limit_hall = 1e-3
        
        self.para_A_upper_limit_hall = 1e7
        self.para_B_upper_limit_hall = 5
        self.para_C_upper_limit_hall = 1e4
        self.para_D_upper_limit_hall = 1e4
        self.para_F_upper_limit_hall = 1e6
        self.para_G_upper_limit_hall = 100
        self.para_E_upper_limit_hall = 100
        # Descriptive label on top
        self.label_p2_hall = QLabel("2PB: <i>S</i> + <i>&rho;</i> + <i>Hall</i>")
        self.label_p2_hall.setProperty("class", "title")
        
        #Combo-box for scattering types:
        self.scattering_type_hall = QComboBox()

        self.scattering_type_hall.addItem("acPh")
        self.scattering_type_hall.addItem("dis")
        self.scattering_type_hall.addItem("acPhDis")
        self.scattering_type_hall.setCurrentIndex(0)
        self.scattering_type_hall.currentIndexChanged.connect(self.type_of_fitting_changed_hall)
        
        # Labels
        self.label1_hall = QLabel('aaaaaa')
        self.label2_hall = QLabel('aaaaaa')
        self.label3_hall = QLabel('aaaaaa')
        self.label1_hall.setProperty("class", "PCH")
        self.label2_hall.setProperty("class", "PCH")
        self.label3_hall.setProperty("class", "PCH")
        
        self.label4_hall = QLabel('aaaaaa')
        self.label5_hall = QLabel('aaaaaa')
        self.label6_hall = QLabel('aaaaaa')
        self.label4_hall.setProperty("class", "PCH")
        self.label5_hall.setProperty("class", "PCH")
        self.label6_hall.setProperty("class", "PCH")
        
        # Band mass slider and related widgets
        self.mass_slider_hall = QSlider(Qt.Horizontal)
        self.mass_slider_hall.setMinimum(-300)
        self.mass_slider_hall.setMaximum(300)
        self.mass_slider_hall.setSingleStep(1)
        self.mass_slider_hall.setValue(101)  # Initial value
        self.mass_slider_hall.valueChanged.connect(self.manipulate_value_changed_hall)  # Assuming you want to use the same slot
        
        self.mass_label_hall = QLabel("<p style='font-size: 14px;'>m<sub>2</sub>/m<sub>1</sub> =</p>")
        self.mass_label_hall.setProperty("class", "sliderlabel")
        self.mass_lineedit_hall = QLineEdit("1.01")
        self.mass_lineedit_hall.setProperty("class", "slideredit")
        self.mass_lineedit_hall.returnPressed.connect(self.line_edit_changed_hall)  # Same slot as the original
        
        self.mass_fit_label_hall = QLabel("<p style='font-size: 14px;'>m<sub>2</sub>/m<sub>1</sub> = {}</p>".format(0))
        self.mass_fit_label_hall.setProperty("class", "fit")
        self.mass_fit_label_hall.setVisible(False)
        self.mass_fit_label_PCH_hall = QLabel()
        self.mass_fit_label_PCH_hall.setProperty("class", "PCH")
        
        # Band gap slider and related widgets
        self.bandgap_slider_hall = QSlider(Qt.Horizontal)
        self.bandgap_slider_hall.setMinimum(-3000)
        self.bandgap_slider_hall.setMaximum(3000)
        self.bandgap_slider_hall.setSingleStep(10)
        self.bandgap_slider_hall.setValue(1000)
        self.bandgap_slider_hall.valueChanged.connect(self.manipulate_value_changed_hall)
        
        self.bandgap_label_hall = QLabel("gap =")
        self.bandgap_label_hall.setProperty("class", "sliderlabel")
        self.bandgap_lineedit_hall = QLineEdit("1000")
        self.bandgap_lineedit_hall.setProperty("class", "slideredit")
        self.bandgap_lineedit_hall.returnPressed.connect(self.line_edit_changed_hall)
        
        self.bandgap_fit_label_hall = QLabel('gap = {}'.format(0))
        self.bandgap_fit_label_hall.setProperty("class", "fit")
        self.bandgap_fit_label_hall.setVisible(False)
        self.bandgap_fit_label_PCH_hall = QLabel()
        self.bandgap_fit_label_PCH_hall.setProperty("class", "PCH")
        
        # Fermi level slider and related widgets
        self.fermi_slider_hall = QSlider(Qt.Horizontal)
        self.fermi_slider_hall.setMinimum(-3000)
        self.fermi_slider_hall.setMaximum(3000)
        self.fermi_slider_hall.setSingleStep(1)
        self.fermi_slider_hall.setValue(-100)
        self.fermi_slider_hall.valueChanged.connect(self.manipulate_value_changed_hall)
        
        self.fermi_label_hall = QLabel("<p style='font-size: 14px;'><i>E</i><sub>F</sub> =</p>")
        self.fermi_label_hall.setProperty("class", "sliderlabel")
        
        self.fermi_lineedit_hall = QLineEdit("-100")
        self.fermi_lineedit_hall.setProperty("class", "slideredit")
        self.fermi_lineedit_hall.returnPressed.connect(self.line_edit_changed_hall)
        self.fermi_fit_label_hall = QLabel("<p style='font-size: 14px;'><i>E</i><sub>F</sub> = {}</p>".format(0))
        self.fermi_fit_label_hall.setProperty("class", "fit")
        self.fermi_fit_label_hall.setVisible(False)
        
        self.fermi_fit_label_PCH_hall = QLabel()
        self.fermi_fit_label_PCH_hall.setProperty("class", "PCH")
        
        # Degeneracy labels and line edits
        self.Nv1_label_hall = QLabel('deg 1')
        self.Nv1_label_hall.setProperty("class", "sliderlabel")
        self.Nv1_label_hall.setToolTip("Degeneracy of band 1")
        self.Nv1_value_hall = QLineEdit('1')
        self.Nv1_value_hall.setProperty("class", "slideredit")
        self.Nv1_value_hall.setValidator(self.onlyInt)  # assuming onlyInt is a QValidator object defined elsewhere
        self.Nv1_value_hall.returnPressed.connect(self.manipulate_value_changed_hall)
        
        self.Nv2_label_hall = QLabel('deg 2')
        self.Nv2_label_hall.setProperty("class", "sliderlabel")
        self.Nv2_label_hall.setToolTip("Degeneracy of band 2")
        self.Nv2_value_hall = QLineEdit('2')
        self.Nv2_value_hall.setProperty("class", "slideredit")
        self.Nv2_value_hall.setValidator(self.onlyInt)  # assuming onlyInt is a QValidator object defined elsewhere
        self.Nv2_value_hall.returnPressed.connect(self.manipulate_value_changed_hall)
        
        self.para_A_slider_hall = QSlider(Qt.Horizontal)
        self.para_A_slider_hall.setMinimum(0)
        self.para_A_slider_hall.setMaximum(int(1e6))
        self.para_A_slider_hall.setSingleStep(1000)
        self.para_A_slider_hall.setValue(40000)  # Initial value
        self.para_A_slider_hall.valueChanged.connect(self.manipulate_value_changed_hall)  # Assuming you want to use the same slot
        
        self.para_A_label_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,1</sub> =</p>")
        self.para_A_label_hall.setProperty("class", "sliderlabel")
        self.para_A_lineedit_hall = QLineEdit("40000")
        self.para_A_lineedit_hall.setProperty("class", "slideredit")
        self.para_A_lineedit_hall.returnPressed.connect(self.line_edit_changed_hall)  # Same slot as the original
        
        self.para_A_fit_label_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,1</sub> = {}</p>".format(0))
        self.para_A_fit_label_hall.setProperty("class", "fit")
        self.para_A_fit_label_hall.setVisible(False)
        self.para_A_fit_label_PCH_hall = QLabel()
        self.para_A_fit_label_PCH_hall.setProperty("class", "PCH")
        
        self.para_B_slider_hall = QSlider(Qt.Horizontal)
        self.para_B_slider_hall.setMinimum(0)
        self.para_B_slider_hall.setMaximum(3000)
        self.para_B_slider_hall.setSingleStep(1)
        self.para_B_slider_hall.setValue(100)  # Initial value
        self.para_B_slider_hall.valueChanged.connect(self.manipulate_value_changed_hall)  # Assuming you want to use the same slot

        self.para_B_label_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,2</sub>/&tau;<sub>ph,1</sub> =</p>")
        self.para_B_label_hall.setProperty("class", "sliderlabel")
        
        self.para_B_lineedit_hall = QLineEdit("1.0")
        self.para_B_lineedit_hall.setProperty("class", "slideredit")
        self.para_B_lineedit_hall.returnPressed.connect(self.line_edit_changed_hall)  # Same slot as the original
        self.para_B_fit_label_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,2</sub>/&tau;<sub>ph,1</sub> = {}</p>".format(0))
        self.para_B_fit_label_hall.setProperty("class", "fit")
        self.para_B_fit_label_hall.setVisible(False)
        
        self.para_B_fit_label_PCH_hall = QLabel()
        self.para_B_fit_label_PCH_hall.setProperty("class", "PCH")
        
        self.para_C_slider_hall = QSlider(Qt.Horizontal)
        self.para_C_slider_hall.setMinimum(0)
        self.para_C_slider_hall.setMaximum(300)
        self.para_C_slider_hall.setSingleStep(1)
        self.para_C_slider_hall.setValue(1)  # Initial value
        self.para_C_slider_hall.valueChanged.connect(self.manipulate_value_changed_hall)  # Assuming you want to use the same slot
        self.para_C_label_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,1</sub>/&tau;<sub>dis,1</sub> =</p>")
        self.para_C_label_hall.setProperty("class", "sliderlabel")
        
        self.para_C_lineedit_hall = QLineEdit("1.0")
        self.para_C_lineedit_hall.setProperty("class", "slideredit")
        self.para_C_lineedit_hall.returnPressed.connect(self.line_edit_changed_hall)  # Same slot as the original
        
        self.para_C_label_hall.setVisible(False)
        self.para_C_slider_hall.setVisible(False)
        self.para_C_lineedit_hall.setVisible(False)
        
        self.para_C_fit_label_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,1</sub>/&tau;<sub>dis,1</sub> = {}</p>".format(0))
        self.para_C_fit_label_hall.setProperty("class", "fit")
        self.para_C_fit_label_hall.setVisible(False)
        
        self.para_C_fit_label_PCH_hall = QLabel()
        self.para_C_fit_label_PCH_hall.setProperty("class", "PCH")
        
        self.para_D_slider_hall = QSlider(Qt.Horizontal)
        self.para_D_slider_hall.setMinimum(0)
        self.para_D_slider_hall.setMaximum(300)
        self.para_D_slider_hall.setSingleStep(1)
        self.para_D_slider_hall.setValue(1)  # Initial value
        self.para_D_slider_hall.valueChanged.connect(self.manipulate_value_changed_hall)  # Assuming you want to use the same slot
        
        self.para_D_label_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,2</sub>/&tau;<sub>dis,2</sub> =</p>")
        self.para_D_label_hall.setProperty("class", "sliderlabel")
        
        self.para_D_lineedit_hall = QLineEdit("1.0")
        self.para_D_lineedit_hall.setProperty("class", "slideredit")
        self.para_D_lineedit_hall.returnPressed.connect(self.line_edit_changed_hall)  # Same slot as the original
        
        self.para_D_fit_label_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,2</sub>/&tau;<sub>dis,2</sub> = {}</p>".format(0))
        self.para_D_fit_label_hall.setProperty("class", "fit")
        self.para_D_fit_label_hall.setVisible(False)
        self.para_D_fit_label_PCH_hall = QLabel()
        self.para_D_fit_label_PCH_hall.setProperty("class", "PCH")
        
        self.para_D_label_hall.setVisible(False)
        self.para_D_slider_hall.setVisible(False)
        self.para_D_lineedit_hall.setVisible(False)
        
        self.para_F_slider_hall = QSlider(Qt.Horizontal)
        self.para_F_slider_hall.setMinimum(0)
        self.para_F_slider_hall.setMaximum(int(1e5))
        self.para_F_slider_hall.setSingleStep(1000)
        self.para_F_slider_hall.setValue(1000)  # Initial value
        self.para_F_slider_hall.valueChanged.connect(self.manipulate_value_changed_hall)  # Assuming you want to use the same slot
        
        self.para_F_label_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>dis,1</sub> =</p>")
        self.para_F_label_hall.setProperty("class", "sliderlabel")
        self.para_F_lineedit_hall = QLineEdit("1.0")
        self.para_F_lineedit_hall.setProperty("class", "slideredit")
        self.para_F_lineedit_hall.returnPressed.connect(self.line_edit_changed_hall)  # Same slot as the original
        
        self.para_F_fit_label_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>dis,1</sub> = {}</p>".format(0))
        self.para_F_fit_label_hall.setProperty("class", "fit")
        self.para_F_fit_label_hall.setVisible(False)
        
        self.para_F_fit_label_PCH_hall = QLabel()
        self.para_F_fit_label_PCH_hall.setProperty("class", "PCH")
        self.para_F_label_hall.setVisible(False)
        self.para_F_slider_hall.setVisible(False)
        self.para_F_lineedit_hall.setVisible(False)
        
        self.para_G_slider_hall = QSlider(Qt.Horizontal)
        self.para_G_slider_hall.setMinimum(0)
        self.para_G_slider_hall.setMaximum(10000)
        self.para_G_slider_hall.setSingleStep(1)
        self.para_G_slider_hall.setValue(100)  # Initial value
        self.para_G_slider_hall.valueChanged.connect(self.manipulate_value_changed_hall)  # Assuming you want to use the same slot
        
        self.para_G_label_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>dis,2</sub>/ &tau;<sub>dis,1</sub> =</p>")
        self.para_G_label_hall.setProperty("class", "sliderlabel")
        self.para_G_lineedit_hall = QLineEdit("1.0")
        self.para_G_lineedit_hall.setProperty("class", "slideredit")
        self.para_G_lineedit_hall.returnPressed.connect(self.line_edit_changed_hall)  # Same slot as the original
        
        self.para_G_fit_label_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>dis,2</sub>/ &tau;<sub>dis,1</sub> = {}</p>".format(0))
        self.para_G_fit_label_hall.setProperty("class", "fit")
        self.para_G_fit_label_hall.setVisible(False)
        self.para_G_fit_label_PCH_hall = QLabel()
        self.para_G_fit_label_PCH_hall.setProperty("class", "PCH")
        
        self.para_G_label_hall.setVisible(False)
        self.para_G_slider_hall.setVisible(False)
        self.para_G_lineedit_hall.setVisible(False)
        
        self.para_E_slider_hall = QSlider(Qt.Horizontal)
        self.para_E_slider_hall.setMinimum(0)
        self.para_E_slider_hall.setMaximum(10000)
        self.para_E_slider_hall.setSingleStep(1)
        self.para_E_slider_hall.setValue(100)  # Initial value
        self.para_E_slider_hall.valueChanged.connect(self.manipulate_value_changed_hall)  # Assuming you want to use the same slot
        
        self.para_E_label_hall = QLabel("<p style='font-size: 14px;'>m<sub>1</sub> [m<sub>e</sub>] = </p>")
        self.para_E_label_hall.setProperty("class", "sliderlabel")
        
        self.para_E_lineedit_hall = QLineEdit("1.0")
        self.para_E_lineedit_hall.setProperty("class", "slideredit")
        self.para_E_lineedit_hall.returnPressed.connect(self.line_edit_changed_hall)  # Same slot as the original
        self.para_E_fit_label_hall = QLabel('m<sub>1</sub> [m<sub>e</sub>] = {}'.format(0))
        self.para_E_fit_label_hall.setProperty("class", "fit")
        self.para_E_fit_label_hall.setVisible(False)
        self.para_E_fit_label_PCH_hall = QLabel()
        self.para_E_fit_label_PCH_hall.setProperty("class", "PCH")
        
        # Additional buttons and their connections
        self.plot_button_hall = QPushButton("Smooth plot")
        self.plot_button_hall.clicked.connect(self.plot_button_clicked)
        
        self.manip_fit_button_hall = QPushButton("Fit from manipulate")
        self.manip_fit_button_hall.clicked.connect(self.manip_fit_button_clicked_2PB_hall)
        
        self.experimental_data_hall = QRadioButton("Experimental data")
        self.experimental_data_hall.toggled.connect(self.experimental_data_toggled_hall)
        self.experimental_data_hall.setDisabled(True)
        
        # Advanced options, adjust to fit, and individual control buttons
        self.fitpar_button_hall = QPushButton("Adjust to fit")
        self.fitpar_button_hall.clicked.connect(self.fitpar_button_clicked_hall)
        self.fitpar_button_hall.setEnabled(False)
        
        self.adv_opt_button_hall = QPushButton("Advanced options")
        self.adv_opt_button_hall.clicked.connect(self.adv_opt_button_clicked_hall)
        
        self.ind_cont_button_hall = QPushButton("Additional graphs")
        self.ind_cont_button_hall.clicked.connect(self.ind_cont_button_clicked_hall)
        
        self.print_data_button_2PB_hall = QPushButton("Data to file")
        self.print_data_button_2PB_hall.setToolTip("Save all visible data to a single file")
        self.print_data_button_2PB_hall.clicked.connect(self.print_data_button_clicked_see_res_hall_2PB)
        
        self.print_params_button_2PB_hall = QPushButton("Parameters to file")
        self.print_params_button_2PB_hall.setToolTip("Save all parameters to a single file")
        self.print_params_button_2PB_hall.clicked.connect(self.print_params_button_clicked_see_res_hall_2PB)
           
        self.deg_PCH_hall = QLabel()
        self.deg_PCH_hall.setProperty("class", "PCH")
        # Graph and plot widgets for the manipulate window, continued
        # Assuming graph_box2, graph2, and related plot setup is similar and focusing on layout addition here
        self.graph_box2_hall = pg.GraphicsLayoutWidget()
        self.graph2_hall = self.graph_box2_hall.addPlot(2, 3)
        self.graph_box2_hall.setStyleSheet(
            "QFrame {border: 5px solid #252530;border-bottom: 5px solid #353540; border-top: 5px solid #151520}")
        self.legend_graph2_hall = self.graph2_hall.addLegend()
        self.graph_box2_hall.setBackground((5, 5, 5))
        self.graph2_hall.setTitle(
            "<span style=\"color:#AAAAAA;font-size:16pt;font:Lato;font-weight:semi-bold\">Calculated Seebeck coefficient</span>")
        self.graph2_hall.setProperty("class", "seebeck")
        self.graph2_Xaxis_hall = self.graph2_hall.getAxis("bottom")
        self.graph2_Yaxis_hall = self.graph2_hall.getAxis("left")
        self.graph2_Xaxis_hall.setStyle(tickFont=self.tick_font)
        self.graph2_Yaxis_hall.setStyle(tickFont=self.tick_font)
        self.graph2_Xaxis_hall.setPen(self.pen_axis)
        self.graph2_Yaxis_hall.setPen(self.pen_axis)
        self.graph2_Xaxis_hall.setTickPen(self.pen_axis)
        self.graph2_Yaxis_hall.setTickPen(self.pen_axis)
        self.graph2_Xaxis_hall.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph2_Yaxis_hall.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph2_Xaxis_hall.setHeight(h=60)
        self.graph2_Yaxis_hall.setWidth(w=60)
        self.graph2_Xaxis_hall.setLabel(
            '<p style="font-size:18px;color=white">Temperature [K] </p>')
        self.graph2_Yaxis_hall.setLabel(
            '<p style="font-size:18px;color=white">Seebeck coefficient [\u03BCV/K] </p>')

        self.legend_graph2_hall.addItem(
            self.manipulate_plot, '<p style="font-size:16px;color=white">Manipulate </p>')
        self.graph2_hall.setContentsMargins(2, 0, 5, 10)

        #Resistivity plot for 2PB
        self.graph_box_res_hall = pg.GraphicsLayoutWidget()
        self.graph_res_hall = self.graph_box_res_hall.addPlot(2, 3)
        self.graph_box_res_hall.setStyleSheet(
            "QFrame {border: 5px solid #252530;border-bottom: 5px solid #353540; border-top: 5px solid #151520}")
        self.legend_res_hall = self.graph_res_hall.addLegend()
        self.graph_box_res_hall.setBackground((5, 5, 5))
        self.graph_res_hall.setTitle(
            "<span style=\"color:#AAAAAA;font-size:16pt;font:Lato;font-weight:semi-bold\">Calculated Resistivity</span>")
        self.graph_res_hall.setProperty("class", "seebeck")
        self.graph_Xaxis_res_hall = self.graph_res_hall.getAxis("bottom")
        self.graph_Yaxis_res_hall = self.graph_res_hall.getAxis("left")
        self.graph_Xaxis_res_hall.setStyle(tickFont=self.tick_font)
        self.graph_Yaxis_res_hall.setStyle(tickFont=self.tick_font)
        self.graph_Xaxis_res_hall.setPen(self.pen_axis)
        self.graph_Yaxis_res_hall.setPen(self.pen_axis)
        self.graph_Xaxis_res_hall.setTickPen(self.pen_axis)
        self.graph_Yaxis_res_hall.setTickPen(self.pen_axis)
        self.graph_Xaxis_res_hall.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_Yaxis_res_hall.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_Xaxis_res_hall.setHeight(h=60)
        self.graph_Yaxis_res_hall.setWidth(w=60)
        self.graph_Xaxis_res_hall.setLabel(
            '<p style="font-size:18px;color=white">Temperature [K] </p>')
        self.graph_Yaxis_res_hall.setLabel(
            '<p style="font-size:18px;color=white">Resistivity [\u03BC\u03A9cm] </p>')


        paras_see = [self.mass_slider_hall.value()/100, self.bandgap_slider_hall.value(), self.fermi_slider_hall.value()]
        paras_res = [self.para_A_slider_hall.value(), self.para_B_slider_hall.value()/100, self.para_C_slider_hall.value(), self.para_D_slider_hall.value(), self.para_F_slider_hall.value(), self.para_G_slider_hall.value()/100]
        
        x_vals, y1_vals, y2_vals, y3_vals = TE.dpb_see_res_hall_calc(self.minT, self.maxT, 50, paras_see, paras_res, self.para_E_slider_hall.value()/100, int(self.Nv2_value_res.text())/int(self.Nv1_value_res.text()), scatter_type =f"{self.scattering_type_res.currentText()}")

        self.manipulate_plot_res_hall = pg.PlotCurveItem()
        self.manipulate_plot_res_hall.setData(x_vals, y2_vals, pen=self.pen2, clear=True)

        #Hall plot for 2PB
        self.graph_box_hall = pg.GraphicsLayoutWidget()
        self.graph_hall = self.graph_box_hall.addPlot(2, 3)
        self.graph_box_hall.setStyleSheet(
            "QFrame {border: 5px solid #252530;border-bottom: 5px solid #353540; border-top: 5px solid #151520}")
        self.legend_hall = self.graph_hall.addLegend()
        self.graph_box_hall.setBackground((5, 5, 5))
        self.graph_hall.setTitle(
            "<span style=\"color:#AAAAAA;font-size:16pt;font:Lato;font-weight:semi-bold\">Calculated Hall coefficient</span>")
        self.graph_hall.setProperty("class", "seebeck")
        self.graph_Xaxis_hall = self.graph_hall.getAxis("bottom")
        self.graph_Yaxis_hall = self.graph_hall.getAxis("left")
        self.graph_Xaxis_hall.setStyle(tickFont=self.tick_font)
        self.graph_Yaxis_hall.setStyle(tickFont=self.tick_font)
        self.graph_Xaxis_hall.setPen(self.pen_axis)
        self.graph_Yaxis_hall.setPen(self.pen_axis)
        self.graph_Xaxis_hall.setTickPen(self.pen_axis)
        self.graph_Yaxis_hall.setTickPen(self.pen_axis)
        self.graph_Xaxis_hall.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_Yaxis_hall.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_Xaxis_hall.setHeight(h=60)
        self.graph_Yaxis_hall.setWidth(w=60)
        self.graph_Xaxis_hall.setLabel(
            '<p style="font-size:18px;color=white">Temperature [K] </p>')
        self.graph_Yaxis_hall.setLabel(
            '<p style="font-size:18px;color=white">Hall coefficient [m<sup>3</sup>/A<sup>-1</sup>s<sup>-1</sup>] </p>')

        self.manipulate_plot_hall_hall = pg.PlotCurveItem()
        self.manipulate_plot_hall_hall.setData(x_vals, y3_vals, pen=self.pen2, clear=True)
        self.manip_fit_plot_hall = pg.PlotCurveItem()
        self.manip_fit_plot_res_hall = pg.PlotCurveItem()
        self.manip_fit_plot_hall_hall = pg.PlotCurveItem()

        # Plot the resulting curve from the initial parameters
        self.manipulate_plot_hall = pg.PlotCurveItem()
        self.manipulate_plot_hall.setData(x_vals, y1_vals, pen=self.pen2, clear=True)

        self.scatter2_hall = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(255, 255, 255, 120))
        self.is_scatter2_there_hall = False

        # Plot for the resulting curve from the fit (is left empty here and filled in a function down below)
        self.manip_fit_plot_hall = pg.PlotCurveItem()
        self.manip_fit_plot_res_hall = pg.PlotCurveItem()

        
        self.legend_graph2_hall = self.graph2.addLegend()
        self.graph2_hall.addItem(self.manipulate_plot_hall)
        self.graph2_hall.addItem(self.manip_fit_plot_hall)
        self.graph_res_hall.addItem(self.manipulate_plot_res_hall)
        self.graph_res_hall.addItem(self.manip_fit_plot_res_hall)
        self.graph_hall.addItem(self.manipulate_plot_hall_hall)
        self.graph_hall.addItem(self.manip_fit_plot_hall_hall)

        # Plot the resulting effective band structure
        self.graph_3bands_box_hall = pg.GraphicsLayoutWidget()
        self.graph_3bands_hall = self.graph_3bands_box_hall.addPlot(2, 3)
        self.graph_3bands_box_hall.setStyleSheet(
            "QFrame {border: 5px solid #252530;border-bottom: 5px solid #353540; border-top: 5px solid #151520}")

        # self.graph_3bands = pg.PlotWidget()
        self.graph_3bands_item_hall = pg.PlotCurveItem()
        self.graph_3bands_Xaxis_hall = self.graph_3bands_hall.getAxis("bottom")
        self.graph_3bands_Yaxis_hall = self.graph_3bands_hall.getAxis("left")
        self.graph_3bands_Xaxis_hall.setStyle(tickFont=self.tick_font_3bands)
        self.graph_3bands_Yaxis_hall.setStyle(tickFont=self.tick_font_3bands)
        self.graph_3bands_Xaxis_hall.setPen(self.pen_axis)
        self.graph_3bands_Yaxis_hall.setPen(self.pen_axis)
        self.graph_3bands_Xaxis_hall.setTickPen(self.pen_axis_3bands)
        self.graph_3bands_Yaxis_hall.setTickPen(self.pen_axis_3bands)
        self.graph_3bands_Xaxis_hall.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_3bands_Yaxis_hall.setStyle(
            tickTextOffset=10, tickLength=-7, tickAlpha=220)
        self.graph_3bands_Xaxis_hall.setHeight(h=60)
        self.graph_3bands_Yaxis_hall.setWidth(w=80)
        self.graph_3bands_Xaxis_hall.setLabel(
            '<p style="font-size:18px;color=white">k [1/m] </p>')
        self.graph_3bands_Yaxis_hall.setLabel(
            '<p style="font-size:18px;color=white">E-E_VB-edge [eV] </p>')

        self.graph_3bands_hall.setContentsMargins(2, 0, 5, 10)

        self.band1_hall = pg.PlotCurveItem()
        self.band2_hall = pg.PlotCurveItem()
        self.band3_hall = pg.PlotCurveItem()

        self.band1_hall.setData(
            self.band1_data[0], self.band1_data[1], pen=self.pen_band1, clear=True)
        self.band2_hall.setData(
            self.band2_data[0], self.band2_data[1], pen=self.pen_band2, clear=True)
        self.band3_hall.setData(
            self.band3_data[0], self.band3_data[1], pen=self.pen_band3, clear=True)

        self.graph_3bands_hall.addItem(self.band1_hall)
        self.graph_3bands_hall.addItem(self.band2_hall)
        self.graph_3bands_hall.addItem(self.band3_hall)
                
# Adding the widgets to the window on their respective grid slots with updated names for page32_layout
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.label_p2_hall, 0, 0, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.experimental_data_hall, 0, 3, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.manip_fit_button_hall, 0, 6, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.plot_button_hall, 0, 9, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.scattering_type_hall, 0, 12, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.gif_label_dpb_hall, 0, 15, 1, 1)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.label3_hall, 0, 16, 1, 2)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.label4_hall, 0, 18, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.label5_hall, 0, 21, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.label5_hall, 0, 24, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.Nv1_label_hall, 1, 0, 1, 2)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.Nv1_value_hall, 1, 2, 1, 1)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.Nv2_label_hall, 2, 0, 1, 2)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.Nv2_value_hall, 2, 2, 1, 1)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.deg_PCH_hall, 3, 0, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.mass_slider_hall, 4, 0, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.mass_label_hall, 5, 0, 1, 2)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.mass_lineedit_hall, 5, 2, 1, 1)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.mass_fit_label_hall, 6, 0, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.mass_fit_label_PCH_hall, 6, 0, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.bandgap_slider_hall, 7, 0, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.bandgap_label_hall, 8, 0, 1, 2)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.bandgap_lineedit_hall, 8, 2, 1, 1)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.bandgap_fit_label_hall, 9, 0, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.bandgap_fit_label_PCH_hall, 9, 0, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.fermi_slider_hall, 10, 0, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.fermi_label_hall, 11, 0, 1, 2)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.fermi_lineedit_hall, 11, 2, 1, 1)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.fermi_fit_label_hall, 12, 0, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.fermi_fit_label_PCH_hall, 12, 0, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_A_slider_hall, 1, 4, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_A_label_hall, 2, 4, 1, 2)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_A_lineedit_hall, 2, 6, 1, 1)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_A_fit_label_hall, 3, 4, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_A_fit_label_PCH_hall, 3, 4, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_B_slider_hall, 4, 4, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_B_label_hall, 5, 4, 1, 2)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_B_lineedit_hall, 5, 6, 1, 1)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_B_fit_label_hall, 6, 4, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_B_fit_label_PCH_hall, 6, 4, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_F_slider_hall, 1, 4, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_F_label_hall, 2, 4, 1, 2)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_F_lineedit_hall, 2, 6, 1, 1)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_F_fit_label_hall, 3, 4, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_G_slider_hall, 4, 4, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_G_label_hall, 5, 4, 1, 2)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_G_lineedit_hall, 5, 6, 1, 1)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_G_fit_label_hall, 6, 4, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_G_fit_label_PCH_hall, 6, 4, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_C_slider_hall, 7, 4, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_C_label_hall, 8, 4, 1, 2)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_C_lineedit_hall, 8, 6, 1, 1)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_C_fit_label_hall, 9, 4, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_D_slider_hall, 10, 4, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_D_label_hall, 11, 4, 1, 2)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_D_lineedit_hall, 11, 6, 1, 1)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_D_fit_label_hall, 12, 4, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_D_fit_label_PCH_hall, 12, 4, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_E_slider_hall, 13, 4, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_E_label_hall, 14, 4, 1, 2)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_E_lineedit_hall, 14, 6, 1, 1)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_E_fit_label_hall, 15, 4, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.para_E_fit_label_PCH_hall, 15, 4, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.graph_box2_hall, 1, 7, 6, 15)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.graph_box_res_hall, 7, 7, 6, 15)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.graph_box_hall, 7, 22, 6, 8)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.graph_3bands_box_hall, 1, 22, 6, 8)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.fitpar_button_hall, 13, 0, 1, 3)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.adv_opt_button_hall, 13, 22, 1, 2)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.ind_cont_button_hall, 13, 24, 1, 2)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.print_data_button_2PB_hall, 14, 22, 1, 2)
        self.page_seebeckRhoHall_2PB_layout.addWidget(self.print_params_button_2PB_hall, 14, 24, 1, 2)

    def set_up_waiting_gifs(self):
        self.gif_label_spb_see = QLabel()
        self.gif_label_spb_res = QLabel()
        self.gif_label_spb_hall = QLabel()
        self.gif_label_dpb_see = QLabel()
        self.gif_label_dpb_res = QLabel()
        self.gif_label_dpb_hall = QLabel()
        gif_label_list = [self.gif_label_spb_see, self.gif_label_spb_res, self.gif_label_spb_hall, self.gif_label_dpb_see, self.gif_label_dpb_res, self.gif_label_dpb_hall]
        
        for label in gif_label_list:
            label.setGeometry(QRect(0, 0, 100, 100))
            label.setProperty("class", "gif")
            
        self.movie = QMovie("Icons\\spinner_new.gif")
        self.movie.setScaledSize(QSize().scaled(60, 60, Qt.KeepAspectRatio))
        
        # self.gif_label.setMovie(self.movie)
        
        # for label in gif_label_list:
        #     self.start_waiting_gif(label)
        
    def start_waiting_gif(self, label: QLabel):
        label.setMovie(self.movie)
        self.movie.start()
        
    def stop_waiting_gif(self, label: QLabel):
        self.movie.stop()
        label.clear()
    
    def error_stop_waiting_gif(self, label_string: str):
        if (label_string == "spb_see"):
            self.stop_waiting_gif(self.gif_label_spb_see)
        elif (label_string == "spb_res"):
            self.stop_waiting_gif(self.gif_label_spb_res)
        elif (label_string == "spb_hall"):
            self.stop_waiting_gif(self.gif_label_spb_hall)
        elif (label_string == "dpb_see"):
            self.stop_waiting_gif(self.gif_label_dpb_see)
        elif (label_string == "dpb_res"):
            self.stop_waiting_gif(self.gif_label_dpb_res)
        elif (label_string == "dpb_hall"):
            self.stop_waiting_gif(self.gif_label_dpb_hall)
                
        
    def update_fit_results_1PB_see(self, fit_x_val, fit_y_val, params):
        self.manip_fit_plot_1PB.setZValue(5)
        self.manip_fit_plot_1PB.setData(fit_x_val, fit_y_val, pen=self.pen3)
        if self.label_here:
            self.legend_graph2.addItem(self.manip_fit_plot_1PB, '<p style="font-size:16px;color=white">Fitting </p>')
            self.label_here = False
        self.fermi_fit_label_1PB.setText('Fermi energy = {:.0f}'.format(params[0]))
        self.fermi_fit_label_1PB.setVisible(True)
        self.fermi_fit_label_PCH_1PB.setVisible(False)
        
        self.stop_waiting_gif(self.gif_label_spb_see)
        
    def update_fit_results_1PB_res(self, fit_x_val, fit_y_val_see, fit_y_val_res, params):
        self.manip_fit_plot_1PB_res.setZValue(5)
        self.manip_fit_plot_1PB_res.setData(fit_x_val, fit_y_val_see, pen=self.pen3)
        self.manip_fit_plot_1PB_rho_res.setZValue(5)
        self.manip_fit_plot_1PB_rho_res.setData(fit_x_val, fit_y_val_res, pen=self.pen3)
        if self.label_here:
            self.legend_graph2.addItem(self.manip_fit_plot_1PB_res, '<p style="font-size:16px;color=white">Fitting </p>')
            self.label_here = False
        self.fermi_fit_label_1PB_res.setText('Fermi energy = {:.0f}'.format(params[0]))
        self.para_A_fit_label_1PB_res.setText('&tau;<sub>ph,1</sub> = {:.0f}'.format(params[1]))
        self.para_C_fit_label_1PB_res.setText('&tau;<sub>ph,1</sub>/&tau;<sub>dis,1</sub> @ 300 K = {:.0f}'.format(params[2]))
        self.para_F_fit_label_1PB_res.setText('&tau;<sub>dis,1</sub> = {:.0f}'.format(params[3]))
        self.fermi_fit_label_1PB_res.setVisible(True)
        self.fermi_fit_label_PCH_1PB_res.setVisible(False)
        
        if self.scattering_type_1PB_res.currentText() == "acPh":
            self.para_A_fit_label_1PB_res.setVisible(True)
            self.para_A_fit_label_PCH_1PB_res.setVisible(False)
            
        elif self.scattering_type_res.currentText() == "dis":
            self.para_F_fit_label_1PB_res.setVisible(True)
            self.para_F_fit_label_PCH_1PB_res.setVisible(False)

        elif self.scattering_type_res.currentText() == "acPhDis":
            self.para_A_fit_label_1PB_res.setVisible(True)
            self.para_C_fit_label_1PB_res.setVisible(True)
            self.para_A_fit_label_PCH_1PB_res.setVisible(False)
            self.para_C_fit_label_PCH_1PB_res.setVisible(False)
        
        self.stop_waiting_gif(self.gif_label_spb_res)
        
    def update_fit_results_1PB_hall(self, fit_x_val, fit_y_val_see, fit_y_val_res, fit_y_val_hall, params):
        self.manip_fit_plot_1PB_hall.setZValue(5)
        self.manip_fit_plot_1PB_hall.setData(fit_x_val, fit_y_val_see, pen=self.pen3)
        self.manip_fit_plot_1PB_rho_hall.setZValue(5)
        self.manip_fit_plot_1PB_rho_hall.setData(fit_x_val, fit_y_val_res, pen=self.pen3)
        self.manip_fit_plot_1PB_Hall_hall.setZValue(5)
        self.manip_fit_plot_1PB_Hall_hall.setData(fit_x_val, fit_y_val_hall, pen=self.pen3)
        if self.label_here:
            self.legend_graph2.addItem(self.manip_fit_plot_1PB_hall, '<p style="font-size:16px;color=white">Fitting </p>')
            self.label_here = False
        self.fermi_fit_label_1PB_hall.setText('Fermi energy = {:.0f}'.format(params[0]))
        self.para_A_fit_label_1PB_hall.setText('&tau;<sub>ph,1</sub> = {:.0f}'.format(params[1]))
        self.para_C_fit_label_1PB_hall.setText('&tau;<sub>ph,1</sub>/&tau;<sub>dis,1</sub> @ 300 K = {:.0f}'.format(params[2]))
        self.para_F_fit_label_1PB_hall.setText('&tau;<sub>dis,1</sub> = {:.0f}'.format(params[3]))
        self.mass_fit_SPB_label_hall.setText('m<sub>1</sub> [m<sub>e</sub>] = {:.2f}'.format(params[4]))
        
        self.mass_fit_SPB_label_hall.setVisible(True)
        self.mass_fit_SPB_label_PCH_hall.setVisible(False)
        self.fermi_fit_label_1PB_hall.setVisible(True)
        self.fermi_fit_label_PCH_1PB_hall.setVisible(False)
        #self.fitpar_button_1PB_hall.setEnabled(True)
        
        if self.scattering_type_1PB_hall.currentText() == "acPh":
            self.para_A_fit_label_1PB_hall.setVisible(True)
            self.para_A_fit_label_PCH_1PB_hall.setVisible(False)
            
        elif self.scattering_type_hall.currentText() == "dis":
            self.para_F_fit_label_1PB_hall.setVisible(True)
            self.para_F_fit_label_PCH_1PB_hall.setVisible(False)

        elif self.scattering_type_hall.currentText() == "acPhDis":
            self.para_A_fit_label_1PB_hall.setVisible(True)
            self.para_C_fit_label_1PB_hall.setVisible(True)
            self.para_A_fit_label_PCH_1PB_hall.setVisible(False)
            self.para_C_fit_label_PCH_1PB_hall.setVisible(False)
        
        self.stop_waiting_gif(self.gif_label_spb_hall)
        
    def update_fit_results_2PB_see(self, fit_x_val, fit_y_val, params):
        self.manip_fit_plot.setZValue(5)
        self.manip_fit_plot.setData(fit_x_val, fit_y_val, pen=self.pen3)
        if self.label_here:
            self.legend_graph2.addItem(
                self.manip_fit_plot, '<p style="font-size:16px;color=white">Fitting </p>')
            self.label_here = False
        self.mass_fit_label.setText('mass = {:.2f}'.format(params[0]))
        self.bandgap_fit_label.setText('gap = {:.0f}'.format(params[1]))
        self.fermi_fit_label.setText('Fermi energy = {:.0f}'.format(params[2]))
        self.mass_fit_label.setVisible(True)
        self.bandgap_fit_label.setVisible(True)
        self.fermi_fit_label.setVisible(True)
        self.mass_fit_label_PCH.setVisible(False)
        self.bandgap_fit_label_PCH.setVisible(False)
        self.fermi_fit_label_PCH.setVisible(False)
        self.fitpar_button.setEnabled(True)
        
        self.stop_waiting_gif(self.gif_label_dpb_see)
    
    def update_fit_results_2PB_res(self, fit_x_val, fit_y_val_see, fit_y_val_res, params):
        self.manip_fit_plot_res.setZValue(5)
        self.manip_fit_plot_res.setData(fit_x_val, fit_y_val_see, pen=self.pen3)
        self.manip_fit_plot_res_res.setZValue(5)
        self.manip_fit_plot_res_res.setData(fit_x_val, fit_y_val_res, pen=self.pen3)

        self.mass_fit_label_res.setText('mass = {:.2f}'.format(params[0]))
        self.bandgap_fit_label_res.setText('gap = {:.0f}'.format(params[1]))
        self.fermi_fit_label_res.setText('Fermi energy = {:.0f}'.format(params[2]))
        self.para_A_fit_label_res.setText('&tau;<sub>ph,1</sub> = {:.2f}'.format(params[3]))
        self.para_B_fit_label_res.setText('&tau;<sub>ph,2</sub>/&tau;<sub>ph,1</sub> = {:.2f}'.format(params[4]))
        self.para_C_fit_label_res.setText('&tau;<sub>ph,1</sub>/&tau;<sub>dis,1</sub> @ 300 K = {:.2f}'.format(params[5]))
        self.para_D_fit_label_res.setText('&tau;<sub>ph,2</sub>/&tau;<sub>dis,2</sub> @ 300 K = {:.2f}'.format(params[6]))
        self.para_F_fit_label_res.setText('&tau;<sub>dis,1</sub> = {:.2f}'.format(params[7]))
        self.para_G_fit_label_res.setText('&tau;<sub>dis,2</sub>/&tau;<sub>dis,1</sub> = {:.2f}'.format(params[8]))
        
        if self.scattering_type_res.currentText() == "acPh":

            self.mass_fit_label_res.setVisible(True)
            self.bandgap_fit_label_res.setVisible(True)
            self.fermi_fit_label_res.setVisible(True)
            self.mass_fit_label_PCH_res.setVisible(False)
            self.bandgap_fit_label_PCH_res.setVisible(False)
            self.fermi_fit_label_PCH_res.setVisible(False)
            
            self.para_A_fit_label_res.setVisible(True)
            self.para_B_fit_label_res.setVisible(True)
            # self.para_E_fit_label_res.setVisible(True)
            self.para_A_fit_label_PCH_res.setVisible(False)
            self.para_B_fit_label_PCH_res.setVisible(False)
            self.para_G_fit_label_PCH_res.setVisible(False)


        elif self.scattering_type_res.currentText() == "dis":
            self.mass_fit_label_res.setVisible(True)
            self.bandgap_fit_label_res.setVisible(True)
            self.fermi_fit_label_res.setVisible(True)
            self.mass_fit_label_PCH_res.setVisible(False)
            self.bandgap_fit_label_PCH_res.setVisible(False)
            self.fermi_fit_label_PCH_res.setVisible(False)
            
            self.para_F_fit_label_res.setVisible(True)
            self.para_G_fit_label_res.setVisible(True)
            self.para_F_fit_label_PCH_res.setVisible(False)
            self.para_G_fit_label_PCH_res.setVisible(False)


        elif self.scattering_type_res.currentText() == "acPhDis":
            self.mass_fit_label_res.setVisible(True)
            self.bandgap_fit_label_res.setVisible(True)
            self.fermi_fit_label_res.setVisible(True)
            
            self.para_A_fit_label_res.setVisible(True)
            self.para_B_fit_label_res.setVisible(True)
            self.para_C_fit_label_res.setVisible(True)
            self.para_D_fit_label_res.setVisible(True)
            
            self.mass_fit_label_PCH_res.setVisible(False)
            self.bandgap_fit_label_PCH_res.setVisible(False)
            self.fermi_fit_label_PCH_res.setVisible(False)
            
            # self.fitpar_button_res.setEnabled(True)
            
            self.para_A_fit_label_PCH_res.setVisible(False)
            self.para_B_fit_label_PCH_res.setVisible(False)
            self.para_C_fit_label_PCH_res.setVisible(False)
            self.para_D_fit_label_PCH_res.setVisible(False)
            self.para_G_fit_label_PCH_res.setVisible(False)
        
        self.fitpar_button_res.setEnabled(True)
        
        self.stop_waiting_gif(self.gif_label_dpb_res)
    
    def update_fit_results_2PB_hall(self, fit_x_val, fit_y_val_see, fit_y_val_res, fit_y_val_hall, params):
        self.manip_fit_plot_hall.setZValue(5)
        self.manip_fit_plot_hall.setData(fit_x_val, fit_y_val_see, pen=self.pen3)
        self.manip_fit_plot_res_hall.setZValue(5)
        self.manip_fit_plot_res_hall.setData(fit_x_val, fit_y_val_res, pen=self.pen3)
        self.manip_fit_plot_hall_hall.setZValue(5)
        self.manip_fit_plot_hall_hall.setData(fit_x_val, fit_y_val_hall, pen=self.pen3)

        self.mass_fit_label_hall.setText('mass = {:.2f}'.format(params[0]))
        self.bandgap_fit_label_hall.setText('gap = {:.0f}'.format(params[1]))
        self.fermi_fit_label_hall.setText('Fermi energy = {:.0f}'.format(params[2]))
        self.para_A_fit_label_hall.setText('&tau;<sub>ph,1</sub> = {:.2f}'.format(params[3]))
        self.para_B_fit_label_hall.setText('&tau;<sub>ph,2</sub>/&tau;<sub>ph,1</sub> = {:.2f}'.format(params[4]))
        self.para_C_fit_label_hall.setText('&tau;<sub>ph,1</sub>/&tau;<sub>dis,1</sub> @ 300 K = {:.2f}'.format(params[5]))
        self.para_D_fit_label_hall.setText('&tau;<sub>ph,2</sub>/&tau;<sub>dis,2</sub> @ 300 K = {:.2f}'.format(params[6]))
        self.para_F_fit_label_hall.setText('&tau;<sub>dis,1</sub> = {:.2f}'.format(params[7]))
        self.para_G_fit_label_hall.setText('&tau;<sub>dis,2</sub>/&tau;<sub>dis,1</sub> = {:.2f}'.format(params[8]))
        self.para_E_fit_label_hall.setText('m<sub>1</sub> [m<sub>e</sub>] = {:.2f}'.format(params[9]))
        
        if self.scattering_type_hall.currentText() == "acPh":

            self.mass_fit_label_hall.setVisible(True)
            self.bandgap_fit_label_hall.setVisible(True)
            self.fermi_fit_label_hall.setVisible(True)
            self.mass_fit_label_PCH_hall.setVisible(False)
            self.bandgap_fit_label_PCH_hall.setVisible(False)
            self.fermi_fit_label_PCH_hall.setVisible(False)
            
            self.para_A_fit_label_hall.setVisible(True)
            self.para_B_fit_label_hall.setVisible(True)
            self.para_E_fit_label_hall.setVisible(True)
            self.para_A_fit_label_PCH_hall.setVisible(False)
            self.para_B_fit_label_PCH_hall.setVisible(False)
            self.para_E_fit_label_PCH_hall.setVisible(False)
            self.para_G_fit_label_PCH_hall.setVisible(False)
            

        elif self.scattering_type_hall.currentText() == "dis":
            self.mass_fit_label_hall.setVisible(True)
            self.bandgap_fit_label_hall.setVisible(True)
            self.fermi_fit_label_hall.setVisible(True)
            self.mass_fit_label_PCH_hall.setVisible(False)
            self.bandgap_fit_label_PCH_hall.setVisible(False)
            self.fermi_fit_label_PCH_hall.setVisible(False)
            
            self.para_F_fit_label_hall.setVisible(True)
            self.para_G_fit_label_hall.setVisible(True)
            self.para_E_fit_label_hall.setVisible(True)
            self.para_F_fit_label_PCH_hall.setVisible(False)
            self.para_G_fit_label_PCH_hall.setVisible(False)
            self.para_E_fit_label_PCH_hall.setVisible(False)


        elif self.scattering_type_hall.currentText() == "acPhDis":
            self.mass_fit_label_hall.setVisible(True)
            self.bandgap_fit_label_hall.setVisible(True)
            self.fermi_fit_label_hall.setVisible(True)
            self.para_A_fit_label_hall.setVisible(True)
            self.para_B_fit_label_hall.setVisible(True)
            self.para_C_fit_label_hall.setVisible(True)
            self.para_D_fit_label_hall.setVisible(True)
            self.para_E_fit_label_hall.setVisible(True)
            self.mass_fit_label_PCH_hall.setVisible(False)
            self.bandgap_fit_label_PCH_hall.setVisible(False)
            self.fermi_fit_label_PCH_hall.setVisible(False)
            self.fitpar_button_hall.setEnabled(True)
            self.para_A_fit_label_PCH_hall.setVisible(False)
            self.para_B_fit_label_PCH_hall.setVisible(False)
            self.para_C_fit_label_PCH_hall.setVisible(False)
            self.para_D_fit_label_PCH_hall.setVisible(False)
            self.para_E_fit_label_PCH_hall.setVisible(False)
            self.para_G_fit_label_PCH_hall.setVisible(False)
        
        self.fitpar_button_hall.setEnabled(True)
        
        self.stop_waiting_gif(self.gif_label_dpb_hall)
    
    '''
    Functions for the toolbar and and all the widgets of all pages initialized above
    -----------------------------------------------------------------------------------------
    '''

    '''
    Toolbar functions
    '''
    # Function for the bug-button
    def onMyToolBar_logo(self, s):
        if s == True:
            pass
        else:
            pass

    # Switch to the fitting window
    def onMyToolBar_importData(self, s):
        self.stacked_layout.setCurrentIndex(0)
        print("Index set to 0")

    # Switch to one parabolic band
    def onMyToolBar_switch1PB(self, s):
        self.switch2PB_action.setChecked(False)
        self.bands = 1
        if 4 <= self.stacked_layout.currentIndex() <= 6:
            self.stacked_layout.setCurrentIndex(self.stacked_layout.currentIndex()-3)

    # Switch to two parabolic bands
    def onMyToolBar_switch2PB(self, s):
        self.switch1PB_action.setChecked(False)
        self.bands = 2
        if 1 <= self.stacked_layout.currentIndex() <= 3:
            self.stacked_layout.setCurrentIndex(self.stacked_layout.currentIndex()+3)
            
    def onMyToolBar_seebeckOnly(self, s):
        if self.bands == 1:
            self.stacked_layout.setCurrentIndex(1)
            print("Index set to 1")
        if self.bands == 2:
            self.stacked_layout.setCurrentIndex(4)
            print("Index set to 4")
            
    def onMyToolBar_seebeckRho(self, s):
        if self.bands == 1:
            self.stacked_layout.setCurrentIndex(2)
            print("Index set to 2")
        if self.bands == 2:
            self.stacked_layout.setCurrentIndex(5)
            print("Index set to 5")
            
    def onMyToolBar_seebeckRhoHall(self, s):
        if self.bands == 1:
            self.stacked_layout.setCurrentIndex(3)
            print("Index set to 3")
        if self.bands == 2:
            self.stacked_layout.setCurrentIndex(6)
            print("Index set to 6")
            
    def onMyToolBar_exit(self):
        self.close()

    # Bring window to the center of the screen
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def mousePressEvent(self, event):
        self.oldPos = event.globalPos()

    # Move window according to mouse movement
    def mouseMoveEvent(self, event):
        delta = QPoint(event.globalPos() - self.oldPos)
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.oldPos = event.globalPos()

    '''
    Seebeck fitting functions
    '''        
    # Start a file dialog to import the Seebeck data
    def seeb_databutton_clicked(self, s):
        self.experimental_data_1PB.setEnabled(True)
        self.experimental_data_1PB_res.setEnabled(True)
        self.experimental_data_1PB_hall.setEnabled(True)
        self.experimental_data.setEnabled(True)
        self.experimental_data_res.setEnabled(True)
        self.experimental_data_hall.setEnabled(True)
        
        file_path, _ = QFileDialog.getOpenFileName()
        try:
            self.data = dp.get_data(file_path)
        except Exception as e:
            print(e)
            return
        self.interpolated_data = self.data
        self.x_mask = (self.interpolated_data[:, 0] >= self.minT) & (
            self.interpolated_data[:, 0] <= self.maxT)
        
        self.graph.addItem(self.scatter)
        self.scatter.setData(self.data[:, 0], self.data[:, 1], clear=True)
        
        #Assign the experimental data to all the scatters for the respective windows
        for key in self.experimental_data_scatter["see"]:
            self.experimental_data_scatter["see"][key].setData(self.data[:, 0], self.data[:, 1])
            
        #get AI prediction
        prop_model = ai.select_model(
            self.data[:,0], 
            self.model_100_400, 
            self.model_200_500, 
            self.model_300_600, 
            self.model_300_800, 
            self.model_400_700)
        if (prop_model != None):
            ai_pred = ai.get_prediction(prop_model, self.data[:, 0], self.data[:, 1])
        
            self.mass_slider.setValue(int(np.round(ai_pred[0]*100)))
            self.bandgap_slider.setValue(int(round(ai_pred[1])))
            self.fermi_slider.setValue(int(round(ai_pred[2])))
            self.manipulate_value_changed_2PB_see()

    def res_databutton_clicked(self, s):
        self.experimental_data_1PB.setEnabled(True)
        self.experimental_data_1PB_res.setEnabled(True)
        self.experimental_data_1PB_hall.setEnabled(True)
        self.experimental_data.setEnabled(True)
        self.experimental_data_res.setEnabled(True)
        self.experimental_data_hall.setEnabled(True)
        
        
        try:
            file_path, _ = QFileDialog.getOpenFileName()
            if not file_path:  # Check if the file path is empty
                print("No file selected.")
                return
            self.res_data = dp.get_data(file_path)
            self.interpolated_res_data = self.res_data
            self.x_mask_res = (self.interpolated_res_data[:, 0] >= self.minT) & (
                self.interpolated_res_data[:, 0] <= self.maxT)
            self.scatter_res.addPoints(self.res_data[:, 0], self.res_data[:, 1], clear=True)
            
            #Assign the experimental data to all the scatters for the respective windows
            for key in self.experimental_data_scatter["res"]:
                self.experimental_data_scatter["res"][key].setData(self.res_data[:, 0], self.res_data[:, 1])
            self.preview_res_graph.addItem(self.scatter_res)

        except Exception as e:
            print(f"{e} happened. Please choose carefully!")
            return
   
    def hall_databutton_clicked(self, s):
        self.experimental_data_1PB.setEnabled(True)
        self.experimental_data_1PB_res.setEnabled(True)
        self.experimental_data_1PB_hall.setEnabled(True)
        self.experimental_data.setEnabled(True)
        self.experimental_data_res.setEnabled(True)
        self.experimental_data_hall.setEnabled(True)
        
        try:
            file = QFileDialog.getOpenFileName()
            self.hall_data = dp.get_data(file[0])
            self.interpolated_hall_data = self.hall_data
            self.x_mask_hall = (self.interpolated_hall_data[:, 0] >= self.minT) & (
                self.interpolated_hall_data[:, 0] <= self.maxT)
            self.scatter_hall.addPoints(self.hall_data[:, 0], self.hall_data[:, 1], clear=True)
            
            #Assign the experimental data to all the scatters for the respective windows
            for key in self.experimental_data_scatter["hall"]:
                self.experimental_data_scatter["hall"][key].setData(self.hall_data[:, 0], self.hall_data[:, 1])
            self.preview_hall_graph.addItem(self.scatter_hall)

        except Exception as e:
            print(f"{e} happened. Please choose carefully!")
            return


    # Remove all imported experimental Seebeck data and delete it from the plots
    #!!!TODO: Aktuell werden die Experimental data radiobuttons nicht wieder ausgeschalten, wenn die Daten entfernt werden
    #!!! Das liegt daran, dass es relativ kompliziert wäre, immer nachzusehen, ob irgendwo die Daten noch drin sind, bevor mans in allen fenstern ausschaltet
    #!!! Man müsste quasi immer Abfragen und das dann für jedes Fenster eigens Konditionell machen (3x 2 (1PB und 2PB sollten gleich sein))
    def clear_seeb_data_button_clicked(self):
        self.graph.removeItem(self.scatter)
        self.graph.removeItem(self.scatter_int)
        self.scatter.clear()
        self.scatter_int.clear()
        
        #Clear the scatter_plots
        for key in self.experimental_data_scatter["see"]:
            self.experimental_data_scatter["see"][key].clear()

        self.data = False
        self.graph.update()
        # self.experimental_data.setChecked(False)
        # self.experimental_data.setDisabled(True)
        # self.experimental_data_res.setChecked(False)
        # self.experimental_data_res.setDisabled(True)
        # self.experimental_data_hall.setChecked(False)
        # self.experimental_data_hall.setDisabled(True)
        # self.experimental_data1.setChecked(False)
        # self.experimental_data1.setDisabled(True)
        self.initialized_widgets["int_seeb_data_button"].setChecked(False)
        
    def clear_res_data_button_clicked(self):
        self.preview_res_graph.removeItem(self.scatter_res)
        self.preview_res_graph.removeItem(self.scatter_res_int)
        self.scatter_res.clear()
        self.scatter_res_int.clear()
        

        #Clear the scatter_plots
        for key in self.experimental_data_scatter["res"]:
            self.experimental_data_scatter["res"][key].clear()

        self.res_data = False
        self.res_data_int = False
        self.preview_res_graph.update()
        self.initialized_widgets["int_res_data_button"].setChecked(False)
        
        #Clear the scatter_plots
        for key in self.experimental_data_scatter["see"]:
            self.experimental_data_scatter["see"][key].clear()
    
    def clear_hall_data_button_clicked(self):
        self.preview_hall_graph.removeItem(self.scatter_hall)
        self.preview_hall_graph.removeItem(self.scatter_hall_int)
        self.scatter_hall.clear()
        self.scatter_hall_int.clear()

        #Assign the experimental data to all the scatters for the respective windows
        for key in self.experimental_data_scatter["hall"]:
            self.experimental_data_scatter["hall"][key].clear()

        self.hall_data = False
        self.hall_data_int = False
        self.preview_hall_graph.update()
        self.initialized_widgets["int_hall_data_button"].setChecked(False)

    # Function to interpolate the measurement data to have equal spacing on the x-axis
    def int_seeb_data_button_clicked(self, s):
        try:
            if s == 1:
                data = dp.interpolate_data(dp.fit_data(self.data, 10), np.linspace(
                    self.minT, self.maxT, int((self.maxT-self.minT)/20.)))
                self.scatter_int.setData(data[:, 0], data[:, 1])
                self.interpolated_data = data
                self.x_mask = (self.interpolated_data[:, 0] >= self.minT) & (
                    self.interpolated_data[:, 0] <= self.maxT)

                self.graph.addItem(self.scatter_int)
                self.graph.removeItem(self.scatter)

                #Assign the interpolated experimental data to all the scatters for the respective windows
                self.scatter.setData(self.data[:, 0], self.data[:, 1], clear=True)
                for key in self.experimental_data_scatter["see"]:
                    self.experimental_data_scatter["see"][key].setData(self.interpolated_data[:, 0], self.interpolated_data[:, 1])

            elif s == 0:
                self.graph.addItem(self.scatter)
                self.graph.removeItem(self.scatter_int)
                
                #Assign the non-interpolated experimental data to all the scatters for the respective windows
                self.scatter.setData(self.data[:, 0], self.data[:, 1], clear=True)
                for key in self.experimental_data_scatter["see"]:
                    self.experimental_data_scatter["see"][key].setData(self.data[:, 0], self.data[:, 1])

                self.x_mask = (self.data[:, 0] >= self.minT) & (
                    self.data[:, 0] <= self.maxT)
                
        except Exception as e:
            print("No data to interpolate:", "({})".format(e))
            self.initialized_widgets["int_seeb_data_button"].setChecked(False)
    
    def int_res_data_button_clicked(self, s):
        try:
            if s == 1:
                data = dp.interpolate_data(dp.fit_data(self.res_data, 10), np.linspace(
                    self.minT, self.maxT, int((self.maxT-self.minT)/20.)))
                self.scatter_res_int.setData(data[:, 0], data[:, 1])
                self.interpolated_res_data = data
                self.x_mask_res = (self.interpolated_res_data[:, 0] >= self.minT) & (
                    self.interpolated_res_data[:, 0] <= self.maxT)
                
                #Assign the interpolated experimental data to all the scatters for the respective windows
                for key in self.experimental_data_scatter["res"]:
                    self.experimental_data_scatter["res"][key].setData(self.interpolated_res_data[:, 0], self.interpolated_res_data[:, 1])

                self.preview_res_graph.addItem(self.scatter_res_int)
                self.preview_res_graph.removeItem(self.scatter_res)

            elif s == 0:
                self.preview_res_graph.addItem(self.scatter_res)
                self.preview_res_graph.removeItem(self.scatter_res_int)
                self.x_mask_res = (self.res_data[:, 0] >= self.minT) & (
                    self.res_data[:, 0] <= self.maxT)
                
                #Assign the experimental data to all the scatters for the respective windows
                for key in self.experimental_data_scatter["res"]:
                    self.experimental_data_scatter["res"][key].setData(self.res_data[:, 0], self.res_data[:, 1])
                    
        except Exception as e:
            print("No data to interpolate:", "({})".format(e))
            self.initialized_widgets["int_res_data_button"].setChecked(False)
    
    def int_hall_data_button_clicked(self, s):
        try:
            if s == 1:
                data = dp.interpolate_data(dp.fit_data(self.hall_data, 10), np.linspace(
                    self.minT, self.maxT, int((self.maxT-self.minT)/20.)))
                self.scatter_hall_int.setData(data[:, 0], data[:, 1])
                self.interpolated_hall_data = data
                self.x_mask_hall = (self.interpolated_hall_data[:, 0] >= self.minT) & (
                    self.interpolated_hall_data[:, 0] <= self.maxT)
                
                #Assign the interpolated experimental data to all the scatters for the respective windows
                for key in self.experimental_data_scatter["hall"]:
                    self.experimental_data_scatter["hall"][key].setData(self.interpolated_hall_data[:, 0], self.interpolated_hall_data[:, 1])

                self.preview_hall_graph.addItem(self.scatter_hall_int)
                self.preview_hall_graph.removeItem(self.scatter_hall)

            elif s == 0:
                self.preview_hall_graph.addItem(self.scatter_hall)
                self.preview_hall_graph.removeItem(self.scatter_hall_int)
                self.x_mask_hall = (self.hall_data[:, 0] >= self.minT) & (
                    self.hall_data[:, 0] <= self.maxT)
                
                #Assign the non-interpolated experimental data to all the scatters for the respective windows
                for key in self.experimental_data_scatter["hall"]:
                    self.experimental_data_scatter["hall"][key].setData(self.hall_data[:, 0], self.hall_data[:, 1])
                
        except Exception as e:
            print("No data to interpolate:", "({})".format(e))
            self.initialized_widgets["int_hall_data_button"].setChecked(False)

    # Function for the smooth-plot button (higher x-values grid for the evaluation of the fit)
    def plot_button_clicked(self, s):
        self.x_val, self.y_val = TE.dpb_see_calc(self.minT, self.maxT, 50, [self.mass_slider.value(
        )/100., self.bandgap_slider.value(), self.fermi_slider.value()], float(self.Nv2_value.text())/float(self.Nv1_value.text()))
        self.manipulate_plot.setData(self.x_val, self.y_val, pen=self.pen2)
        self.manipulate_plot.setZValue(10)
        if self.experimental_data.isChecked():
            self.graph2.addItem(self.scatter_see_1PB_S)
        else:
            self.graph2.removeItem(self.scatter_see_1PB_S)

    '''
    Manipulate window changed
    '''
    def manipulate_value_changed_1PB_see(self):

        self.mass_lineedit_1PB.setText('{}'.format(float(self.mass_slider_1PB.value())/100.))
        self.fermi_lineedit_1PB.setText('{}'.format(self.fermi_slider_1PB.value()))
        #Check for experimental data
        if self.experimental_data_1PB.isChecked() and self.is_scatter2_there == False:
            self.graph_SPB.addItem(self.scatter_see_1PB_S)
            self.is_scatter2_there = True
        elif self.experimental_data_1PB.isChecked() == False and self.is_scatter2_there == True:
            self.graph_SPB.removeItem(self.scatter_see_1PB_S)
            self.is_scatter2_there = False
        #Recalculate the data
        self.x_val_1PB, self.y_val_1PB = TE.spb_see_calc(self.minT, self.maxT, 50, self.mass_slider_1PB.value()/100., self.fermi_slider_1PB.value())
        # Redraw regular plots
        self.manipulate_plot_1PB.setData(self.x_val_1PB, self.y_val_1PB, pen=self.pen2)
        self.manipulate_plot_1PB.setZValue(10)
        # Redraw the band structure sketch
        self.band1_data_1PB = TE.get_parabolic_band(-40, 40, 100, self.mass_slider_1PB.value()/100., 0)
        self.band3_data_1PB = TE.get_parabolic_band(-40, 40, 100, 100000000, self.fermi_slider_1PB.value())
        self.band1_1PB.setData(self.band1_data_1PB[0], self.band1_data_1PB[1], pen=self.pen_band1)
        self.band3_1PB.setData(self.band3_data_1PB[0], self.band3_data_1PB[1], pen=self.pen_band3)
        
    def manipulate_value_changed_1PB_res(self):
        self.mass_lineedit_1PB_res.setText(f"{float(self.mass_slider_1PB_res.value())/100.}")
        self.fermi_lineedit_1PB_res.setText(f"{self.fermi_slider_1PB_res.value()}")
        self.para_A_lineedit_1PB_res.setText(f"{self.para_A_slider_1PB_res.value()}")
        self.para_C_lineedit_1PB_res.setText(f"{self.para_C_slider_1PB_res.value()/100.}")
        self.para_F_lineedit_1PB_res.setText(f"{self.para_F_slider_1PB_res.value()}")

        if self.experimental_data_1PB_res.isChecked() and self.is_scatter2_there_1PB_res == False:
            self.graph_1PB_res.addItem(self.scatter_see_1PB_Srho)
            self.graph_1PB_rho_res.addItem(self.scatter_res_1PB_Srho)
            self.is_scatter2_there_1PB_res = True
        elif self.experimental_data_1PB_res.isChecked() == False and self.is_scatter2_there_1PB_res == True:
            self.graph_1PB_res.removeItem(self.scatter_see_1PB_Srho)
            self.graph_1PB_rho_res.removeItem(self.scatter_res_1PB_Srho)
            self.is_scatter2_there = False
            
        self.x_val_1PB, self.y_val_1PB, self.y_val_1PB_res = TE.spb_see_res_calc(
            self.minT, self.maxT, 50, self.mass_slider_1PB_res.value()/100., self.fermi_slider_1PB_res.value(), [self.para_A_slider_1PB_res.value(), self.para_C_slider_1PB_res.value(), self.para_F_slider_1PB_res.value()], f"{self.scattering_type_1PB_res.currentText()}")
        self.x_val_1PB_res = self.x_val_1PB
        # Redraw regular plots
        self.manipulate_plot_1PB_res.setData(self.x_val_1PB, self.y_val_1PB, pen=self.pen2)
        self.manipulate_plot_1PB_rho_res.setData(self.x_val_1PB_res, self.y_val_1PB_res, pen=self.pen2)
        self.manipulate_plot_1PB_res.setZValue(10)
        self.manipulate_plot_1PB_rho_res.setZValue(10)
        # Redraw the band structure sketch
        self.band1_data_1PB_res = TE.get_parabolic_band(-40, 40, 100, self.mass_slider_1PB_res.value()/100., 0)
        self.band3_data_1PB_res = TE.get_parabolic_band(-40, 40, 100, 100000000, self.fermi_slider_1PB_res.value())
        self.band1_1PB_res.setData(self.band1_data_1PB_res[0],  self.band1_data_1PB_res[1], pen=self.pen_band1)
        self.band3_1PB_res.setData(self.band3_data_1PB_res[0],  self.band3_data_1PB_res[1], pen=self.pen_band3)
        
    def manipulate_value_changed_1PB_hall(self):
        self.mass_lineedit_1PB_hall.setText(f"{float(self.mass_slider_1PB_hall.value())/100.}")
        self.fermi_lineedit_1PB_hall.setText(f"{self.fermi_slider_1PB_hall.value()}")
        self.para_A_lineedit_1PB_hall.setText(f"{self.para_A_slider_1PB_hall.value()}")
        self.para_C_lineedit_1PB_hall.setText(f"{self.para_C_slider_1PB_hall.value()/100.}")
        self.para_F_lineedit_1PB_hall.setText(f"{self.para_F_slider_1PB_hall.value()}")

        if self.experimental_data_1PB_hall.isChecked() and self.is_scatter2_there_1PB_hall == False:
            self.graph_1PB_hall.addItem(self.scatter_see_1PB_SrhoHall)
            self.graph_1PB_rho_hall.addItem(self.scatter_res_1PB_SrhoHall)
            self.graph_1PB_Hall_hall.addItem(self.scatter_hall_1PB_SrhoHall)
            self.is_scatter2_there_1PB_hall = True
        elif self.experimental_data_1PB_hall.isChecked() == False and self.is_scatter2_there_1PB_hall == True:
            self.graph_1PB_hall.removeItem(self.scatter_see_1PB_SrhoHall)
            self.graph_1PB_rho_hall.removeItem(self.scatter_res_1PB_SrhoHall)
            self.graph_1PB_Hall_hall.removeItem(self.scatter_hall_1PB_SrhoHall)
            self.is_scatter2_there = False
            
        self.x_val_1PB, self.y_val_1PB, self.y_val_1PB_res, self.y_val_1PB_hall = TE.spb_see_res_hall_calc(
            self.minT, self.maxT, 50, self.mass_slider_1PB_hall.value()/100., self.fermi_slider_1PB_hall.value(), [self.para_A_slider_1PB_hall.value(), self.para_C_slider_1PB_hall.value()*300, self.para_F_slider_1PB_hall.value()], f"{self.scattering_type_1PB_hall.currentText()}")
        self.x_val_1PB_res, self.x_val_1PB_hall = self.x_val_1PB, self.x_val_1PB
        # Redraw regular plots
        self.manipulate_plot_1PB_hall.setData(self.x_val_1PB, self.y_val_1PB, pen=self.pen2)
        self.manipulate_plot_1PB_rho_hall.setData(self.x_val_1PB_res, self.y_val_1PB_res, pen=self.pen2)
        self.manipulate_plot_1PB_Hall_hall.setData(self.x_val_1PB_hall, self.y_val_1PB_hall, pen=self.pen2)
        self.manipulate_plot_1PB_res.setZValue(10)
        self.manipulate_plot_1PB_rho_res.setZValue(10)
        self.manipulate_plot_1PB_Hall_hall.setZValue(10)
        # Redraw the band structure sketch
        self.band1_data_1PB_hall = TE.get_parabolic_band(-40, 40, 100, self.mass_slider_1PB_hall.value()/100., 0)
        self.band3_data_1PB_hall = TE.get_parabolic_band(-40, 40, 100, 100000000, self.fermi_slider_1PB_hall.value())
        self.band1_1PB_hall.setData(self.band1_data_1PB_hall[0],  self.band1_data_1PB_hall[1], pen=self.pen_band1)
        self.band3_1PB_hall.setData(self.band3_data_1PB_hall[0],  self.band3_data_1PB_hall[1], pen=self.pen_band3)

    # Function to redraw the manipulate plot, when the slider values are changed (2PB window)
    def manipulate_value_changed_2PB_see(self):
        #Set lineedits to the according values of the sliders
        self.mass_lineedit.setText("{}".format(self.mass_slider.value()/100.))
        self.bandgap_lineedit.setText("{}".format(self.bandgap_slider.value()))
        self.fermi_lineedit.setText("{}".format(self.fermi_slider.value()))

        #Calculate the transport properties from the given parameters
        self.x_val, self.y_val = TE.dpb_see_calc(self.minT, self.maxT, 50, [self.mass_slider.value(
                )/100., self.bandgap_slider.value(), self.fermi_slider.value()], float(self.Nv2_value.text())/float(self.Nv1_value.text()))

        #Calculate the effective band structure and set the data to the plot
        self.band2_data = TE.get_parabolic_band(-40, 40, 100, self.mass_slider.value()/100./(
            int(self.Nv1_value.text())/int(self.Nv2_value.text())), self.bandgap_slider.value())
        self.band3_data = TE.get_parabolic_band(
            -40, 40, 100, 100000000, self.fermi_slider.value())
        self.band2.setData(
            self.band2_data[0], self.band2_data[1], pen=self.pen_band2)
        self.band3.setData(
            self.band3_data[0], self.band3_data[1], pen=self.pen_band3)
        
        #Set the transport data to the plots
        self.manipulate_plot.setData(self.x_val, self.y_val, pen=self.pen2)
        # self.ind_man_plot.setData(self.x_val, self.y_val, pen=self.pen_ind_tot)
        self.manipulate_plot.setZValue(10)

        #Check if experimental data button is on or off and show or remove the experimental data scatter plot
        if self.experimental_data.isChecked() and self.is_scatter2_there == False:
            self.graph2.addItem(self.scatter_see_2PB_S)
            self.is_scatter2_there = True
        elif self.experimental_data.isChecked() == False and self.is_scatter2_there == True:
            self.graph2.removeItem(self.scatter_see_2PB_S)
            self.is_scatter2_there = False

    def manipulate_value_changed_res(self):
        #Set lineedits to the according values of the sliders
        self.para_A_lineedit_res.setText("{}".format(self.para_A_slider_res.value()))
        self.para_B_lineedit_res.setText("{}".format(self.para_B_slider_res.value()/100.))
        self.para_C_lineedit_res.setText("{}".format(self.para_C_slider_res.value()/100.))
        self.para_D_lineedit_res.setText("{}".format(self.para_D_slider_res.value()/100.))  
        self.para_F_lineedit_res.setText("{}".format(self.para_F_slider_res.value()))  
        self.para_G_lineedit_res.setText("{}".format(self.para_G_slider_res.value()/100.))  
        self.mass_lineedit_res.setText("{}".format(self.mass_slider_res.value()/100.))
        self.bandgap_lineedit_res.setText("{}".format(self.bandgap_slider_res.value()))
        self.fermi_lineedit_res.setText("{}".format(self.fermi_slider_res.value()))
        self.mass_lineedit_res.setText("{}".format(self.mass_slider_res.value()/100.))
        self.bandgap_lineedit_res.setText("{}".format(self.bandgap_slider_res.value()))
        self.fermi_lineedit_res.setText("{}".format(self.fermi_slider_res.value()))
        
        #Calculate the transport properties from the given parameters
        self.x_val,self.y_val,self.y_val_res = TE.dpb_see_res_calc(self.minT, self.maxT, 50, [self.mass_slider_res.value(
                                                                    )/100., self.bandgap_slider_res.value(), self.fermi_slider_res.value()], [self.para_A_slider_res.value(), self.para_B_slider_res.value()/100.,
                                                                    self.para_C_slider_res.value()/100., self.para_D_slider_res.value()/100., self.para_F_slider_res.value(), self.para_G_slider_res.value()/100.], 
                                                                    int(self.Nv2_value_res.text())/int(self.Nv1_value_res.text()), scatter_type = f"{self.scattering_type_res.currentText()}")
        self.x_val_res = self.x_val
        
        #Calculate the effective band structure and set the data to the plot
        self.band2_data = TE.get_parabolic_band(-40, 40, 100, self.mass_slider_res.value()/100./(
            int(self.Nv1_value_res.text())/int(self.Nv2_value_res.text())), self.bandgap_slider_res.value())
        self.band3_data = TE.get_parabolic_band(
            -40, 40, 100, 100000000, self.fermi_slider_res.value())
        self.band2_res.setData(
            self.band2_data[0], self.band2_data[1], pen=self.pen_band2)
        self.band3_res.setData(
            self.band3_data[0], self.band3_data[1], pen=self.pen_band3)
        
        #Set the transport data to the plots
        self.manipulate_plot_res.setData(self.x_val, self.y_val, pen=self.pen2)
        self.manipulate_plot_res_res.setData(self.x_val_res, self.y_val_res, pen=self.pen2)
        self.manipulate_plot_res.setZValue(10)
        
        #Check if experimental data button is on or off and show or remove the experimental data scatter plot
        if self.experimental_data_res.isChecked() and self.is_scatter2_there_res==False:
            self.graph2_res.addItem(self.scatter_see_2PB_Srho)
            self.graph_res.addItem(self.scatter_res_2PB_Srho)
            self.is_scatter2_there_res=True
        elif self.experimental_data_res.isChecked()==False and self.is_scatter2_there_res==True:
            self.graph2_res.removeItem(self.scatter_see_2PB_Srho)
            self.graph_res.removeItem(self.scatter_res_2PB_Srho)
            self.is_scatter2_there_res=False
             
    def manipulate_value_changed_hall(self):
        #Set lineedits to the according values of the sliders
        self.mass_lineedit_hall.setText("{}".format(self.mass_slider_hall.value()/100.))
        self.bandgap_lineedit_hall.setText("{}".format(self.bandgap_slider_hall.value()))
        self.fermi_lineedit_hall.setText("{}".format(self.fermi_slider_hall.value()))
        self.para_A_lineedit_hall.setText("{}".format(self.para_A_slider_hall.value()))
        self.para_B_lineedit_hall.setText("{}".format(self.para_B_slider_hall.value()/100.))
        self.para_C_lineedit_hall.setText("{}".format(self.para_C_slider_hall.value()/100.))
        self.para_D_lineedit_hall.setText("{}".format(self.para_D_slider_hall.value()/100.))
        self.para_E_lineedit_hall.setText("{}".format(self.para_E_slider_hall.value()/100.))
        self.para_F_lineedit_hall.setText("{}".format(self.para_F_slider_hall.value()))
        self.para_G_lineedit_hall.setText("{}".format(self.para_G_slider_hall.value()/100.))
        
        #Calculate the transport properties from the given parameters
        self.x_val,self.y_val,self.y_val_res, self.y_val_hall = TE.dpb_see_res_hall_calc(self.minT, self.maxT, 50, [self.mass_slider_hall.value(
        )/100., self.bandgap_slider_hall.value(), self.fermi_slider_hall.value()], [self.para_A_slider_hall.value(), self.para_B_slider_hall.value()/100.,self.para_C_slider_hall.value()/100., 
                                                                                  self.para_D_slider_hall.value()/100., self.para_F_slider_hall.value(), self.para_G_slider_hall.value()/100.], self.para_E_slider_hall.value()/100., int(self.Nv2_value_hall.text())/int(self.Nv1_value_hall.text())
                                                                                  , scatter_type = f"{self.scattering_type_hall.currentText()}")
        self.x_val_res, self.x_val_hall = self.x_val, self.x_val
        
        #Calculate the effective band structure and set the data to the plot
        self.band2_data = TE.get_parabolic_band(-40, 40, 100, self.mass_slider_hall.value()/100./(
            int(self.Nv1_value_hall.text())/int(self.Nv2_value_hall.text())), self.bandgap_slider_hall.value())
        self.band3_data = TE.get_parabolic_band(
            -40, 40, 100, 100000000, self.fermi_slider_hall.value())
        self.band2_hall.setData(
            self.band2_data[0], self.band2_data[1], pen=self.pen_band2)
        self.band3_hall.setData(
            self.band3_data[0], self.band3_data[1], pen=self.pen_band3)
                
        #Set the transport data to the plots
        self.manipulate_plot_hall.setData(self.x_val, self.y_val, pen=self.pen2)
        self.manipulate_plot_res_hall.setData(self.x_val_res, self.y_val_res, pen=self.pen2)
        self.manipulate_plot_hall_hall.setData(self.x_val_hall, self.y_val_hall, pen=self.pen2)
        self.manipulate_plot_hall.setZValue(10)

        #Check if experimental data button is on or off and show or remove the experimental data scatter plot
        if self.experimental_data_hall.isChecked() and self.is_scatter2_there_hall == False:
            self.graph2_hall.addItem(self.scatter_see_2PB_SrhoHall)
            self.graph_res_hall.addItem(self.scatter_res_2PB_SrhoHall)
            self.graph_hall.addItem(self.scatter_hall_2PB_SrhoHall)
            self.is_scatter2_there_hall=True
        elif self.experimental_data_hall.isChecked() == False and self.is_scatter2_there_hall == True:
            self.graph2_hall.removeItem(self.scatter_see_2PB_SrhoHall)
            self.graph_res_hall.removeItem(self.scatter_res_2PB_SrhoHall)
            self.graph_hall.removeItem(self.scatter_hall_2PB_SrhoHall)
            self.is_scatter2_there_hall=False
            
            
    def line_edit_changed_1PB(self):
        #Set sliders to the according values of the lineedits
        self.mass_slider_1PB.setValue(int(float(self.mass_lineedit_1PB.text())*100))
        self.mass_lineedit_1PB.setText("{}".format(self.mass_slider_1PB.value()/100.))
        self.fermi_slider_1PB.setValue(int(self.fermi_lineedit_1PB.text()))
        self.fermi_lineedit_1PB.setText("{}".format(self.fermi_slider_1PB.value()))
        #Calculate the transport properties from the given parameters
        self.x_val_1PB, self.y_val_1PB = TE.spb_see_calc(self.minT, self.maxT, 50, self.mass_slider_1PB.value()/100., self.fermi_slider_1PB.value())
        # Redraw regular plots
        self.manipulate_plot_1PB.setData(self.x_val_1PB, self.y_val_1PB, pen=self.pen2)
        self.manipulate_plot_1PB.setZValue(10)
        # Redraw the band structure sketch
        self.band1_data = TE.get_parabolic_band(-40, 40, 100, self.mass_slider_1PB.value()/100., 0)
        self.band3_data = TE.get_parabolic_band(-40, 40, 100, 100000000, self.fermi_slider_1PB.value())
        self.band1_1PB.setData(self.band1_data[0],  self.band1_data[1], pen=self.pen_band1)
        self.band3_1PB.setData(self.band3_data[0],  self.band3_data[1], pen=self.pen_band3)       
        
    def line_edit_changed_1PB_res(self):
        #Set sliders to the according values of the lineedits
        self.mass_slider_1PB_res.setValue(int(float(self.mass_lineedit_1PB_res.text())*100))
        self.mass_lineedit_1PB_res.setText("{}".format(self.mass_slider_1PB_res.value()/100.))

        self.fermi_slider_1PB_res.setValue(int(self.fermi_lineedit_1PB_res.text()))
        self.fermi_lineedit_1PB_res.setText("{}".format(self.fermi_slider_1PB_res.value()))
        self.para_A_slider_1PB_res.setValue(int(self.para_A_lineedit_1PB_res.text()))
        self.para_C_slider_1PB_res.setValue(int(float(self.para_C_lineedit_1PB_res.text())*100.))
        self.para_F_slider_1PB_res.setValue(int(self.para_F_lineedit_1PB_res.text()))
        
        #Calculate the transport properties from the given parameters
        self.x_val_1PB, self.y_val_1PB, self.y_val_1PB_res = TE.spb_see_res_calc(self.minT, self.maxT, 50, self.mass_slider_1PB_res.value()/100., self.fermi_slider_1PB_res.value(), [
            self.para_A_slider_1PB_res.value(), self.para_C_slider_1PB_res.value(), self.para_F_slider_1PB_res.value()], f"{self.scattering_type_1PB_res.currentText()}")
        self.x_val_1PB_res = self.x_val_1PB
        # Redraw regular plots
        self.manipulate_plot_1PB_res.setData(self.x_val_1PB, self.y_val_1PB, pen=self.pen2)
        self.manipulate_plot_1PB_rho_res.setData(self.x_val_1PB_res, self.y_val_1PB_res, pen=self.pen2)
        self.manipulate_plot_1PB_res.setZValue(10)
        self.manipulate_plot_1PB_rho_res.setZValue(10)
        # Redraw the band structure sketch
        self.band1_data_1PB_res = TE.get_parabolic_band(-40, 40, 100, self.mass_slider_1PB_res.value()/100., 0)
        self.band3_data_1PB_res = TE.get_parabolic_band(-40, 40, 100, 100000000, self.fermi_slider_1PB_res.value())
        self.band1_1PB_res.setData(self.band1_data_1PB_res[0],  self.band1_data_1PB_res[1], pen=self.pen_band1)
        self.band3_1PB_res.setData(self.band3_data_1PB_res[0],  self.band3_data_1PB_res[1], pen=self.pen_band3)       
       
    def line_edit_changed_1PB_hall(self):
        #Set sliders to the according values of the lineedits
        self.mass_slider_1PB_hall.setValue(int(float(self.mass_lineedit_1PB_hall.text())*100))
        self.mass_lineedit_1PB_hall.setText("{}".format(self.mass_slider_1PB_hall.value()/100.))

        self.fermi_slider_1PB_hall.setValue(int(self.fermi_lineedit_1PB_hall.text()))
        self.fermi_lineedit_1PB_hall.setText("{}".format(self.fermi_slider_1PB_hall.value()))
        self.para_A_slider_1PB_hall.setValue(int(self.para_A_lineedit_1PB_hall.text()))
        self.para_C_slider_1PB_hall.setValue(int(float(self.para_C_lineedit_1PB_hall.text())*100.))
        self.para_F_slider_1PB_hall.setValue(int(self.para_F_lineedit_1PB_hall.text()))
        
        #Calculate the transport properties from the given parameters
        self.x_val_1PB, self.y_val_1PB, self.y_val_1PB_res, self.y_val_1PB_hall = TE.spb_see_res_hall_calc(self.minT, self.maxT, 50, self.mass_slider_1PB_hall.value()/100., self.fermi_slider_1PB_hall.value(), [
            self.para_A_slider_1PB_hall.value(), self.para_C_slider_1PB_hall.value(), self.para_F_slider_1PB_hall.value()], f"{self.scattering_type_1PB_hall.currentText()}")
        self.x_val_1PB_res, self.x_val_1PB_hall = self.x_val_1PB, self.x_val_1PB
        # Redraw regular plots
        self.manipulate_plot_1PB_hall.setData(self.x_val_1PB, self.y_val_1PB, pen=self.pen2)
        self.manipulate_plot_1PB_rho_hall.setData(self.x_val_1PB_res, self.y_val_1PB_res, pen=self.pen2)
        self.manipulate_plot_1PB_Hall_hall.setData(self.x_val_1PB_hall, self.y_val_1PB_hall, pen=self.pen2)
        self.manipulate_plot_1PB_hall.setZValue(10)
        self.manipulate_plot_1PB_rho_hall.setZValue(10)
        self.manipulate_plot_1PB_Hall_hall.setZValue(10)
        # Redraw the band structure sketch
        self.band1_data_1PB_hall = TE.get_parabolic_band(-40, 40, 100, self.mass_slider_1PB_hall.value()/100., 0)
        self.band3_data_1PB_hall = TE.get_parabolic_band(-40, 40, 100, 100000000, self.fermi_slider_1PB_hall.value())
        self.band1_1PB_hall.setData(self.band1_data_1PB_hall[0],  self.band1_data_1PB_hall[1], pen=self.pen_band1)
        self.band3_1PB_hall.setData(self.band3_data_1PB_hall[0],  self.band3_data_1PB_hall[1], pen=self.pen_band3)       
    
    def line_edit_changed(self):
        #Set sliders to the according values of the lineedits
        self.mass_slider.setValue(int(float(self.mass_lineedit.text())*100))
        self.mass_lineedit.setText("{}".format(self.mass_slider.value()/100.))
        self.bandgap_slider.setValue(int(self.bandgap_lineedit.text()))
        self.bandgap_lineedit.setText("{}".format(self.bandgap_slider.value()))
        self.fermi_slider.setValue(int(self.fermi_lineedit.text()))
        self.fermi_lineedit.setText("{}".format(self.fermi_slider.value()))
        
        #Calculate the transport properties from the given parameters
        self.x_val, self.y_val = TE.dpb_see_calc(self.minT, self.maxT, 50, [self.mass_slider.value(
                )/100., self.bandgap_slider.value(), self.fermi_slider.value()], float(self.Nv2_value.text())/float(self.Nv1_value.text()))
        

        #Calculate the effective band structure and set the data to the plot
        self.band2_data = TE.get_parabolic_band(-40, 40, 100, self.mass_slider.value()/100./(
            int(self.Nv1_value.text())/int(self.Nv2_value.text())), self.bandgap_slider.value())
        self.band3_data = TE.get_parabolic_band(
            -40, 40, 100, 100000000, self.fermi_slider.value())
        self.band2.setData(
            self.band2_data[0], self.band2_data[1], pen=self.pen_band2)
        self.band3.setData(
            self.band3_data[0], self.band3_data[1], pen=self.pen_band3)
                
        #Set the transport data to the plots
        self.manipulate_plot.setData(self.x_val, self.y_val, pen=self.pen2)
        self.manipulate_plot.setZValue(10)
        
    def line_edit_changed_res(self):
        #Set sliders to the according values of the lineedits
        self.mass_slider_res.setValue(int(float(self.mass_lineedit_res.text())*100.))
        self.bandgap_slider_res.setValue(int(self.bandgap_lineedit_res.text()))
        self.fermi_slider_res.setValue(int(self.fermi_lineedit_res.text()))
        self.para_A_slider_res.setValue(int(float(self.para_A_lineedit_res.text())))
        self.para_B_slider_res.setValue(int(float(self.para_B_lineedit_res.text())*100))
        self.para_C_slider_res.setValue(int(float(self.para_C_lineedit_res.text())))
        self.para_D_slider_res.setValue(int(float(self.para_D_lineedit_res.text())))
        self.para_F_slider_res.setValue(int(float(self.para_F_lineedit_res.text())))
        self.para_G_slider_res.setValue(int(float(self.para_G_lineedit_res.text())*100))
        
        #Calculate the transport properties from the given parameters
        self.x_val,self.y_val,self.y_val_res = TE.dpb_see_res_calc(self.minT, self.maxT, 50, [self.mass_slider_res.value(
        )/100., self.bandgap_slider_res.value(), self.fermi_slider_res.value()], [self.para_A_slider_res.value(), self.para_B_slider_res.value()/100.,self.para_C_slider_res.value(), 
                                                                                  self.para_D_slider_res.value(), self.para_F_slider_res.value(), self.para_G_slider_res.value()/100.], int(self.Nv2_value_res.text())/int(self.Nv1_value_res.text()), 
                                                                                  scatter_type = f"{self.scattering_type_res.currentText()}")                                                           
        self.x_val_res = self.x_val
        
        #Calculate the effective band structure and set the data to the plot
        self.band2_data = TE.get_parabolic_band(-40, 40, 100, self.mass_slider_res.value()/100./(
            int(self.Nv1_value_res.text())/int(self.Nv2_value_res.text())), self.bandgap_slider_res.value())
        self.band3_data = TE.get_parabolic_band(
            -40, 40, 100, 100000000, self.fermi_slider_res.value())  
        self.band2_res.setData(
            self.band2_data[0], self.band2_data[1], pen=self.pen_band2)
        self.band3_res.setData(
            self.band3_data[0], self.band3_data[1], pen=self.pen_band3)
        
        #Set the transport data to the plots
        self.manipulate_plot_res.setData(self.x_val, self.y_val, pen=self.pen2)
        self.manipulate_plot_res_res.setData(self.x_val_res, self.y_val_res, pen=self.pen2)
        self.manipulate_plot_res.setZValue(10)
        self.manipulate_plot_res_res.setZValue(10)

    def line_edit_changed_hall(self):            
        #Set sliders to the according values of the lineedits
        self.mass_slider_hall.setValue(int(float(self.mass_lineedit_hall.text())*100.))
        self.bandgap_slider_hall.setValue(int(self.bandgap_lineedit_hall.text()))
        self.fermi_slider_hall.setValue(int(self.fermi_lineedit_hall.text()))
        self.para_A_slider_hall.setValue(int(float(self.para_A_lineedit_hall.text())))
        self.para_B_slider_hall.setValue(int(float(self.para_B_lineedit_hall.text())*100))
        self.para_C_slider_hall.setValue(int(float(self.para_C_lineedit_hall.text())))
        self.para_D_slider_hall.setValue(int(float(self.para_D_lineedit_hall.text())))
        self.para_E_slider_hall.setValue(int(float(self.para_E_lineedit_hall.text())*100.))
        self.para_F_slider_hall.setValue(int(float(self.para_F_lineedit_hall.text())))
        self.para_G_slider_hall.setValue(int(float(self.para_G_lineedit_hall.text())*100.))
        
        #Calculate the effective band structure and set the data to the plot
        self.x_val,self.y_val,self.y_val_res, self.y_val_hall = TE.dpb_see_res_hall_calc(self.minT, self.maxT, 50, [self.mass_slider_hall.value(
        )/100., self.bandgap_slider_hall.value(), self.fermi_slider_hall.value()], [self.para_A_slider_hall.value(), self.para_B_slider_hall.value()/100.,self.para_C_slider_hall.value(), 
                                                                                  self.para_D_slider_hall.value(), self.para_F_slider_hall.value(), self.para_G_slider_hall.value()/100.], self.para_E_slider_hall.value()/100.,
                                                                                  int(self.Nv2_value_hall.text())/int(self.Nv1_value_hall.text()), scatter_type = f"{self.scattering_type_hall.currentText()}")
        self.x_val_res, self.x_val_hall = self.x_val, self.x_val
        
        #Calculate the effective band structure and set the data to the plot
        self.band2_data = TE.get_parabolic_band(-40, 40, 100, self.mass_slider_hall.value()/100./(
            int(self.Nv1_value_hall.text())/int(self.Nv2_value_hall.text())), self.bandgap_slider_hall.value())
        self.band3_data = TE.get_parabolic_band(
            -40, 40, 100, 100000000, self.fermi_slider_hall.value())
        self.band2_hall.setData(
            self.band2_data[0], self.band2_data[1], pen=self.pen_band2)
        self.band3_hall.setData(
            self.band3_data[0], self.band3_data[1], pen=self.pen_band3)
        
        #Set the transport data to the plots
        self.manipulate_plot_hall.setData(self.x_val, self.y_val, pen=self.pen2)
        self.manipulate_plot_res_hall.setData(self.x_val_res, self.y_val_res, pen=self.pen2)
        self.manipulate_plot_hall_hall.setData(self.x_val_hall, self.y_val_hall, pen=self.pen2)
        self.manipulate_plot_hall.setZValue(10)
        self.manipulate_plot_res_hall.setZValue(10)
        self.manipulate_plot_hall_hall.setZValue(10)


    # Function to execute the fit using the parameters from the manipulate sliders + plotting it (1PB)
    def manip_fit_button_clicked_1PB_see(self):
        self.start_waiting_gif(self.gif_label_spb_see)
        
        self.computation_thread = FittingThread_1PB_See(self)
        self.computation_thread.updateGUI.connect(self.update_fit_results_1PB_see)
        self.computation_thread.error_occurred.connect(self.error_stop_waiting_gif)
        self.computation_thread.start()
    
    def manip_fit_button_clicked_1PB_res(self):
        self.start_waiting_gif(self.gif_label_spb_res)
             
        self.computation_thread = FittingThread_1PB_Res(self)
        self.computation_thread.updateGUI.connect(self.update_fit_results_1PB_res)
        self.computation_thread.error_occurred.connect(self.error_stop_waiting_gif)
        self.computation_thread.start()
            
    def manip_fit_button_clicked_1PB_hall(self):
        self.start_waiting_gif(self.gif_label_spb_hall)
             
        self.computation_thread = FittingThread_1PB_Hall(self)
        self.computation_thread.updateGUI.connect(self.update_fit_results_1PB_hall)
        self.computation_thread.error_occurred.connect(self.error_stop_waiting_gif)
        self.computation_thread.start()

    # Function to execute the fit using the parameters from the manipulate sliders + plotting it (2PB)
    def manip_fit_button_clicked_2PB_see(self):
        self.start_waiting_gif(self.gif_label_dpb_see)
            
        self.computation_thread = FittingThread_2PB_See(self)
        self.computation_thread.updateGUI.connect(self.update_fit_results_2PB_see)
        self.computation_thread.error_occurred.connect(self.error_stop_waiting_gif)
        self.computation_thread.start()
             
    def manip_fit_button_clicked_2PB_res(self):
        self.start_waiting_gif(self.gif_label_dpb_res)

        self.computation_thread = FittingThread_2PB_Res(self)
        self.computation_thread.updateGUI.connect(self.update_fit_results_2PB_res)
        self.computation_thread.error_occurred.connect(self.error_stop_waiting_gif)
        self.computation_thread.start()
          
    def manip_fit_button_clicked_2PB_hall(self):
        self.start_waiting_gif(self.gif_label_dpb_hall)
             
        self.computation_thread = FittingThread_2PB_Hall(self)
        self.computation_thread.updateGUI.connect(self.update_fit_results_2PB_hall)
        self.computation_thread.error_occurred.connect(self.error_stop_waiting_gif)
        self.computation_thread.start()
        
    #Functions for the radio buttons for the experimental data
    def experimental_data_toggled_1PB(self, s):
        self.manipulate_value_changed_1PB_see()
    
    def experimental_data_toggled_1PB_res(self, s):
        self.manipulate_value_changed_1PB_res()
        
    def experimental_data_toggled_1PB_hall(self, s):
        self.manipulate_value_changed_1PB_hall()

    def experimental_data_toggled(self, s):
        self.manipulate_value_changed_2PB_see()

    def experimental_data_toggled_res(self, s):
        self.manipulate_value_changed_res()
        
    def experimental_data_toggled_hall(self, s):
        self.manipulate_value_changed_hall()

    #Functions for the "Adjust to fit" buttons 
    def fitpar_button_clicked(self):
        self.mass_slider.setValue(
            int(float(self.mass_fit_label.text().strip("mass = "))*100))
        self.bandgap_slider.setValue(
            int(self.bandgap_fit_label.text().strip("gap = ")))
        self.fermi_slider.setValue(
            int(self.fermi_fit_label.text().strip("Fermi energy = ")))
        self.manipulate_value_changed_2PB_see()
        
    def fitpar_button_clicked_res(self):
        self.mass_slider_res.setValue(
            int(float(self.mass_fit_label_res.text().strip("mass = "))*100))
        self.bandgap_slider_res.setValue(
            int(self.bandgap_fit_label_res.text().strip("gap = ")))
        self.fermi_slider_res.setValue(
            int(self.fermi_fit_label_res.text().strip("Fermi energy = ")))
        self.para_A_slider_res.setValue(
            int(float(self.para_A_fit_label_res.text().strip("&tau;<sub>ph,1</sub> = "))))
        self.para_B_slider_res.setValue(
            int(float(self.para_B_fit_label_res.text().strip("&tau;<sub>ph,2</sub>/&tau;<sub>ph,1</sub> = "))*100))
        self.para_C_slider_res.setValue(
            int(float(self.para_C_fit_label_res.text().replace("&tau;<sub>ph,1</sub>/&tau;<sub>dis,1</sub> @ 300 K = ",""))*100.))
        self.para_D_slider_res.setValue(
            int(float(self.para_D_fit_label_res.text().replace("&tau;<sub>ph,2</sub>/&tau;<sub>dis,2</sub> @ 300 K = ",""))*100.))
        self.para_F_slider_res.setValue(
            int(float(self.para_F_fit_label_res.text().strip("&tau;<sub>dis,1</sub> = "))))
        self.para_G_slider_res.setValue(
            int(float(self.para_G_fit_label_res.text().strip("<p style='font-size: 14px;'>&tau;<sub>dis,2</sub>/ &tau;<sub>dis,1</sub> =</p>"))))
        self.manipulate_value_changed_res()

    def fitpar_button_clicked_hall(self):
        self.mass_slider_hall.setValue(
            int(float(self.mass_fit_label_hall.text().strip("mass = "))*100.))
        self.bandgap_slider_hall.setValue(
            int(self.bandgap_fit_label_hall.text().strip("gap = ")))
        self.fermi_slider_hall.setValue(
            int(self.fermi_fit_label_hall.text().strip("Fermi energy = ")))
        self.para_A_slider_hall.setValue(
            int(float(self.para_A_fit_label_hall.text().strip("&tau;<sub>ph,1</sub> = "))))
        self.para_B_slider_hall.setValue(
            int(float(self.para_B_fit_label_hall.text().strip("&tau;<sub>ph,2</sub>/&tau;<sub>ph,1</sub> = "))*100.))
        self.para_C_slider_hall.setValue(
            int(float(self.para_C_fit_label_hall.text().replace("&tau;<sub>ph,1</sub>/&tau;<sub>dis,1</sub> @ 300 K = ",""))*100.))
        self.para_D_slider_hall.setValue(
            int(float(self.para_D_fit_label_hall.text().replace("&tau;<sub>ph,2</sub>/&tau;<sub>dis,2</sub> @ 300 K = ",""))*100.))
        self.para_E_slider_hall.setValue(
            int(float(self.para_E_fit_label_hall.text().strip("m<sub>1</sub> [m<sub>e</sub>] =  "))*100.))
        self.para_F_slider_hall.setValue(
            int(float(self.para_F_fit_label_hall.text().strip("&tau;<sub>dis,1</sub> = "))))
        self.para_G_slider_hall.setValue(
            int(float(self.para_G_fit_label_hall.text().strip("<p style='font-size: 14px;'>&tau;<sub>dis,2</sub>/ &tau;<sub>dis,1</sub> =</p>"))*100.))
        self.manipulate_value_changed_hall()
    
    #Individual advanced options windows
    def adv_opt_button_clicked_1PB(self):
        self.newWindow = SecondWindow_1PB_See(self)
        
    def adv_opt_button_clicked_1PB_res(self):
        self.newWindow = SecondWindow_1PB_Res(self)
        
    def adv_opt_button_clicked_1PB_hall(self):
        self.newWindow = SecondWindow_1PB_Hall(self)
    
    def adv_opt_button_clicked(self):
        self.newWindow = SecondWindow_2PB_See(self)
        
    def adv_opt_button_clicked_res(self):
        self.newWindow = SecondWindow_2PB_Res(self)
            
    def adv_opt_button_clicked_hall(self):
        self.newWindow = SecondWindow_2PB_Hall(self)
    
    #Individual individual contribution windows   
    def ind_cont_button_clicked_1PB(self):
        self.indContWindow = AddGraphs_Window_1PB_See(self)
    
    def ind_cont_button_clicked_1PB_res(self):
        self.indContWindow = AddGraphs_Window_1PB_Res(self)
        
    def ind_cont_button_clicked_1PB_hall(self):
        self.indContWindow = AddGraphs_Window_1PB_Hall(self)
    
    def ind_cont_button_clicked(self):
        self.indContWindow = AddGraphs_Window_2PB_See(self)
        
    def ind_cont_button_clicked_res(self):
        self.indContWindow = AddGraphs_Window_2PB_Res(self)
        
    def ind_cont_button_clicked_hall(self):
        self.indContWindow_hall = AddGraphs_Window_2PB_Hall(self)
        
    #print data functions
    def print_data_button_clicked_see_1PB(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "all_data.txt", filter="Text file (*.txt);;All files (*.*)")
        
        # Check if saving was aborted
        if(path == ""):
            return
        data = []
        y_legends = []
        
        # Add experimental data
        if (self.is_scatter2_there):
            data.extend((self.scatter_see_1PB_S.getData()[0], self.scatter_see_1PB_S.getData()[1]))
            y_legends.extend(("Temperature [K]", "Experimental Seebeck coefficient [V/K]"))
        
        # Add pre-fit curve    
        data.extend(TE.spb_see_calc(self.minT, self.maxT, 50, self.mass_slider_1PB.value()/100., self.fermi_slider_1PB.value()))
        y_legends.extend(("Temperature [K]", "Pre-fit Seebeck coefficient [V/K]"))
        
        # Add fit curve
        if (hasattr(self, 'params_1PB')):
            data.extend(TE.spb_see_calc(self.minT, self.maxT, 50, self.mass_slider_1PB.value()/100., self.params_1PB[0]))
            y_legends.extend(("Temperature [K]", "Fit Seebeck coefficient [V/K]"))
        
        # Save all data
        dp.save_data(path = path, data = data, x_legend = "", y_legends = y_legends)
        
    def print_data_button_clicked_see_res_1PB(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "all_data.txt", filter="Text file (*.txt);;All files (*.*)")
        
        # Check if saving was aborted
        if(path == ""):
            return
        data = []
        y_legends = []
        
        
        x_vals, y1_vals, y2_vals = TE.spb_see_res_calc(self.minT, self.maxT, 50, self.mass_slider_1PB_res.value()/100., self.fermi_slider_1PB_res.value(), [self.para_A_slider_1PB_res.value(), self.para_C_slider_1PB_res.value()/100.,self.para_F_slider_1PB_res.value(),], f"{self.scattering_type_1PB_res.currentText()}")
        
        if (hasattr(self, 'params_1PB_res')):
            x_vals_fit, y1_vals_fit, y2_vals_fit = TE.spb_see_res_calc(self.minT, self.maxT, 50, self.mass_slider_1PB_res.value()/100., self.params_1PB_res[0], self.params_1PB_res[1:4], scatter_type = f"{self.scattering_type_1PB_res.currentText()}")
        
        """
        Seebeck coefficient
        """
        # Add experimental data
        if (self.is_scatter2_there_1PB_res):
            data.extend((self.scatter_see_1PB_Srho.getData()[0], self.scatter_see_1PB_Srho.getData()[1]))
            y_legends.extend(("Temperature [K]", "Experimental Seebeck coefficient [V/K]"))
        
        # Add pre-fit curve
        data.extend((x_vals, y1_vals))
        y_legends.extend(("Temperature [K]", "Pre-fit Seebeck coefficient [V/K]"))
        
        # Add fit curve
        if (hasattr(self, 'params_1PB_res')):
            data.extend((x_vals_fit, y1_vals_fit))
            y_legends.extend(("Temperature [K]", "Fit Seebeck coefficient [V/K]"))
        
        """
        Electrical resistivity
        """
        # Add experimental data
        if (self.is_scatter2_there_1PB_res):
            data.extend((self.scatter_res_1PB_Srho.getData()[0], self.scatter_res_1PB_Srho.getData()[1]))
            y_legends.extend(("Temperature [K]", "Experimental electrical resistivity [Ohm*m]"))
        
        # Add pre-fit curve
        data.extend((x_vals, y2_vals))
        y_legends.extend(("Temperature [K]", "Pre-fit electrical resistivity [Ohm*m]"))
        
        # Add fit curve
        if (hasattr(self, 'params_1PB_res')):
            data.extend((x_vals_fit, y2_vals_fit))
            y_legends.extend(("Temperature [K]", "Fit electrical resistivity [Ohm*m]"))
        
        # Save all data
        dp.save_data(path = path, data = data, x_legend = "", y_legends = y_legends)

    def print_data_button_clicked_see_res_hall_1PB(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "all_data.txt", filter="Text file (*.txt);;All files (*.*)")
        
        # Check if saving was aborted
        if(path == ""):
            return
        data = []
        y_legends = []
        
        
        x_vals, y1_vals, y2_vals, y3_vals = TE.spb_see_res_hall_calc(self.minT, self.maxT, 50, self.mass_slider_1PB_hall.value()/100., self.fermi_slider_1PB_hall.value(), [self.para_A_slider_1PB_hall.value(), self.para_C_slider_1PB_hall.value()/100.,self.para_F_slider_1PB_hall.value(),], f"{self.scattering_type_1PB_hall.currentText()}")
        
        if (hasattr(self, 'params_1PB_hall')):
            x_vals_fit, y1_vals_fit, y2_vals_fit, y3_vals_fit = TE.spb_see_res_hall_calc(self.minT, self.maxT, 50, self.mass_slider_1PB_hall.value()/100., self.params_1PB_hall[0], self.params_1PB_hall[1:4], scatter_type = f"{self.scattering_type_1PB_hall.currentText()}")
        
        """
        Seebeck coefficient
        """
        # Add experimental data
        if (self.is_scatter2_there_1PB_hall):
            data.extend((self.scatter_see_1PB_SrhoHall.getData()[0], self.scatter_see_1PB_SrhoHall.getData()[1]))
            y_legends.extend(("Temperature [K]", "Experimental Seebeck coefficient [V/K]"))
        
        # Add pre-fit curve
        data.extend((x_vals, y1_vals))
        y_legends.extend(("Temperature [K]", "Pre-fit Seebeck coefficient [V/K]"))
        
        # Add fit curve
        if (hasattr(self, 'params_1PB_hall')):
            data.extend((x_vals_fit, y1_vals_fit))
            y_legends.extend(("Temperature [K]", "Fit Seebeck coefficient [V/K]"))
        
        """
        Electrical resistivity
        """
        # Add experimental data
        if (self.is_scatter2_there_1PB_hall):
            data.extend((self.scatter_res_1PB_SrhoHall.getData()[0], self.scatter_res_1PB_SrhoHall.getData()[1]))
            y_legends.extend(("Temperature [K]", "Experimental electrical resistivity [Ohm*m]"))
        
        # Add pre-fit curve
        data.extend((x_vals, y2_vals))
        y_legends.extend(("Temperature [K]", "Pre-fit electrical resistivity [Ohm*m]"))
        
        # Add fit curve
        if (hasattr(self, 'params_1PB_hall')):
            data.extend((x_vals_fit, y2_vals_fit))
            y_legends.extend(("Temperature [K]", "Fit electrical resistivity [Ohm*m]"))
            
        """
        Hall coefficient
        """
        # Add experimental data
        if (self.is_scatter2_there_1PB_hall):
            data.extend((self.scatter_hall_1PB_SrhoHall.getData()[0], self.scatter_hall_1PB_SrhoHall.getData()[1]))
            y_legends.extend(("Temperature [K]", "Experimental Hall coefficient [m^3/(A*s)]"))
        
        # Add pre-fit curve
        data.extend((x_vals, y3_vals))
        y_legends.extend(("Temperature [K]", "Pre-fit Hall coefficient [m^3/(A*s)]"))
        
        # Add fit curve
        if (hasattr(self, 'params_1PB_hall')):
            data.extend((x_vals_fit, y3_vals_fit))
            y_legends.extend(("Temperature [K]", "Fit Hall coefficient [m^3/(A*s)]"))
        
        # Save all data
        dp.save_data(path = path, data = data, x_legend = "", y_legends = y_legends)

    def print_data_button_clicked_see_2PB(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "all_data.txt", filter="Text file (*.txt);;All files (*.*)")
        
        # Check if saving was aborted
        if(path == ""):
            return
        data = []
        y_legends = []
        
        # Add experimental data
        if (self.is_scatter2_there):
            data.extend((self.scatter_see_2PB_S.getData()[0], self.scatter_see_2PB_S.getData()[1]))
            y_legends.extend(("Temperature [K]", "Experimental Seebeck coefficient [V/K]"))
        
        # Add pre-fit curve
        data.extend(TE.dpb_see_calc(self.minT, self.maxT, 50, [self.mass_slider.value()/100., self.bandgap_slider.value(), self.fermi_slider.value()], int(self.Nv2_value.text())/int(self.Nv1_value.text())))
        y_legends.extend(("Temperature [K]", "Pre-fit Seebeck coefficient [V/K]"))
        
        # Add fit curve
        if (hasattr(self, 'params_2PB')):
            data.extend(TE.dpb_see_calc(self.minT, self.maxT, 50, [self.params_2PB[0], self.params_2PB[1], self.params_2PB[2]], int(self.Nv2_value.text())/int(self.Nv1_value.text())))
            y_legends.extend(("Temperature [K]", "Fit Seebeck coefficient [V/K]"))
        
        # Save all data
        dp.save_data(path = path, data = data, x_legend = "", y_legends = y_legends)

    def print_data_button_clicked_see_res_2PB(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "all_data.txt", filter="Text file (*.txt);;All files (*.*)")
        
        # Check if saving was aborted
        if(path == ""):
            return
        data = []
        y_legends = []
        
        paras_see = [self.mass_slider_res.value()/100, self.bandgap_slider_res.value(), self.fermi_slider_res.value()]
        paras_res = [self.para_A_slider_res.value(), self.para_B_slider_res.value()/100, self.para_C_slider_res.value(), self.para_D_slider_res.value(), self.para_F_slider_res.value(), self.para_G_slider_res.value()/100]
        
        x_vals, y1_vals, y2_vals = TE.dpb_see_res_calc(self.minT, self.maxT, 50, paras_see, paras_res, int(self.Nv2_value_res.text())/int(self.Nv1_value_res.text()), scatter_type =f"{self.scattering_type_res.currentText()}")
        
        if (hasattr(self, 'params_2PB_res')):
            x_vals_fit, y1_vals_fit, y2_vals_fit = TE.dpb_see_res_calc(self.minT, self.maxT, 50, self.params_2PB_res[:3], self.params_2PB_res[3:], int(self.Nv2_value_res.text())/int(self.Nv1_value_res.text()), scatter_type =f"{self.scattering_type_res.currentText()}")
        
        """
        Seebeck coefficient
        """
        # Add experimental data
        if (self.is_scatter2_there_res):
            data.extend((self.scatter_see_2PB_Srho.getData()[0], self.scatter_see_2PB_Srho.getData()[1]))
            y_legends.extend(("Temperature [K]", "Experimental Seebeck coefficient [V/K]"))
        
        # Add pre-fit curve
        data.extend((x_vals, y1_vals))
        y_legends.extend(("Temperature [K]", "Pre-fit Seebeck coefficient [V/K]"))
        
        # Add fit curve
        if (hasattr(self, 'params_2PB_res')):
            data.extend((x_vals_fit, y1_vals_fit))
            y_legends.extend(("Temperature [K]", "Fit Seebeck coefficient [V/K]"))
        
        """
        Electrical resistivity
        """
        # Add experimental data
        if (self.is_scatter2_there_res):
            data.extend((self.scatter_res_2PB_Srho.getData()[0], self.scatter_res_2PB_Srho.getData()[1]))
            y_legends.extend(("Temperature [K]", "Experimental electrical resistivity [Ohm*m]"))
        
        # Add pre-fit curve
        data.extend((x_vals, y2_vals))
        y_legends.extend(("Temperature [K]", "Pre-fit electrical resistivity [Ohm*m]"))
        
        # Add fit curve
        if (hasattr(self, 'params_2PB_res')):
            data.extend((x_vals_fit, y2_vals_fit))
            y_legends.extend(("Temperature [K]", "Fit electrical resistivity [Ohm*m]"))
        
        # Save all data
        dp.save_data(path = path, data = data, x_legend = "", y_legends = y_legends)

    def print_data_button_clicked_see_res_hall_2PB(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "all_data.txt", filter="Text file (*.txt);;All files (*.*)")
        
        # Check if saving was aborted
        if(path == ""):
            return
        data = []
        y_legends = []
        
        paras_see = [self.mass_slider_hall.value()/100, self.bandgap_slider_hall.value(), self.fermi_slider_hall.value()]
        paras_res = [self.para_A_slider_hall.value(), self.para_B_slider_hall.value()/100, self.para_C_slider_hall.value(), self.para_D_slider_hall.value(), self.para_F_slider_hall.value(), self.para_G_slider_hall.value()/100]
        
        x_vals, y1_vals, y2_vals, y3_vals = TE.dpb_see_res_hall_calc(self.minT, self.maxT, 50, paras_see, paras_res, self.para_E_slider_hall.value()/100, int(self.Nv2_value_res.text())/int(self.Nv1_value_res.text()), scatter_type =f"{self.scattering_type_res.currentText()}")
        
        if (hasattr(self, 'params_2PB_hall')):
            x_vals_fit, y1_vals_fit, y2_vals_fit, y3_vals_fit = TE.dpb_see_res_hall_calc(self.minT, self.maxT, 50, self.params_2PB_hall[:3], self.params_2PB_hall[3:9], self.params_2PB_hall[-1], int(self.Nv2_value_res.text())/int(self.Nv1_value_res.text()), scatter_type =f"{self.scattering_type_res.currentText()}")
        
        """
        Seebeck coefficient
        """
        # Add experimental data
        if (self.is_scatter2_there_hall):
            data.extend((self.scatter_see_2PB_SrhoHall.getData()[0], self.scatter_see_2PB_SrhoHall.getData()[1]))
            y_legends.extend(("Temperature [K]", "Experimental Seebeck coefficient [V/K]"))
        
        # Add pre-fit curve
        data.extend((x_vals, y1_vals))
        y_legends.extend(("Temperature [K]", "Pre-fit Seebeck coefficient [V/K]"))
        
        # Add fit curve
        if (hasattr(self, 'params_2PB_hall')):
            data.extend((x_vals_fit, y1_vals_fit))
            y_legends.extend(("Temperature [K]", "Fit Seebeck coefficient [V/K]"))
        
        """
        Electrical resistivity
        """
        # Add experimental data
        if (self.is_scatter2_there_hall):
            data.extend((self.scatter_res_2PB_SrhoHall.getData()[0], self.scatter_res_2PB_SrhoHall.getData()[1]))
            y_legends.extend(("Temperature [K]", "Experimental electrical resistivity [Ohm*m]"))
        
        # Add pre-fit curve
        data.extend((x_vals, y2_vals))
        y_legends.extend(("Temperature [K]", "Pre-fit electrical resistivity [Ohm*m]"))
        
        # Add fit curve
        if (hasattr(self, 'params_2PB_hall')):
            data.extend((x_vals_fit, y2_vals_fit))
            y_legends.extend(("Temperature [K]", "Fit electrical resistivity [Ohm*m]"))
            
        """
        Hall coefficient
        """
        # Add experimental data
        if (self.is_scatter2_there_hall):
            data.extend((self.scatter_hall_2PB_SrhoHall.getData()[0], self.scatter_hall_2PB_SrhoHall.getData()[1]))
            y_legends.extend(("Temperature [K]", "Experimental Hall coefficient [m^3/(A*s)]"))
        
        # Add pre-fit curve
        data.extend((x_vals, y3_vals))
        y_legends.extend(("Temperature [K]", "Pre-fit Hall coefficient [m^3/(A*s)]"))
        
        # Add fit curve
        if (hasattr(self, 'params_2PB_hall')):
            data.extend((x_vals_fit, y3_vals_fit))
            y_legends.extend(("Temperature [K]", "Fit Hall coefficient [m^3/(A*s)]"))
        
        # Save all data
        dp.save_data(path = path, data = data, x_legend = "", y_legends = y_legends)


    #print parameter functions
    def print_params_button_clicked_see_1PB(self):
        if (hasattr(self, 'params_1PB')):
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "all_parameters.txt", filter="Text file (*.txt);;All files (*.*)")
            
            # Check if saving was aborted
            if(path == ""):
                return
        
            para_dict = {'Fermi energy [K]': self.params_1PB[0]}
            dp.save_params(path, para_dict)
            
    def print_params_button_clicked_see_res_1PB(self):
        if (hasattr(self, 'params_1PB_res')):
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "all_parameters.txt", filter="Text file (*.txt);;All files (*.*)")
            
            # Check if saving was aborted
            if(path == ""):
                return
        
            para_dict = {}
            para_dict['Fermi energy [K]'] = self.params_1PB_res[0]
            if (self.scattering_type_1PB_res.currentText() == "acPh"):
                para_dict['Acoustic phonon scattering time'] = self.params_1PB_res[1]
            elif (self.scattering_type_1PB_res.currentText() == "dis"):
                para_dict['Alloy disorder scattering time'] = self.params_1PB_res[3]
            else:
                para_dict['Acoustic phonon scattering time'] = self.params_1PB_res[1]
                para_dict['Scattering time ratio'] = self.params_1PB_res[2]
            dp.save_params(path, para_dict)

    def print_params_button_clicked_see_res_hall_1PB(self):
        if (hasattr(self, 'params_1PB_hall')):
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "all_parameters.txt", filter="Text file (*.txt);;All files (*.*)")
            
            # Check if saving was aborted
            if(path == ""):
                return
        
            para_dict = {}
            para_dict['Fermi energy [K]'] = self.params_1PB_hall[0]
            if (self.scattering_type_1PB_hall.currentText() == "acPh"):
                para_dict['Acoustic phonon scattering time'] = self.params_1PB_hall[1]
            elif (self.scattering_type_1PB_hall.currentText() == "dis"):
                para_dict['Alloy disorder scattering time'] = self.params_1PB_hall[3]
            else:
                para_dict['Acoustic phonon scattering time'] = self.params_1PB_hall[1]
                para_dict['Scattering time ratio'] = self.params_1PB_hall[2]
            dp.save_params(path, para_dict)
            
    def print_params_button_clicked_see_2PB(self):
        if (hasattr(self, 'params_2PB')):
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "all_parameters.txt", filter="Text file (*.txt);;All files (*.*)")
            
            # Check if saving was aborted
            if(path == ""):
                return
        
            para_dict = {}
            para_dict['Band-mass ratio'] = self.params_2PB[0]
            para_dict['Band gap [K]'] = self.params_2PB[1]
            para_dict['Fermi energy [K]'] = self.params_2PB[2]

            dp.save_params(path, para_dict)

    def print_params_button_clicked_see_res_2PB(self):
        if (hasattr(self, 'params_2PB_res')):
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "all_parameters.txt", filter="Text file (*.txt);;All files (*.*)")
            
            # Check if saving was aborted
            if(path == ""):
                return
        
            para_dict = {}
            para_dict['Band-mass ratio'] = self.params_2PB_res[0]
            para_dict['Band gap [K]'] = self.params_2PB_res[1]
            para_dict['Fermi energy [K]'] = self.params_2PB_res[2]
            
            if (self.scattering_type_res.currentText() == "acPh"):
                para_dict['Acoustic phonon scattering time band 1'] = self.params_2PB_res[3]
                para_dict['Acoustic phonon scattering time ratio'] = self.params_2PB_res[4]
            elif (self.scattering_type_res.currentText() == "dis"):
                para_dict['Alloy disorder scattering time band 1'] = self.params_2PB_res[7]
                para_dict['Alloy disorder scattering time ratio'] = self.params_2PB_res[8]
            else:
                para_dict['Acoustic phonon scattering time band 1'] = self.params_2PB_res[3]
                para_dict['Acoustic phonon scattering time ratio'] = self.params_2PB_res[4]
                para_dict['Scattering time ratio band 1'] = self.params_2PB_res[5]
                para_dict['Scattering time ratio band 2'] = self.params_2PB_res[6]
            dp.save_params(path, para_dict)

    def print_params_button_clicked_see_res_hall_2PB(self):
        if (hasattr(self, 'params_2PB_hall')):
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "all_parameters.txt", filter="Text file (*.txt);;All files (*.*)")
            
            # Check if saving was aborted
            if(path == ""):
                return
        
            para_dict = {}
            para_dict['Band-mass ratio'] = self.params_2PB_hall[0]
            para_dict['Band gap [K]'] = self.params_2PB_hall[1]
            para_dict['Fermi energy [K]'] = self.params_2PB_hall[2]
            
            if (self.scattering_type_hall.currentText() == "acPh"):
                para_dict['Acoustic phonon scattering time band 1'] = self.params_2PB_hall[3]
                para_dict['Acoustic phonon scattering time ratio'] = self.params_2PB_hall[4]
            elif (self.scattering_type_hall.currentText() == "dis"):
                para_dict['Alloy disorder scattering time band 1'] = self.params_2PB_hall[7]
                para_dict['Alloy disorder scattering time ratio'] = self.params_2PB_hall[8]
            else:
                para_dict['Acoustic phonon scattering time band 1'] = self.params_2PB_hall[3]
                para_dict['Acoustic phonon scattering time ratio'] = self.params_2PB_hall[4]
                para_dict['Scattering time ratio band 1'] = self.params_2PB_hall[5]
                para_dict['Scattering time ratio band 2'] = self.params_2PB_hall[6]
            
            para_dict['Mass band 1'] = self.params_2PB_hall[9] 
            
            dp.save_params(path, para_dict)


    """
    Functions for the advanced options window
    """
    #set the new boundaries from the advanced options window (6 x Second_windows)
    def fitrange_button_clicked_1PB(self, minimum, maximum, mass_min, mass_max, fermi_min, fermi_max, T_changed):
        self.minT = minimum
        self.maxT = maximum
        self.mass_slider_1PB.setMinimum(mass_min)
        self.mass_slider_1PB.setMaximum(mass_max)
        self.fermi_slider_1PB.setMinimum(fermi_min)
        self.fermi_slider_1PB.setMaximum(fermi_max)
        if T_changed == True:    
            self.x_mask = (self.interpolated_data[:, 0] >= minimum) & (
                self.interpolated_data[:, 0] <= maximum)
               
    def fitrange_button_clicked_1PB_res(self, minimum, maximum, mass_min, mass_max, fermi_min, fermi_max, parA_min, parA_max, T_changed):
        self.minT = minimum
        self.maxT = maximum
        self.mass_slider_1PB_res.setMinimum(mass_min)
        self.mass_slider_1PB_res.setMaximum(mass_max)
        self.fermi_slider_1PB_res.setMinimum(fermi_min)
        self.fermi_slider_1PB_res.setMaximum(fermi_max)
        self.para_A_lower_limit_res = parA_min
        self.para_A_upper_limit_res = parA_max
        self.fermi_slider_1PB_res.setMaximum(fermi_max)
        if T_changed == True:    
            self.x_mask = (self.interpolated_data[:, 0] >= minimum) & (
                self.interpolated_data[:, 0] <= maximum)
            
    def fitrange_button_clicked_1PB_hall(self, minimum, maximum, mass_min, mass_max, fermi_min, fermi_max, parA_min, parA_max, T_changed):
        self.minT = minimum
        self.maxT = maximum
        self.mass_slider_1PB_hall.setMinimum(mass_min)
        self.mass_slider_1PB_hall.setMaximum(mass_max)
        self.fermi_slider_1PB_hall.setMinimum(fermi_min)
        self.fermi_slider_1PB_hall.setMaximum(fermi_max)
        self.para_A_slider_hall.setMinimum(parA_min)
        self.para_A_upper_limit_hall = parA_max
        self.fermi_slider_1PB_hall.setMaximum(fermi_max)
        if T_changed == True:    
            self.x_mask = (self.interpolated_data[:, 0] >= minimum) & (
                self.interpolated_data[:, 0] <= maximum)
        
    def fitrange_button_clicked(self, minimum, maximum, mass_min, mass_max, bandgap_min, bandgap_max, fermi_min, fermi_max, T_changed):
        self.minT = minimum
        self.maxT = maximum
        self.mass_slider.setMinimum(mass_min)
        self.mass_slider.setMaximum(mass_max)
        self.bandgap_slider.setMinimum(bandgap_min)
        self.bandgap_slider.setMaximum(bandgap_max)
        self.fermi_slider.setMinimum(fermi_min)
        self.fermi_slider.setMaximum(fermi_max)
        if T_changed == True:    
            self.x_mask = (self.interpolated_data[:, 0] >= minimum) & (
                self.interpolated_data[:, 0] <= maximum)
    
    def fitrange_button_clicked_res(self, minimum, maximum, mass_min, mass_max, bandgap_min, bandgap_max, fermi_min, fermi_max, parA_min, parB_min, parC_min, parD_min, parF_min, parG_min, parA_max, parB_max, parC_max, parD_max, parF_max, parG_max, T_changed):
        self.minT = minimum
        self.maxT = maximum
        if T_changed == True:    
            self.x_mask = (self.interpolated_data[:, 0] >= minimum) & (
                self.interpolated_data[:, 0] <= maximum)
            self.x_mask_res = (self.interpolated_res_data[:, 0] >= minimum) & (
                self.interpolated_res_data[:, 0] <= maximum)
            
        self.mass_slider_res.setMinimum(mass_min)
        self.mass_slider_res.setMaximum(mass_max)
        self.bandgap_slider_res.setMinimum(bandgap_min)
        
        self.bandgap_slider_res.setMaximum(bandgap_max)
        self.fermi_slider_res.setMinimum(fermi_min)
        self.fermi_slider_res.setMaximum(fermi_max)
        
        self.para_A_slider_res.setMinimum(int(float(parA_min)))
        self.para_B_slider_res.setMinimum(int(float(parB_min)))
        self.para_C_slider_res.setMinimum(int(float(parC_min)))
        self.para_D_slider_res.setMinimum(int(float(parD_min)))
        self.para_F_slider_res.setMinimum(int(float(parF_min)))
        self.para_G_slider_res.setMinimum(int(float(parG_min)))
        
        self.para_A_slider_res.setMaximum(int(float(parA_max)))
        self.para_B_slider_res.setMaximum(int(float(parB_max)))
        self.para_C_slider_res.setMaximum(int(float(parC_max)))
        self.para_D_slider_res.setMaximum(int(float(parD_max)))
        self.para_F_slider_res.setMaximum(int(float(parF_max)))
        self.para_G_slider_res.setMaximum(int(float(parG_max)))
        
    def fitrange_button_clicked_hall(self, minimum, maximum, mass_min, mass_max, bandgap_min, bandgap_max, fermi_min, fermi_max, parA_min, parB_min, parC_min, parD_min, parE_min, parF_min, parG_min, parA_max, parB_max, parC_max, parD_max, parE_max, parF_max, parG_max, T_changed):
        self.minT = minimum
        self.maxT = maximum
        if T_changed == True:     
            self.x_mask = (self.interpolated_data[:, 0] >= minimum) & (
                self.interpolated_data[:, 0] <= maximum)
            self.x_mask_res = (self.interpolated_res_data[:, 0] >= minimum) & (
                self.interpolated_res_data[:, 0] <= maximum)
            self.x_mask_hall = (self.interpolated_hall_data[:, 0] >= minimum) & (
                self.interpolated_hall_data[:, 0] <= maximum)
            
        self.mass_slider_hall.setMinimum(mass_min)
        self.mass_slider_hall.setMaximum(mass_max)
        self.bandgap_slider_hall.setMinimum(bandgap_min)
        
        self.bandgap_slider_hall.setMaximum(bandgap_max)
        self.fermi_slider_hall.setMinimum(fermi_min)
        self.fermi_slider_hall.setMaximum(fermi_max)    
        
        self.para_A_slider_hall.setMinimum(parA_min)
        self.para_B_slider_hall.setMinimum(parB_min)
        self.para_C_slider_hall.setMinimum(parC_min)
        self.para_D_slider_hall.setMinimum(parD_min)
        self.para_E_slider_hall.setMinimum(parE_min)
        self.para_F_slider_hall.setMinimum(parF_min)
        self.para_G_slider_hall.setMinimum(parG_min)
        
        self.para_A_slider_hall.setMaximum(parA_max)
        self.para_B_slider_hall.setMaximum(parB_max)
        self.para_C_slider_hall.setMaximum(parC_max)
        self.para_D_slider_hall.setMaximum(parD_max)
        self.para_E_slider_hall.setMaximum(parE_max)
        self.para_F_slider_hall.setMaximum(parF_max)
        self.para_G_slider_hall.setMaximum(parG_max)
           
    #  Functions for the combo-box determining which type of scattering is used for the calculations
    def type_of_fitting_changed_res(self):
          if self.scattering_type_res.currentText() == "acPh":
        
              self.para_F_label_res.setVisible(False)
              self.para_F_slider_res.setVisible(False)
              self.para_F_lineedit_res.setVisible(False)
              self.para_G_label_res.setVisible(False)
              self.para_G_slider_res.setVisible(False)
              self.para_G_lineedit_res.setVisible(False)
              self.para_C_label_res.setVisible(False)
              self.para_C_slider_res.setVisible(False)
              self.para_C_lineedit_res.setVisible(False)
              self.para_D_label_res.setVisible(False)
              self.para_D_slider_res.setVisible(False)
              self.para_D_lineedit_res.setVisible(False)
              self.para_F_fit_label_res.setVisible(False)
              self.para_G_fit_label_res.setVisible(False)
              self.para_C_fit_label_res.setVisible(False)
              self.para_D_fit_label_res.setVisible(False)
              
              self.para_A_label_res.setVisible(True)
              self.para_A_slider_res.setVisible(True)
              self.para_A_lineedit_res.setVisible(True)
              self.para_B_label_res.setVisible(True)
              self.para_B_slider_res.setVisible(True)
              self.para_B_lineedit_res.setVisible(True)
              
              self.manipulate_value_changed_res()
        
          if self.scattering_type_res.currentText() == "dis":
              self.para_A_label_res.setVisible(False)
              self.para_A_slider_res.setVisible(False)
              self.para_A_lineedit_res.setVisible(False)
              self.para_B_label_res.setVisible(False)
              self.para_B_slider_res.setVisible(False)
              self.para_B_lineedit_res.setVisible(False)
              self.para_C_label_res.setVisible(False)
              self.para_C_slider_res.setVisible(False)
              self.para_C_lineedit_res.setVisible(False)
              self.para_D_label_res.setVisible(False)
              self.para_D_slider_res.setVisible(False)
              self.para_D_lineedit_res.setVisible(False)
              self.para_A_fit_label_res.setVisible(False)
              self.para_B_fit_label_res.setVisible(False)
              self.para_C_fit_label_res.setVisible(False)
              self.para_D_fit_label_res.setVisible(False)
              
              self.para_F_label_res.setVisible(True)
              self.para_F_slider_res.setVisible(True)
              self.para_F_lineedit_res.setVisible(True)
              self.para_G_label_res.setVisible(True)
              self.para_G_slider_res.setVisible(True)
              self.para_G_lineedit_res.setVisible(True)
              
              self.manipulate_value_changed_res()
        
          if self.scattering_type_res.currentText() == "acPhDis":
              self.para_F_label_res.setVisible(False)
              self.para_F_slider_res.setVisible(False)
              self.para_F_lineedit_res.setVisible(False)
              self.para_G_label_res.setVisible(False)
              self.para_G_slider_res.setVisible(False)
              self.para_G_lineedit_res.setVisible(False)
              self.para_F_fit_label_res.setVisible(False)
              self.para_G_fit_label_res.setVisible(False)

              self.para_A_label_res.setVisible(True)
              self.para_A_slider_res.setVisible(True)
              self.para_A_lineedit_res.setVisible(True)
              self.para_B_label_res.setVisible(True)
              self.para_B_slider_res.setVisible(True)
              self.para_B_lineedit_res.setVisible(True)
              self.para_C_label_res.setVisible(True)
              self.para_C_slider_res.setVisible(True)
              self.para_C_lineedit_res.setVisible(True)
              self.para_D_label_res.setVisible(True)
              self.para_D_slider_res.setVisible(True)
              self.para_D_lineedit_res.setVisible(True)
              
              self.manipulate_value_changed_res()
    
    def type_of_fitting_changed_1PB_res(self):
          if self.scattering_type_res.currentText() == "acPh":
        
              self.para_F_label_1PB_res.setVisible(False)
              self.para_F_slider_1PB_res.setVisible(False)
              self.para_F_lineedit_1PB_res.setVisible(False)

              self.para_C_label_1PB_res.setVisible(False)
              self.para_C_slider_1PB_res.setVisible(False)
              self.para_C_lineedit_1PB_res.setVisible(False)
              self.para_F_fit_label_1PB_res.setVisible(False)
              self.para_C_fit_label_1PB_res.setVisible(False)
              
              self.para_A_label_1PB_res.setVisible(True)
              self.para_A_slider_1PB_res.setVisible(True)
              self.para_A_lineedit_1PB_res.setVisible(True)
              
              self.manipulate_value_changed_1PB_res()
        
          if self.scattering_type_1PB_res.currentText() == "dis":
              self.para_A_label_1PB_res.setVisible(False)
              self.para_A_slider_1PB_res.setVisible(False)
              self.para_A_lineedit_1PB_res.setVisible(False)
              self.para_C_label_1PB_res.setVisible(False)
              self.para_C_slider_1PB_res.setVisible(False)
              self.para_C_lineedit_1PB_res.setVisible(False)

              self.para_A_fit_label_1PB_res.setVisible(False)
              self.para_C_fit_label_1PB_res.setVisible(False)
              self.para_F_label_1PB_res.setVisible(True)
              self.para_F_slider_1PB_res.setVisible(True)
              self.para_F_lineedit_1PB_res.setVisible(True)
              
              self.manipulate_value_changed_1PB_res()
        
          if self.scattering_type_1PB_res.currentText() == "acPhDis":
              self.para_F_label_1PB_res.setVisible(False)
              self.para_F_slider_1PB_res.setVisible(False)
              self.para_F_lineedit_1PB_res.setVisible(False)
              self.para_F_fit_label_1PB_res.setVisible(False)

              self.para_A_label_1PB_res.setVisible(True)
              self.para_A_slider_1PB_res.setVisible(True)
              self.para_A_lineedit_1PB_res.setVisible(True)
              self.para_C_label_1PB_res.setVisible(True)
              self.para_C_slider_1PB_res.setVisible(True)
              self.para_C_lineedit_1PB_res.setVisible(True)
              
              self.manipulate_value_changed_1PB_res()
    
    def type_of_fitting_changed_1PB_hall(self):
        pass
    
    def type_of_fitting_changed_hall(self):
        if self.scattering_type_hall.currentText() == "acPh":

            self.para_F_label_hall.setVisible(False)
            self.para_F_slider_hall.setVisible(False)
            self.para_F_lineedit_hall.setVisible(False)
            self.para_G_label_hall.setVisible(False)
            self.para_G_slider_hall.setVisible(False)
            self.para_G_lineedit_hall.setVisible(False)
            self.para_C_label_hall.setVisible(False)
            self.para_C_slider_hall.setVisible(False)
            self.para_C_lineedit_hall.setVisible(False)
            self.para_D_label_hall.setVisible(False)
            self.para_D_slider_hall.setVisible(False)
            self.para_D_lineedit_hall.setVisible(False)
            self.para_F_fit_label_hall.setVisible(False)
            self.para_G_fit_label_hall.setVisible(False)
            self.para_C_fit_label_hall.setVisible(False)
            self.para_D_fit_label_hall.setVisible(False)
            
            self.para_A_label_hall.setVisible(True)
            self.para_A_slider_hall.setVisible(True)
            self.para_A_lineedit_hall.setVisible(True)
            self.para_B_label_hall.setVisible(True)
            self.para_B_slider_hall.setVisible(True)
            self.para_B_lineedit_hall.setVisible(True)
            
            self.manipulate_value_changed_hall()

        if self.scattering_type_hall.currentText() == "dis":
            self.para_A_label_hall.setVisible(False)
            self.para_A_slider_hall.setVisible(False)
            self.para_A_lineedit_hall.setVisible(False)
            self.para_B_label_hall.setVisible(False)
            self.para_B_slider_hall.setVisible(False)
            self.para_B_lineedit_hall.setVisible(False)
            self.para_C_label_hall.setVisible(False)
            self.para_C_slider_hall.setVisible(False)
            self.para_C_lineedit_hall.setVisible(False)
            self.para_D_label_hall.setVisible(False)
            self.para_D_slider_hall.setVisible(False)
            self.para_D_lineedit_hall.setVisible(False)
            self.para_A_fit_label_hall.setVisible(False)
            self.para_B_fit_label_hall.setVisible(False)
            self.para_C_fit_label_hall.setVisible(False)
            self.para_D_fit_label_hall.setVisible(False)
            
            self.para_F_label_hall.setVisible(True)
            self.para_F_slider_hall.setVisible(True)
            self.para_F_lineedit_hall.setVisible(True)
            self.para_G_label_hall.setVisible(True)
            self.para_G_slider_hall.setVisible(True)
            self.para_G_lineedit_hall.setVisible(True)
            
            self.manipulate_value_changed_hall()

        if self.scattering_type_hall.currentText() == "acPhDis":
            self.para_F_label_hall.setVisible(False)
            self.para_F_slider_hall.setVisible(False)
            self.para_F_lineedit_hall.setVisible(False)
            self.para_G_label_hall.setVisible(False)
            self.para_G_slider_hall.setVisible(False)
            self.para_G_lineedit_hall.setVisible(False)
            self.para_F_fit_label_hall.setVisible(False)
            self.para_G_fit_label_hall.setVisible(False)

            self.para_A_label_hall.setVisible(True)
            self.para_A_slider_hall.setVisible(True)
            self.para_A_lineedit_hall.setVisible(True)
            self.para_B_label_hall.setVisible(True)
            self.para_B_slider_hall.setVisible(True)
            self.para_B_lineedit_hall.setVisible(True)
            self.para_C_label_hall.setVisible(True)
            self.para_C_slider_hall.setVisible(True)
            self.para_C_lineedit_hall.setVisible(True)
            self.para_D_label_hall.setVisible(True)
            self.para_D_slider_hall.setVisible(True)
            self.para_D_lineedit_hall.setVisible(True)
            
            self.manipulate_value_changed_hall()

"""
Classes for computation Threads
"""
class FittingThread_1PB_See(QThread):
    
    updateGUI = pyqtSignal(np.ndarray, np.ndarray, list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, MainWindow):
        super().__init__()
        self.main = MainWindow
        pass
    
    def run(self):
        shape = (50,)
        fit_x_val, fit_y_val_see = np.zeros((2,) + shape)
        params = []
        
        try:
            initial = [self.main.fermi_slider_1PB.value()]
            limits = [self.main.fermi_slider_1PB.minimum()
                    ], [self.main.fermi_slider_1PB.maximum()]
    #        params = TE.start_fit(self.interpolated_data[self.x_mask,0], self.interpolated_data[self.x_mask,1], initial, limits )
            params = TE.spb_see_fit(self.main.interpolated_data[self.main.x_mask, 0], self.main.interpolated_data[self.main.x_mask, 1], self.main.mass_slider_1PB.value()/100., initial ,limits, print_info=True)  # last argument is Nv2/Nv1
            self.main.params_1PB = params
            print(params[0])
            
            fit_x_val, fit_y_val_see = TE.spb_see_calc(self.main.minT, self.main.maxT, 50, self.main.mass_slider_1PB.value()/100., params[0])
            
        except Exception as e:
            print(f"Exception: {e}")
            self.error_occurred.emit("spb_see")
            return
            
        self.updateGUI.emit(fit_x_val, fit_y_val_see, params)
    
class FittingThread_1PB_Res(QThread):
    
    updateGUI = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, MainWindow):
        super().__init__()
        self.main = MainWindow
    
    def run(self):
        shape = (50,)
        fit_x_val, fit_y_val_see, fit_y_val_res = np.zeros((3,) + shape)
        params = []
        
        try:
            initial_see = [self.main.fermi_slider_1PB_res.value()]
            limits_see = [self.main.fermi_slider_1PB_res.minimum()], [self.main.fermi_slider_1PB.maximum()]
            initial_res = [self.main.para_A_slider_1PB_res.value(), self.main.para_C_slider_1PB_res.value(), self.main.para_F_slider_1PB_res.value()]
            limits_res = [self.main.para_A_lower_limit_1PB_res, self.main.para_C_lower_limit_1PB_res, self.main.para_F_lower_limit_1PB_res],[ 
            self.main.para_A_upper_limit_1PB_res,self.main.para_C_upper_limit_1PB_res, self.main.para_F_upper_limit_1PB_res]

            params = TE.spb_see_res_fit(self.main.interpolated_data[self.main.x_mask, 0], self.main.interpolated_data[self.main.x_mask, 1], self.main.interpolated_res_data[self.main.x_mask_res, 0], self.main.interpolated_res_data[self.main.x_mask_res, 1], 
                                        self.main.mass_slider_1PB_res.value()/100., initial_see , limits_see, initial_res, limits_res, scatter_type = f"{self.main.scattering_type_1PB_res.currentText()}", print_info=True)
            self.main.params_1PB_res = params
            print(params[0:3])
            
            fit_x_val, fit_y_val_see, fit_y_val_res = TE.spb_see_res_calc(self.main.minT, self.main.maxT, 50, self.main.mass_slider_1PB_res.value()/100., params[0], params[1:4], scatter_type = f"{self.main.scattering_type_1PB_res.currentText()}")
        
        except Exception as e:
            print(f"Exception: {e}")
            self.error_occurred.emit("spb_res")
            return
              
        self.updateGUI.emit(fit_x_val, fit_y_val_see, fit_y_val_res, params)
    
class FittingThread_1PB_Hall(QThread):
    
    updateGUI = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, MainWindow):
        super().__init__()
        self.main = MainWindow
    
    def run(self):
        shape = (50,)
        fit_x_val, fit_y_val_see, fit_y_val_res, fit_y_val_hall = np.zeros((4,) + shape)
        params = []
        
        try:
            initial_see = [self.main.fermi_slider_1PB_hall.value()]
            limits_see = [self.main.fermi_slider_1PB_hall.minimum()], [self.main.fermi_slider_1PB.maximum()]
            initial_res = [self.main.para_A_slider_1PB_hall.value(), self.main.para_C_slider_1PB_hall.value(), self.main.para_F_slider_1PB_hall.value()]
            limits_res = [self.main.para_A_lower_limit_1PB_hall, self.main.para_C_lower_limit_1PB_hall, self.main.para_F_lower_limit_1PB_hall],[ 
            self.main.para_A_upper_limit_1PB_hall,self.main.para_C_upper_limit_1PB_hall, self.main.para_F_upper_limit_1PB_hall]
            initial_mass = [self.main.mass_slider_1PB_hall.value()]
            limits_mass = [self.main.mass_slider_1PB_hall.minimum()], [self.main.mass_slider_1PB_hall.maximum()]

            params = TE.spb_see_res_hall_fit(self.main.interpolated_data[self.main.x_mask, 0], self.main.interpolated_data[self.main.x_mask, 1], self.main.interpolated_res_data[self.main.x_mask_res, 0], self.main.interpolated_res_data[self.main.x_mask_res, 1], self.main.interpolated_hall_data[self.main.x_mask_hall, 0], self.main.interpolated_hall_data[self.main.x_mask_hall, 1],
                                        initial_see , limits_see, initial_res, limits_res, initial_mass, limits_mass, scatter_type = f"{self.main.scattering_type_1PB_hall.currentText()}", print_info=True)
            self.main.params_1PB_hall = params
            print(params[0:5])

            fit_x_val, fit_y_val_see, fit_y_val_res, fit_y_val_hall = TE.spb_see_res_hall_calc(self.main.minT, self.main.maxT, 50,  -params[4], params[0], params[1:3], scatter_type = f"{self.main.scattering_type_1PB_hall.currentText()}")
        
        except Exception as e:
            print(f"Exception: {e}")
            self.error_occurred.emit("spb_hall")
            return

        self.updateGUI.emit(fit_x_val, fit_y_val_see, fit_y_val_res, fit_y_val_hall, params)
    
class FittingThread_2PB_See(QThread):
    
    updateGUI = pyqtSignal(np.ndarray, np.ndarray, list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, MainWindow):
        super().__init__()
        self.main = MainWindow
        pass
    
    def run(self):
        shape = (50,)
        fit_x_val, fit_y_val_see = np.zeros((2,) + shape)
        params = []
        
        try:
            initial = [self.main.mass_slider.value()/100.,
                    self.main.bandgap_slider.value(), self.main.fermi_slider.value()]
            limits = [self.main.mass_slider.minimum()/100., self.main.bandgap_slider.minimum(), self.main.fermi_slider.minimum()
                    ], [self.main.mass_slider.maximum()/100., self.main.bandgap_slider.maximum(), self.main.fermi_slider.maximum()]
            params = TE.dpb_see_fit(self.main.interpolated_data[self.main.x_mask, 0], self.main.interpolated_data[self.main.x_mask, 1], initial, limits, int(
                self.main.Nv2_value.text())/int(self.main.Nv1_value.text()), print_info=True)  # last argument is Nv2/Nv1
        
            self.main.params_2PB = params
            print(params[0], params[1], params[2])

            fit_x_val, fit_y_val_see = TE.dpb_see_calc(self.main.minT, self.main.maxT, 50, [params[0], params[1], params[2]], int(
                self.main.Nv2_value.text())/int(self.main.Nv1_value.text()))
        
        except Exception as e:
            print(f"Exception: {e}")
            self.error_occurred.emit("dpb_see")
            return
        
        self.updateGUI.emit(fit_x_val, fit_y_val_see, params)
        
class FittingThread_2PB_Res(QThread):
    
    updateGUI = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, MainWindow):
        super().__init__()
        self.main = MainWindow
        pass
    
    def run(self):
        shape = (50,)
        fit_x_val, fit_y_val_see, fit_y_val_res = np.zeros((3,) + shape)
        params = []
        
        try:
            initial_see = [self.main.mass_slider_res.value()/100.,
                    self.main.bandgap_slider_res.value(), self.main.fermi_slider_res.value()]
            limits_see = [self.main.mass_slider_res.minimum()/100., self.main.bandgap_slider_res.minimum(), self.main.fermi_slider_res.minimum()
                    ], [self.main.mass_slider_res.maximum()/100., self.main.bandgap_slider_res.maximum(), self.main.fermi_slider_res.maximum()]
            initial_res = [self.main.para_A_slider_res.value(), self.main.para_B_slider_res.value()/100., self.main.para_C_slider_res.value(), self.main.para_D_slider_res.value(), self.main.para_F_slider_res.value(), self.main.para_G_slider_res.value()/100.]


            limits_res = [self.main.para_A_lower_limit_res, self.main.para_B_lower_limit_res, self.main.para_C_lower_limit_res, self.main.para_D_lower_limit_res, self.main.para_F_lower_limit_res, self.main.para_G_lower_limit_res],[ 
            self.main.para_A_upper_limit_res, self.main.para_B_upper_limit_res, self.main.para_C_upper_limit_res, self.main.para_D_upper_limit_res, self.main.para_F_upper_limit_res, self.main.para_G_upper_limit_res]
            

            params = TE.dpb_see_res_fit(self.main.interpolated_data[self.main.x_mask, 0], self.main.interpolated_data[self.main.x_mask, 1], 
                                            self.main.interpolated_res_data[self.main.x_mask_res, 0], self.main.interpolated_res_data[self.main.x_mask_res, 1],
                                            initial_see, limits_see, initial_res, limits_res, int(self.main.Nv2_value_res.text())/int(self.main.Nv1_value_res.text()),
                                            f"{self.main.scattering_type_res.currentText()}", 0.1, 1000, True)  # last argument is Nv2/Nv1
            self.main.params_2PB_res = params
            print(f"params[0] = {params[0]}, params[1] = {params[1]}, params[2] = {params[2]}, params[3] = {params[3]}, params[4] = {params[4]}, params[5] = {params[5]}, params[6] = {params[6]}, params[7] = {params[7]}, params[8] = {params[8]}")
        
            fit_x_val, fit_y_val_see, fit_y_val_res = TE.dpb_see_res_calc(self.main.minT, self.main.maxT, 50, params[0:3],params[3:], int(self.main.Nv2_value_res.text())/int(self.main.Nv1_value_res.text()), 
                                                                   scatter_type =f"{self.main.scattering_type_res.currentText()}")
        
        except Exception as e:
            print(f"Exception: {e}")
            self.error_occurred.emit("dpb_res")
            return
        
        self.updateGUI.emit(fit_x_val, fit_y_val_see, fit_y_val_res, params)
        
class FittingThread_2PB_Hall(QThread):
    
    updateGUI = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray, list)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, MainWindow):
        super().__init__()
        self.main = MainWindow
        pass
    
    def run(self):
        shape = (50,)
        fit_x_val, fit_y_val_see, fit_y_val_res, fit_y_val_hall = np.zeros((4,) + shape)
        params = []
        
        try:
            initial_see = [self.main.mass_slider_hall.value()/100.,
                    self.main.bandgap_slider_hall.value(), self.main.fermi_slider_hall.value()]
            limits_see = [self.main.mass_slider_hall.minimum()/100., self.main.bandgap_slider_hall.minimum(), self.main.fermi_slider_hall.minimum()
                    ], [self.main.mass_slider_hall.maximum()/100., self.main.bandgap_slider_hall.maximum(), self.main.fermi_slider_hall.maximum()]
            initial_res = [self.main.para_A_slider_hall.value(), self.main.para_B_slider_hall.value()/100., self.main.para_C_slider_hall.value(), self.main.para_D_slider_hall.value(), self.main.para_F_slider_hall.value(), self.main.para_G_slider_hall.value()/100.]
            limits_res = [self.main.para_A_slider_hall.minimum(), self.main.para_B_slider_hall.minimum()/100., self.main.para_C_slider_hall.minimum(), self.main.para_D_slider_hall.minimum(), self.main.para_F_slider_hall.minimum(), self.main.para_G_slider_hall.minimum()/100.], [self.main.para_A_slider_hall.maximum(), self.main.para_B_slider_hall.maximum()/100.,self.main.para_C_slider_hall.maximum(), self.main.para_D_slider_hall.maximum(),self.main.para_F_slider_hall.maximum(), self.main.para_G_slider_hall.maximum()/100.]     
            initial_hall = self.main.para_E_slider_hall.value()/100.
            limits_hall = [[self.main.para_E_slider_hall.minimum()], [self.main.para_E_slider_hall.maximum()]]

            params = TE.dpb_see_res_hall_fit(self.main.interpolated_data[self.main.x_mask, 0], self.main.interpolated_data[self.main.x_mask, 1], self.main.interpolated_res_data[self.main.x_mask_res, 0], 
                                            self.main.interpolated_res_data[self.main.x_mask_res, 1], self.main.interpolated_hall_data[self.main.x_mask_hall, 0], self.main.interpolated_hall_data[self.main.x_mask_hall, 1],
                                            initial_see, limits_see, initial_res, limits_res, initial_hall, limits_hall, int(self.main.Nv2_value_hall.text())/int(self.main.Nv1_value_hall.text()),
                                            f"{self.main.scattering_type_hall.currentText()}", self.main.threshold, self.main.max_iter, True)  # last argument is Nv2/Nv1
            self.main.params_2PB_hall = params
            if self.main.scattering_type_hall.currentText() == "acPhDis":
                print(f"params[0] = {params[0]}, params[1] = {params[1]}, params[2] = {params[2]}, params[3] = {params[3]}, params[4] = {params[4]}, params[5] = {params[5]}, params[6] = {params[6]},, params[7] = {params[7]}")
            else:
                print(f"params[0] = {params[0]}, params[1] = {params[1]}, params[2] = {params[2]}, params[3] = {params[3]}, params[4] = {params[4]}, params[5] = {params[5]}")
            print(params)

            fit_x_val, fit_y_val_see, fit_y_val_res, fit_y_val_hall = TE.dpb_see_res_hall_calc(self.main.minT, self.main.maxT, 50, [params[0], params[1], params[2]], 
                                                                    [params[3], params[4], params[5], params[6], params[7], params[8]], params[9], int(self.main.Nv2_value_hall.text())/int(self.main.Nv1_value_hall.text()), 
                                                                    scatter_type =f"{self.main.scattering_type_hall.currentText()}")
        except Exception as e:
            print(f"Exception: {e}")
            self.error_occurred.emit("dpb_hall")
            return

        self.updateGUI.emit(fit_x_val, fit_y_val_see, fit_y_val_res, fit_y_val_hall, params)

"""
Advanced options (set limits) windows
"""      
class SecondWindow_1PB_See(QWidget):
    def __init__(self, MainWindow):
        super().__init__()
        self.main = MainWindow
        self.initUI()

    
    def initUI(self):
        self.setWindowTitle('Advanced options')
        self.setGeometry(500, 200, 800, 600)  # x, y, width, height
        # Create a grid layout
        gridLayout = QGridLayout()
        self.T_changed = False
    
        # Input Widgets for lower and upper temperature boundaries of the fit
        onlyInt = QIntValidator()
        onlyInt.setRange(-10000, 10000)
        onlyFloat = QDoubleValidator()
        onlyFloat.setRange(-100000.00, 100000.00)
        self.min_value_label = QLabel("Lower fitting range")
        self.max_value_label = QLabel("Upper fitting range") 
        self.mass_label = QLabel("Limits mass")
        self.fermi_label = QLabel("Limits Fermi energy")

        self.min_value = QLineEdit(f"{self.main.minT}")
        self.min_value.textChanged.connect(self.set_T_changed)
        self.min_value.setValidator(onlyInt)
        self.mass_lower_limit = QLineEdit(f"{self.main.mass_slider_1PB.minimum()/100.}")
        self.fermi_lower_limit = QLineEdit(f"{self.main.fermi_slider_1PB.minimum()}")


        self.max_value_label = QLabel("Upper fitting range")
        self.max_value = QLineEdit(f"{self.main.maxT}")
        self.max_value.textChanged.connect(self.set_T_changed)
        self.max_value.setValidator(onlyInt)
        self.mass_upper_limit = QLineEdit(f"{self.main.mass_slider_1PB.maximum()/100.}")
        self.fermi_upper_limit = QLineEdit(f"{self.main.fermi_slider_1PB.maximum()}")

        self.fitrange_button = QPushButton("Set range")
        self.fitrange_button.clicked.connect(self.fitrange_button_clicked2)
        
        # Add buttons to the grid layout
        gridLayout.addWidget(self.mass_label, 0, 0)
        gridLayout.addWidget(self.mass_lower_limit, 0, 1)
        gridLayout.addWidget(self.mass_upper_limit, 0, 2)
        gridLayout.addWidget(self.fermi_label, 1, 0)
        gridLayout.addWidget(self.fermi_lower_limit, 1, 1)
        gridLayout.addWidget(self.fermi_upper_limit, 1, 2)
        gridLayout.addWidget(self.min_value_label, 2, 0)
        gridLayout.addWidget(self.max_value_label, 3, 0)
        gridLayout.addWidget(self.min_value, 2, 1)
        gridLayout.addWidget(self.max_value, 3, 1)
        gridLayout.addWidget(self.fitrange_button, 2,2)
        
        # Set the layout to the grid layout
        self.setLayout(gridLayout)
        self.show()
        
    def fitrange_button_clicked2(self):
        self.minT = int(self.min_value.text())
        self.maxT = int(self.max_value.text())
        self.mass_min = int(float(self.mass_lower_limit.text())*100)
        self.mass_max = int(float(self.mass_upper_limit.text())*100)
        self.fermi_min = int(float(self.fermi_lower_limit.text()))
        self.fermi_max = int(float(self.fermi_upper_limit.text()))
    
        self.main.fitrange_button_clicked_1PB(self.minT, self.maxT, self.mass_min, self.mass_max, self.fermi_min, self.fermi_max, self.T_changed)

    def set_T_changed(self):
        self.T_changed = True
  
class SecondWindow_1PB_Res(QWidget):
    def __init__(self, MainWindow):
        super().__init__()
        self.main = MainWindow
        self.initUI()

    
    def initUI(self):
        self.setWindowTitle('Advanced options')
        self.setGeometry(500, 200, 800, 600)  # x, y, width, height
        # Create a grid layout
        gridLayout = QGridLayout()
        self.T_changed = False
    
        # Input Widgets for lower and upper temperature boundaries of the fit
        onlyInt = QIntValidator()
        onlyInt.setRange(-10000, 10000)
        onlyFloat = QDoubleValidator()
        onlyFloat.setRange(-100000.00, 100000.00)
        self.min_value_label = QLabel("Lower fitting range")
        self.max_value_label = QLabel("Upper fitting range") 
        self.mass_label = QLabel("Electron mass")
        self.fermi_label = QLabel("Fermi energy")
        self.para_A_label = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,1</sub></p>")

        self.min_value = QLineEdit(f"{self.main.minT}")
        self.min_value.textChanged.connect(self.set_T_changed)
        self.min_value.setValidator(onlyInt)
        self.mass_lower_limit = QLineEdit(f"{self.main.mass_slider_1PB_res.minimum()/100.}")
        self.fermi_lower_limit = QLineEdit(f"{self.main.fermi_slider_1PB_res.minimum()}")
        self.para_A_lower_limit = QLineEdit(f"{self.main.para_A_slider_1PB_res.minimum()}")

        self.max_value_label = QLabel("Upper fitting range")
        self.max_value = QLineEdit(f"{self.main.maxT}")
        self.max_value.textChanged.connect(self.set_T_changed)
        self.max_value.setValidator(onlyInt)
        self.mass_upper_limit = QLineEdit(f"{self.main.mass_slider_1PB_res.maximum()/100.}")
        self.fermi_upper_limit = QLineEdit(f"{self.main.fermi_slider_1PB_res.maximum()}")
        self.para_A_upper_limit = QLineEdit(f"{self.main.para_A_slider_1PB_res.maximum()}")

        self.fitrange_button = QPushButton("Set range")
        self.fitrange_button.clicked.connect(self.fitrange_button_clicked2)
        
        self.label_params = QLabel("Parameter")
        self.label_params.setProperty("class","header")
        self.label_lower_limit = QLabel("Lower limit")
        self.label_lower_limit.setProperty("class","header")
        self.label_upper_limit = QLabel("Upper limit")
        self.label_upper_limit.setProperty("class","header")
        
        # Add buttons to the grid layout
        gridLayout.addWidget(self.label_params, 0, 0)
        gridLayout.addWidget(self.label_lower_limit, 0, 1)
        gridLayout.addWidget(self.label_upper_limit, 0, 2)
        gridLayout.addWidget(self.mass_label, 1, 0)
        gridLayout.addWidget(self.mass_lower_limit, 1, 1)
        gridLayout.addWidget(self.mass_upper_limit, 1, 2)
        gridLayout.addWidget(self.fermi_label, 2, 0)
        gridLayout.addWidget(self.fermi_lower_limit, 2, 1)
        gridLayout.addWidget(self.fermi_upper_limit, 2, 2)
        gridLayout.addWidget(self.para_A_label, 3, 0)
        gridLayout.addWidget(self.para_A_lower_limit, 3, 1)
        gridLayout.addWidget(self.para_A_upper_limit, 3, 2)
        gridLayout.addWidget(self.min_value_label, 4, 0)
        gridLayout.addWidget(self.max_value_label, 5, 0)
        gridLayout.addWidget(self.min_value, 4, 1)
        gridLayout.addWidget(self.max_value, 5, 1)
        gridLayout.addWidget(self.fitrange_button, 4,2)
        
        # Set the layout to the grid layout
        self.setLayout(gridLayout)
        self.show()
        
    def fitrange_button_clicked2(self):
        self.minT = int(self.min_value.text())
        self.maxT = int(self.max_value.text())
        self.mass_min = int(float(self.mass_lower_limit.text())*100)
        self.mass_max = int(float(self.mass_upper_limit.text())*100)
        self.fermi_min = int(float(self.fermi_lower_limit.text()))
        self.fermi_max = int(float(self.fermi_upper_limit.text()))
        self.para_A_min = int(float(self.para_A_lower_limit.text()))
        self.para_A_max = int(float(self.para_A_upper_limit.text()))
    
        self.main.fitrange_button_clicked_1PB_res(self.minT, self.maxT, self.mass_min, self.mass_max, self.fermi_min, self.fermi_max, self.para_A_min, self.para_A_max, self.T_changed)

    def set_T_changed(self):
        self.T_changed = True
     
class SecondWindow_1PB_Hall(QWidget):
    def __init__(self, MainWindow):
        super().__init__()
        self.main = MainWindow
        self.initUI()

    
    def initUI(self):
        self.setWindowTitle('Advanced options')
        self.setGeometry(500, 200, 800, 600)  # x, y, width, height
        # Create a grid layout
        gridLayout = QGridLayout()
        self.T_changed = False
    
        # Input Widgets for lower and upper temperature boundaries of the fit
        onlyInt = QIntValidator()
        onlyInt.setRange(-10000, 10000)
        onlyFloat = QDoubleValidator()
        onlyFloat.setRange(-100000.00, 100000.00)
        self.min_value_label = QLabel("Lower fitting range")
        self.max_value_label = QLabel("Upper fitting range") 
        self.mass_label = QLabel("Electron mass")
        self.fermi_label = QLabel("Fermi energy")
        self.para_A_label = QLabel("<p style='font-size: 14px;'>Limits &tau;<sub>ph,1</sub></p>")

        self.min_value = QLineEdit(f"{self.main.minT}")
        self.min_value.textChanged.connect(self.set_T_changed)
        self.min_value.setValidator(onlyInt)
        self.mass_lower_limit = QLineEdit(f"{self.main.mass_slider_1PB_hall.minimum()/100.}")
        self.fermi_lower_limit = QLineEdit(f"{self.main.fermi_slider_1PB_hall.minimum()}")
        self.para_A_lower_limit = QLineEdit(f"{self.main.para_A_slider_1PB_hall.minimum()}")

        self.max_value_label = QLabel("Upper fitting range")
        self.max_value = QLineEdit(f"{self.main.maxT}")
        self.max_value.textChanged.connect(self.set_T_changed)
        self.max_value.setValidator(onlyInt)
        self.mass_upper_limit = QLineEdit(f"{self.main.mass_slider_1PB_hall.maximum()/100.}")
        self.fermi_upper_limit = QLineEdit(f"{self.main.fermi_slider_1PB_hall.maximum()}")
        self.para_A_upper_limit = QLineEdit(f"{self.main.para_A_slider_1PB_hall.maximum()}")

        self.fitrange_button = QPushButton("Set range")
        self.fitrange_button.clicked.connect(self.fitrange_button_clicked2)
        
        self.label_params = QLabel("Parameter")
        self.label_params.setProperty("class","header")
        self.label_lower_limit = QLabel("Lower limit")
        self.label_lower_limit.setProperty("class","header")
        self.label_upper_limit = QLabel("Upper limit")
        self.label_upper_limit.setProperty("class","header")
        
        # Add buttons to the grid layout
        gridLayout.addWidget(self.label_params, 0, 0)
        gridLayout.addWidget(self.label_lower_limit, 0, 1)
        gridLayout.addWidget(self.label_upper_limit, 0, 2)
        gridLayout.addWidget(self.mass_label, 1, 0)
        gridLayout.addWidget(self.mass_lower_limit, 1, 1)
        gridLayout.addWidget(self.mass_upper_limit, 1, 2)
        gridLayout.addWidget(self.fermi_label, 2, 0)
        gridLayout.addWidget(self.fermi_lower_limit, 2, 1)
        gridLayout.addWidget(self.fermi_upper_limit, 2, 2)
        gridLayout.addWidget(self.para_A_label, 3, 0)
        gridLayout.addWidget(self.para_A_lower_limit, 3, 1)
        gridLayout.addWidget(self.para_A_upper_limit, 3, 2)
        gridLayout.addWidget(self.min_value_label, 4, 0)
        gridLayout.addWidget(self.max_value_label, 5, 0)
        gridLayout.addWidget(self.min_value, 4, 1)
        gridLayout.addWidget(self.max_value, 5, 1)
        gridLayout.addWidget(self.fitrange_button, 4,2)
        
        # Set the layout to the grid layout
        self.setLayout(gridLayout)
        self.show()
        
    def fitrange_button_clicked2(self):
        self.minT = int(self.min_value.text())
        self.maxT = int(self.max_value.text())
        self.mass_min = int(float(self.mass_lower_limit.text())*100)
        self.mass_max = int(float(self.mass_upper_limit.text())*100)
        self.fermi_min = int(float(self.fermi_lower_limit.text()))
        self.fermi_max = int(float(self.fermi_upper_limit.text()))
        self.para_A_min = int(float(self.para_A_lower_limit.text()))
        self.para_A_max = int(float(self.para_A_upper_limit.text()))
    
        self.main.fitrange_button_clicked_1PB_hall(self.minT, self.maxT, self.mass_min, self.mass_max, self.fermi_min, self.fermi_max, self.para_A_min, self.para_A_max, self.T_changed)

    def set_T_changed(self):
        self.T_changed = True
    
class SecondWindow_2PB_See(QWidget):
    def __init__(self, MainWindow):
        super().__init__()
        self.main = MainWindow
        self.initUI()

    
    def initUI(self):
        self.setWindowTitle('Advanced options')
        self.setGeometry(500, 200, 800, 600)  # x, y, width, height
        # Create a grid layout
        gridLayout = QGridLayout()
        self.T_changed = False
    
        # Input Widgets for lower and upper temperature boundaries of the fit
        onlyInt = QIntValidator()
        onlyInt.setRange(-10000, 10000)
        onlyFloat = QDoubleValidator()
        onlyFloat.setRange(-100000.00, 100000.00)
        self.min_value_label = QLabel("Lower fitting range")
        self.max_value_label = QLabel("Upper fitting range")
        self.mass_label = QLabel("<p style='font-size: 14px;'>m<sub>2</sub>/m<sub>1</sub> =</p>")
        self.bandgap_label = QLabel("Band gap")
        self.fermi_label = QLabel("Fermi energy")

        self.min_value = QLineEdit(f"{self.main.minT}")
        self.min_value.textChanged.connect(self.set_T_changed)
        self.min_value.setValidator(onlyInt)
        self.mass_lower_limit = QLineEdit(f"{self.main.mass_slider.minimum()/100.}")
        self.bandgap_lower_limit = QLineEdit(f"{self.main.bandgap_slider.minimum()}")
        self.fermi_lower_limit = QLineEdit(f"{self.main.fermi_slider.minimum()}")

        self.max_value = QLineEdit(f"{self.main.maxT}")
        self.max_value.textChanged.connect(self.set_T_changed)
        self.max_value.setValidator(onlyInt)
        self.mass_upper_limit = QLineEdit(f"{self.main.mass_slider.maximum()/100.}")
        self.bandgap_upper_limit = QLineEdit(f"{self.main.bandgap_slider.maximum()}")
        self.fermi_upper_limit = QLineEdit(f"{self.main.fermi_slider.maximum()}")

        self.fitrange_button = QPushButton("Set range")
        self.fitrange_button.clicked.connect(self.fitrange_button_clicked2)
        
        self.label_params = QLabel("Parameter")
        self.label_params.setProperty("class","header")
        self.label_lower_limit = QLabel("Lower limit")
        self.label_lower_limit.setProperty("class","header")
        self.label_upper_limit = QLabel("Upper limit")
        self.label_upper_limit.setProperty("class","header")
        
        # Add buttons to the grid layout
        gridLayout.addWidget(self.label_params, 0, 0)
        gridLayout.addWidget(self.label_lower_limit, 0, 1)
        gridLayout.addWidget(self.label_upper_limit, 0, 2)
        gridLayout.addWidget(self.mass_label, 1, 0)
        gridLayout.addWidget(self.mass_lower_limit, 1, 1)
        gridLayout.addWidget(self.mass_upper_limit, 1, 2)
        gridLayout.addWidget(self.bandgap_label, 2, 0)
        gridLayout.addWidget(self.bandgap_lower_limit, 2, 1)
        gridLayout.addWidget(self.bandgap_upper_limit, 2, 2)
        gridLayout.addWidget(self.fermi_label, 3, 0)
        gridLayout.addWidget(self.fermi_lower_limit, 3, 1)
        gridLayout.addWidget(self.fermi_upper_limit, 3, 2)
        gridLayout.addWidget(self.min_value_label, 4, 0)
        gridLayout.addWidget(self.max_value_label, 5, 0)
        gridLayout.addWidget(self.min_value, 4, 1)
        gridLayout.addWidget(self.max_value, 5, 1)
        gridLayout.addWidget(self.fitrange_button, 4,2)
        
        
        # Set the layout to the grid layout
        self.setLayout(gridLayout)
        
        self.show()
        
    def fitrange_button_clicked2(self):
        self.minT = int(self.min_value.text())
        self.maxT = int(self.max_value.text())
        self.mass_min = int(float(self.mass_lower_limit.text())*100)
        self.mass_max = int(float(self.mass_upper_limit.text())*100)
        self.bandgap_min = int(float(self.bandgap_lower_limit.text()))
        self.bandgap_max = int(float(self.bandgap_upper_limit.text()))
        self.fermi_min = int(float(self.fermi_lower_limit.text()))
        self.fermi_max = int(float(self.fermi_upper_limit.text()))

        self.main.fitrange_button_clicked(self.minT, self.maxT, self.mass_min, self.mass_max, self.bandgap_min, self.bandgap_max, self.fermi_min, self.fermi_max, self.T_changed)

    def set_T_changed(self):
        self.T_changed = True       
    
class SecondWindow_2PB_Res(QWidget):
    def __init__(self, MainWindow):
        super().__init__()
        self.main = MainWindow
        self.initUI()

    
    def initUI(self):
        self.setWindowTitle('Advanced options')
        self.setGeometry(500, 200, 800, 600)  # x, y, width, height
        # Create a grid layout
        gridLayout = QGridLayout()
        self.T_changed = False
        
        # Create buttons
        self.mass_label = QLabel("<p style='font-size: 14px;'>m<sub>2</sub>/m<sub>1</sub> =</p>")
        self.bandgap_label = QLabel("Band gap")
        self.fermi_label = QLabel("Fermi energy")
        self.para_A_label_res = QLabel("<p style='font-size: 14px;'>Limits &tau;<sub>ph,1</sub></p>")
        self.para_B_label_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,2</sub>/&tau;<sub>ph,1</sub></p>")
        self.para_C_label_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,1</sub>/&tau;<sub>dis,1</sub> =</p>")
        self.para_D_label_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,2</sub>/&tau;<sub>dis,2</sub></p>")
        self.para_F_label_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>dis,1</sub></p>")
        self.para_G_label_res = QLabel("<p style='font-size: 14px;'>&tau;<sub>dis,2</sub>/&tau;<sub>dis,1</sub></p>")

        self.mass_lower_limit = QLineEdit(f"{self.main.mass_slider_res.minimum()/100.}")
        self.bandgap_lower_limit = QLineEdit(f"{self.main.bandgap_slider_res.minimum()}")
        self.fermi_lower_limit = QLineEdit(f"{self.main.fermi_slider_res.minimum()}")
        self.para_A_lower_limit_res = QLineEdit(f"{self.main.para_A_slider_res.minimum()}")
        self.para_B_lower_limit_res = QLineEdit(f"{self.main.para_B_slider_res.minimum()/100.}")
        self.para_C_lower_limit_res = QLineEdit(f"{self.main.para_C_slider_res.minimum()/100.}")
        self.para_D_lower_limit_res = QLineEdit(f"{self.main.para_D_slider_res.minimum()/100.}")
        self.para_F_lower_limit_res = QLineEdit(f"{self.main.para_F_slider_res.minimum()}")
        self.para_G_lower_limit_res = QLineEdit(f"{self.main.para_G_slider_res.minimum()/100.}")
        
        self.mass_upper_limit = QLineEdit(f"{self.main.mass_slider_res.maximum()/100.}")
        self.bandgap_upper_limit = QLineEdit(f"{self.main.bandgap_slider_res.maximum()}")
        self.fermi_upper_limit = QLineEdit(f"{self.main.fermi_slider_res.maximum()}")
        self.para_A_upper_limit_res = QLineEdit(f"{self.main.para_A_slider_res.maximum()}")
        self.para_B_upper_limit_res = QLineEdit(f"{self.main.para_B_slider_res.maximum()/100.}")
        self.para_C_upper_limit_res = QLineEdit(f"{self.main.para_C_slider_res.maximum()/100.}")
        self.para_D_upper_limit_res = QLineEdit(f"{self.main.para_D_slider_res.maximum()/100.}")
        self.para_F_upper_limit_res = QLineEdit(f"{self.main.para_F_slider_res.maximum()}")
        self.para_G_upper_limit_res = QLineEdit(f"{self.main.para_G_slider_res.maximum()/100.}")
        
        # Input Widgets for lower and upper temperature boundaries of the fit
        onlyInt = QIntValidator()
        onlyInt.setRange(-10000, 10000)
        onlyFloat = QDoubleValidator()
        onlyFloat.setRange(-100000.00, 100000.00)
        self.min_value_label = QLabel("Lower fitting range")
        self.max_value_label = QLabel("Upper fitting range")

        self.min_value = QLineEdit(f"{self.main.minT}")
        self.min_value.textChanged.connect(self.set_T_changed)
        self.min_value.setValidator(onlyInt)

        self.max_value = QLineEdit(f"{self.main.maxT}")
        self.max_value.textChanged.connect(self.set_T_changed)
        self.max_value.setValidator(onlyInt)

        self.fitrange_button = QPushButton("Set range")
        self.fitrange_button.clicked.connect(self.fitrange_button_clicked2)
        
        self.label_params = QLabel("Parameter")
        self.label_params.setProperty("class","header")
        self.label_lower_limit = QLabel("Lower limit")
        self.label_lower_limit.setProperty("class","header")
        self.label_upper_limit = QLabel("Upper limit")
        self.label_upper_limit.setProperty("class","header")
        
        # Add buttons to the grid layout
        gridLayout.addWidget(self.label_params, 0, 0)
        gridLayout.addWidget(self.label_lower_limit, 0, 1)
        gridLayout.addWidget(self.label_upper_limit, 0, 2)
        gridLayout.addWidget(self.mass_label, 1, 0)
        gridLayout.addWidget(self.mass_lower_limit, 1, 1)
        gridLayout.addWidget(self.mass_upper_limit, 1, 2)
        gridLayout.addWidget(self.bandgap_label, 2, 0)
        gridLayout.addWidget(self.bandgap_lower_limit, 2, 1)
        gridLayout.addWidget(self.bandgap_upper_limit, 2, 2)
        gridLayout.addWidget(self.fermi_label, 3, 0)
        gridLayout.addWidget(self.fermi_lower_limit, 3, 1)
        gridLayout.addWidget(self.fermi_upper_limit, 3, 2)
        gridLayout.addWidget(self.para_A_label_res, 4, 0)
        gridLayout.addWidget(self.para_A_lower_limit_res, 4, 1)
        gridLayout.addWidget(self.para_A_upper_limit_res, 4, 2)
        gridLayout.addWidget(self.para_B_label_res, 5, 0)
        gridLayout.addWidget(self.para_B_lower_limit_res, 5, 1)
        gridLayout.addWidget(self.para_B_upper_limit_res, 5, 2)
        gridLayout.addWidget(self.para_C_label_res, 6, 0)
        gridLayout.addWidget(self.para_C_lower_limit_res, 6, 1)
        gridLayout.addWidget(self.para_C_upper_limit_res, 6, 2)
        gridLayout.addWidget(self.para_D_label_res, 7, 0)
        gridLayout.addWidget(self.para_D_lower_limit_res, 7, 1)
        gridLayout.addWidget(self.para_D_upper_limit_res, 7, 2)
        gridLayout.addWidget(self.para_F_label_res, 8, 0)
        gridLayout.addWidget(self.para_F_lower_limit_res, 8, 1)
        gridLayout.addWidget(self.para_F_upper_limit_res, 8, 2)
        gridLayout.addWidget(self.para_G_label_res, 9, 0)
        gridLayout.addWidget(self.para_G_lower_limit_res, 9, 1)
        gridLayout.addWidget(self.para_G_upper_limit_res, 9, 2)
        gridLayout.addWidget(self.min_value_label, 10, 0)
        gridLayout.addWidget(self.max_value_label, 11, 0)
        gridLayout.addWidget(self.min_value, 10, 1)
        gridLayout.addWidget(self.max_value, 11, 1)
        gridLayout.addWidget(self.fitrange_button, 10,2)
         
        # Set the layout to the grid layout
        self.setLayout(gridLayout)
        self.show()
        
    def fitrange_button_clicked2(self):
        self.minT = int(self.min_value.text())
        self.maxT = int(self.max_value.text())
        self.mass_min = int(float(self.mass_lower_limit.text())*100)
        self.mass_max = int(float(self.mass_upper_limit.text())*100)
        self.bandgap_min = int(float(self.bandgap_lower_limit.text()))
        self.bandgap_max = int(float(self.bandgap_upper_limit.text()))
        self.fermi_min = int(float(self.fermi_lower_limit.text()))
        self.fermi_max = int(float(self.fermi_upper_limit.text()))

        self.para_A_min = int(float(self.para_A_lower_limit_res.text()))
        self.para_B_min = int(float(self.para_B_lower_limit_res.text())*100)
        self.para_C_min = int(float(self.para_C_lower_limit_res.text())*100)
        self.para_D_min = int(float(self.para_D_lower_limit_res.text())*100)
        self.para_F_min = int(float(self.para_F_lower_limit_res.text()))
        self.para_G_min = int(float(self.para_G_lower_limit_res.text())*100)
        self.para_A_max = int(float(self.para_A_upper_limit_res.text()))
        self.para_B_max = int(float(self.para_B_upper_limit_res.text())*100)
        self.para_C_max = int(float(self.para_C_upper_limit_res.text())*100)
        self.para_D_max = int(float(self.para_D_upper_limit_res.text())*100)
        self.para_F_max = int(float(self.para_F_upper_limit_res.text()))
        self.para_G_max = int(float(self.para_G_upper_limit_res.text())*100)
        
        
        self.main.fitrange_button_clicked_res(self.minT, self.maxT, self.mass_min, self.mass_max, self.bandgap_min, self.bandgap_max, self.fermi_min, self.fermi_max, self.para_A_min, self.para_B_min, self.para_C_min, self.para_D_min, self.para_F_min, self.para_G_min, self.para_A_max, 
                                          self.para_B_max, self.para_C_max, self.para_D_max, self.para_F_max, self.para_G_max, self.T_changed)
    def set_T_changed(self):
        self.T_changed = True
     
class SecondWindow_2PB_Hall(QWidget):
    def __init__(self, MainWindow):
        super().__init__()
        self.main = MainWindow
        self.initUI()

    
    def initUI(self):
        self.setWindowTitle('Advanced options')
        self.setGeometry(500, 200, 800, 600)  # x, y, width, height
        # Create a grid layout
        gridLayout = QGridLayout()
        
        self.T_changed = False
        
        # Create buttons
        self.mass_label = QLabel("<p style='font-size: 14px;'>m<sub>2</sub>/m<sub>1</sub> =</p>")
        self.bandgap_label = QLabel("Band gap")
        self.fermi_label = QLabel("Fermi energy")
        self.para_A_label_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,1</sub></p>")
        self.para_B_label_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,2</sub>/&tau;<sub>ph,1</sub></p>")
        self.para_C_label_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,1</sub>/&tau;<sub>dis,1</sub> =</p>")
        self.para_D_label_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>ph,2</sub>/&tau;<sub>dis,2</sub></p>")
        self.para_E_label_hall = QLabel("<p style='font-size: 14px;'>m<sub>1</sub> [m<sub>e</sub>] </p>")
        self.para_F_label_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>dis,1</sub></p>")
        self.para_G_label_hall = QLabel("<p style='font-size: 14px;'>&tau;<sub>dis,2</sub>/ &tau;<sub>dis,1</sub></p>")
        
        self.mass_lower_limit = QLineEdit(f"{self.main.mass_slider_hall.minimum()/100.}")
        self.bandgap_lower_limit = QLineEdit(f"{self.main.bandgap_slider_hall.minimum()}")
        self.fermi_lower_limit = QLineEdit(f"{self.main.fermi_slider_hall.minimum()}")
        self.para_A_lower_limit_hall = QLineEdit(f"{self.main.para_A_slider_hall.minimum()}")
        self.para_B_lower_limit_hall = QLineEdit(f"{self.main.para_B_slider_hall.minimum()/100.}")
        self.para_C_lower_limit_hall = QLineEdit(f"{self.main.para_C_slider_hall.minimum()/100.}")
        self.para_D_lower_limit_hall = QLineEdit(f"{self.main.para_D_slider_hall.minimum()/100.}")
        self.para_E_lower_limit_hall = QLineEdit(f"{self.main.para_E_slider_hall.minimum()/100.}")
        self.para_F_lower_limit_hall = QLineEdit(f"{self.main.para_F_slider_hall.minimum()}")
        self.para_G_lower_limit_hall = QLineEdit(f"{self.main.para_G_slider_hall.minimum()/100.}")
        
        self.mass_upper_limit = QLineEdit(f"{self.main.mass_slider_hall.maximum()/100.}")
        self.bandgap_upper_limit = QLineEdit(f"{self.main.bandgap_slider_hall.maximum()}")
        self.fermi_upper_limit = QLineEdit(f"{self.main.fermi_slider_hall.maximum()}")
        self.para_A_upper_limit_hall = QLineEdit(f"{self.main.para_A_slider_hall.maximum()}")
        self.para_B_upper_limit_hall = QLineEdit(f"{self.main.para_B_slider_hall.maximum()/100.}")
        self.para_C_upper_limit_hall = QLineEdit(f"{self.main.para_C_slider_hall.maximum()/100.}")
        self.para_D_upper_limit_hall = QLineEdit(f"{self.main.para_D_slider_hall.maximum()/100.}")
        self.para_E_upper_limit_hall = QLineEdit(f"{self.main.para_E_slider_hall.maximum()/100.}")
        self.para_F_upper_limit_hall = QLineEdit(f"{self.main.para_F_slider_hall.maximum()/100.}")
        self.para_G_upper_limit_hall = QLineEdit(f"{self.main.para_G_slider_hall.maximum()/100.}")                
        
        # Input Widgets for lower and upper temperature boundaries of the fit
        onlyInt = QIntValidator()
        onlyInt.setRange(-10000, 10000)
        onlyFloat = QDoubleValidator()
        onlyFloat.setRange(-100000.00, 100000.00)
        self.min_value_label = QLabel("Lower fitting range")
        self.max_value_label = QLabel("Upper fitting range")

        self.min_value = QLineEdit(f"{self.main.minT}")
        # self.min_value.textChanged.connect(self.boundary_value_changed)
        self.min_value.setValidator(onlyInt)
        self.min_value.textChanged.connect(self.set_T_changed)

        self.max_value = QLineEdit(f"{self.main.maxT}")
        # self.max_value.textChanged.connect(self.boundary_value_changed)
        self.max_value.setValidator(onlyInt)
        self.max_value.textChanged.connect(self.set_T_changed)

        self.fitrange_button = QPushButton("Set range")
        self.fitrange_button.clicked.connect(self.fitrange_button_clicked2_hall)
        
        
        self.label_params = QLabel("Parameter")
        self.label_params.setProperty("class","header")
        self.label_lower_limit = QLabel("Lower limit")
        self.label_lower_limit.setProperty("class","header")
        self.label_upper_limit = QLabel("Upper limit")
        self.label_upper_limit.setProperty("class","header")
        
        # Add buttons to the grid layout
        gridLayout.addWidget(self.label_params, 0, 0)
        gridLayout.addWidget(self.label_lower_limit, 0, 1)
        gridLayout.addWidget(self.label_upper_limit, 0, 2)
        gridLayout.addWidget(self.mass_label, 1, 0)
        gridLayout.addWidget(self.mass_lower_limit, 1, 1)
        gridLayout.addWidget(self.mass_upper_limit, 1, 2)
        gridLayout.addWidget(self.bandgap_label, 2, 0)
        gridLayout.addWidget(self.bandgap_lower_limit, 2, 1)
        gridLayout.addWidget(self.bandgap_upper_limit, 2, 2)
        gridLayout.addWidget(self.fermi_label, 3, 0)
        gridLayout.addWidget(self.fermi_lower_limit, 3, 1)
        gridLayout.addWidget(self.fermi_upper_limit, 3, 2)
        gridLayout.addWidget(self.para_A_label_hall, 4, 0)
        gridLayout.addWidget(self.para_A_lower_limit_hall, 4, 1)
        gridLayout.addWidget(self.para_A_upper_limit_hall, 4, 2)
        gridLayout.addWidget(self.para_B_label_hall, 5, 0)
        gridLayout.addWidget(self.para_B_lower_limit_hall, 5, 1)
        gridLayout.addWidget(self.para_B_upper_limit_hall, 5, 2)
        gridLayout.addWidget(self.para_C_label_hall, 6, 0)
        gridLayout.addWidget(self.para_C_lower_limit_hall, 6, 1)
        gridLayout.addWidget(self.para_C_upper_limit_hall, 6, 2)
        gridLayout.addWidget(self.para_D_label_hall, 7, 0)
        gridLayout.addWidget(self.para_D_lower_limit_hall, 7, 1)
        gridLayout.addWidget(self.para_D_upper_limit_hall, 7, 2)
        gridLayout.addWidget(self.para_E_label_hall, 8, 0)
        gridLayout.addWidget(self.para_E_lower_limit_hall, 8, 1)
        gridLayout.addWidget(self.para_E_upper_limit_hall, 8, 2)
        gridLayout.addWidget(self.para_F_label_hall, 9, 0)
        gridLayout.addWidget(self.para_F_lower_limit_hall, 9, 1)
        gridLayout.addWidget(self.para_F_upper_limit_hall, 9, 2)
        gridLayout.addWidget(self.para_G_label_hall, 10, 0)
        gridLayout.addWidget(self.para_G_lower_limit_hall, 10, 1)
        gridLayout.addWidget(self.para_G_upper_limit_hall, 10, 2)
        gridLayout.addWidget(self.min_value_label, 11, 0)
        gridLayout.addWidget(self.max_value_label, 12, 0)
        gridLayout.addWidget(self.min_value, 11, 1)
        gridLayout.addWidget(self.max_value, 12, 1)
        gridLayout.addWidget(self.fitrange_button, 11,2)
        
        # Set the layout to the grid layout
        self.setLayout(gridLayout)
        self.show()
        
    def fitrange_button_clicked2_hall(self):
        self.minT = int(self.min_value.text())
        self.maxT = int(self.max_value.text())
        
        self.mass_min = int(float(self.mass_lower_limit.text())*100)
        self.mass_max = int(float(self.mass_upper_limit.text())*100)
        self.bandgap_min = int(float(self.bandgap_lower_limit.text()))
        self.bandgap_max = int(float(self.bandgap_upper_limit.text()))
        self.fermi_min = int(float(self.fermi_lower_limit.text()))
        self.fermi_max = int(float(self.fermi_upper_limit.text()))
        
        self.para_A_min = int(float(self.para_A_lower_limit_hall.text()))
        self.para_B_min = int(float(self.para_B_lower_limit_hall.text())*100)
        self.para_C_min = int(float(self.para_C_lower_limit_hall.text())*100)
        self.para_D_min = int(float(self.para_D_lower_limit_hall.text())*100)
        self.para_E_min = int(float(self.para_E_lower_limit_hall.text())*100)
        self.para_F_min = int(float(self.para_F_lower_limit_hall.text()))
        self.para_G_min = int(float(self.para_G_lower_limit_hall.text())*100)
        
        self.para_A_max = int(float(self.para_A_upper_limit_hall.text()))
        self.para_B_max = int(float(self.para_B_upper_limit_hall.text())*100)
        self.para_C_max = int(float(self.para_C_upper_limit_hall.text())*100)
        self.para_D_max = int(float(self.para_D_upper_limit_hall.text())*100)
        self.para_E_max = int(float(self.para_E_upper_limit_hall.text())*100)
        self.para_F_max = int(float(self.para_F_upper_limit_hall.text()))
        self.para_G_max = int(float(self.para_G_upper_limit_hall.text())*100)
        
        self.main.fitrange_button_clicked_hall(self.minT, self.maxT, self.mass_min, self.mass_max, self.bandgap_min, self.bandgap_max, self.fermi_min, self.fermi_max, self.para_A_min, self.para_B_min, self.para_C_min, self.para_D_min, self.para_E_min, self.para_F_min, self.para_G_min, self.para_A_max, 
                                          self.para_B_max, self.para_C_max, self.para_D_max, self.para_E_max, self.para_F_max, self.para_G_max, self.T_changed)
        
    def set_T_changed(self):
        self.T_changed = True

"""
Individual contribution windows
"""        
class AddGraphs_Window_1PB_See(QWidget):
    def __init__(self, MainWindow):
        super().__init__()
        self.main = MainWindow
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Additional graphs')
        self.setGeometry(500, 200, 800, 600)  # x, y, width, height
        
        # Create a grid layout
        self.gridLayout = QGridLayout()
        self.label_ind = QLabel("Additional graphs")
        self.label_ind.setProperty("class", "title")
        
        # Create various buttons
        self.print_button = QPushButton("Print to file")
        self.print_button.clicked.connect(self.print_button_clicked)
        
        self.plot_chemPot_button = QPushButton("Chemical potential")
        self.plot_chemPot_button.clicked.connect(self.plot_chemPot_button_clicked)

        # Main graph for the depiction of the individual contributions
        self.addInfo_graph = pg.PlotWidget()
        self.addInfo_graph.setBackground((255, 255, 255))
        self.addInfo_graph.setTitle(
            "<span style=\"color:white;font-size:12pt;font:Lato;font-weight:semi-bold\">Calculated Seebeck coefficient</span>")
        self.addInfo_graph.setProperty("class", "seebeck")
        self.pen_chemPot = pg.mkPen('#008200', width=3, style=Qt.SolidLine)
        
        # Show chemical potential when window is opened
        self.plot_chemPot_button_clicked()

        self.gridLayout.addWidget(self.label_ind, 0, 0)
        self.gridLayout.addWidget(self.plot_chemPot_button, 4, 0)
        self.gridLayout.addWidget(self.print_button, 0, 3)
        self.setLayout(self.gridLayout)
        
        self.show()
    
    def reset_graph_window(self):
        widget = self.gridLayout.itemAtPosition(1, 1)
        if(widget != None):
            self.gridLayout.removeWidget(widget.widget())
        
        self.addInfo_graph = pg.PlotWidget()
        self.addInfo_graph.setBackground((255, 255, 255))
        self.addInfo_graph.setProperty("class", "seebeck")
        
        self.gridLayout.addWidget(self.addInfo_graph, 1, 1, 10, 5)
    
    def plot_chemPot_button_clicked(self):
        self.active_graph = "ChemPot"
        self.reset_graph_window()
        
        xy_data = TE.spb_chemPot_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            self.main.mass_slider_1PB.value()/100., 
            self.main.fermi_slider_1PB.value()
            )
        
        self.x_values, self.y_values = xy_data[0], xy_data[1]
        
        plot_item = pg.PlotCurveItem()
        plot_item.setData(self.x_values, self.y_values, pen=self.pen_chemPot, clear=True)
        self.addInfo_graph.addItem(plot_item)
        
        legend = self.addInfo_graph.addLegend()
        legend.addItem(plot_item, "\u03b7")
        
        addInfo_graph_xAxis = self.addInfo_graph.getAxis("bottom")
        addInfo_graph_yAxis = self.addInfo_graph.getAxis("left")
        addInfo_graph_xAxis.setStyle(tickTextOffset=10)
        addInfo_graph_yAxis.setStyle(tickTextOffset=10)
        addInfo_graph_xAxis.setHeight(h=60)
        addInfo_graph_yAxis.setWidth(w=60)
        addInfo_graph_xAxis.setLabel(
            '<p style="font-size:14px;color=white">Temperature [K]</p>')
        addInfo_graph_yAxis.setLabel(
            '<p style="font-size:14px;color=white">Chemical potential [K]</p>')  
        return
    
    def print_button_clicked(self):
        if self.active_graph == "ChemPot":
            x_vals = np.insert(self.x_values, 0, 0)
            y_vals = np.insert(self.y_values, 0, self.main.fermi_slider_1PB.value())
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "Chemical_potential.txt", filter="Text file (*.txt);;All files (*.*)")
            dp.save_data(path = path, data = [x_vals, y_vals], x_legend = "Temperature [K]", y_legends = ["Chemical potential [K]"])
     
class AddGraphs_Window_1PB_Res(QWidget):
    def __init__(self, MainWindow):
        super().__init__()
        self.main = MainWindow
        self.initUI()
         
    def initUI(self):
        self.setWindowTitle('Additional graphs')
        self.setGeometry(500, 200, 800, 600)  # x, y, width, height
        
        # Create a grid layout
        self.gridLayout = QGridLayout()
        self.label_ind = QLabel("Additional graphs")
        self.label_ind.setProperty("class", "title")
        
        # Buttons for switching the graph to the wanted quantity
        self.print_button = QPushButton("Print to file")
        self.print_button.clicked.connect(self.print_button_clicked)

        self.plot_chemPot_button = QPushButton("Chemical potential")
        self.plot_chemPot_button.clicked.connect(self.plot_chempot_button_clicked)
        
        self.plot_elecCond_button = QPushButton("Electrical conductivity")
        self.plot_elecCond_button.clicked.connect(self.plot_elecCond_button_clicked)
        
        self.plot_thermCond_button = QPushButton("Thermal conductivity")
        self.plot_thermCond_button.clicked.connect(self.plot_thermCond_button_clicked)

        self.pen_chemPot = pg.mkPen("#008200", width=3, style=Qt.SolidLine)
        self.pen_elecCond = pg.mkPen("#008200", width=3, style=Qt.SolidLine)
        self.pen_thermCond = pg.mkPen("#008200", width=3, style=Qt.SolidLine)

        # Show chemical potential when window is opened
        self.plot_chempot_button_clicked()

        self.gridLayout.addWidget(self.label_ind, 0, 0)
        self.gridLayout.addWidget(self.plot_chemPot_button, 4, 0)
        self.gridLayout.addWidget(self.plot_elecCond_button, 5, 0)
        self.gridLayout.addWidget(self.plot_thermCond_button, 6, 0)
        self.gridLayout.addWidget(self.print_button, 0, 3)
        self.setLayout(self.gridLayout)
        
        self.show()
    
    def reset_graph_window(self):
        widget = self.gridLayout.itemAtPosition(1, 1)
        if(widget != None):
            self.gridLayout.removeWidget(widget.widget())
        
        self.addInfo_graph = pg.PlotWidget()
        self.addInfo_graph.setBackground((255, 255, 255))
        self.addInfo_graph.setProperty("class", "seebeck")
        
        self.gridLayout.addWidget(self.addInfo_graph, 1, 1, 10, 5)
    
    def plot_chempot_button_clicked(self):
        self.active_graph = "ChemPot"
        self.reset_graph_window()
        
        xy_data = TE.spb_chemPot_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            self.main.mass_slider_1PB_res.value()/100., 
            self.main.fermi_slider_1PB_res.value()
            )
        
        self.x_values, self.y_values = xy_data[0], xy_data[1]
        
        plot_item = pg.PlotCurveItem()
        plot_item.setData(self.x_values, self.y_values, pen=self.pen_chemPot, clear=True)
        self.addInfo_graph.addItem(plot_item)
        
        legend = self.addInfo_graph.addLegend()
        legend.addItem(plot_item, "\u03b7")
        
        addInfo_graph_xAxis = self.addInfo_graph.getAxis("bottom")
        addInfo_graph_yAxis = self.addInfo_graph.getAxis("left")
        addInfo_graph_xAxis.setStyle(tickTextOffset=10)
        addInfo_graph_yAxis.setStyle(tickTextOffset=10)
        addInfo_graph_xAxis.setHeight(h=60)
        addInfo_graph_yAxis.setWidth(w=60)
        addInfo_graph_xAxis.setLabel('<p style="font-size:14px;color=white">Temperature [K]</p>')
        addInfo_graph_yAxis.setLabel('<p style="font-size:14px;color=white">Chemical potential [K]</p>')  
        return
    
    def plot_elecCond_button_clicked(self):
        self.active_graph = "ElecCond"
        self.reset_graph_window()
        
        xy_data = TE.spb_elecCond_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            self.main.mass_slider_1PB_res.value()/100., 
            self.main.fermi_slider_1PB_res.value(),
            [self.main.para_A_slider_1PB_res.value(), self.main.para_C_slider_1PB_res.value()/100.,self.main.para_F_slider_1PB_res.value()],
            f"{self.main.scattering_type_1PB_res.currentText()}"
            )
        
        self.x_values, self.y_values = xy_data[0], xy_data[1]
        
        plot_item = pg.PlotCurveItem()
        plot_item.setData(self.x_values, self.y_values, pen=self.pen_elecCond, clear=True)
        self.addInfo_graph.addItem(plot_item)
        
        legend = self.addInfo_graph.addLegend()
        legend.addItem(plot_item, "\u03c3")
        
        addInfo_graph_xAxis = self.addInfo_graph.getAxis("bottom")
        addInfo_graph_yAxis = self.addInfo_graph.getAxis("left")
        addInfo_graph_xAxis.setStyle(tickTextOffset=10)
        addInfo_graph_yAxis.setStyle(tickTextOffset=10)
        addInfo_graph_xAxis.setHeight(h=60)
        addInfo_graph_yAxis.setWidth(w=60)
        addInfo_graph_xAxis.setLabel(
            '<p style="font-size:14px;color=white">Temperature [K]</p>')
        addInfo_graph_yAxis.setLabel(
            '<p style="font-size:14px;color=white">Electrical conductivity [S/m]</p>')  
        return
   
    def plot_thermCond_button_clicked(self):
        self.active_graph = "ThermCond"
        self.reset_graph_window()
        
        xy_data = TE.spb_thermCond_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            self.main.mass_slider_1PB_res.value()/100., 
            self.main.fermi_slider_1PB_res.value(),
            [self.main.para_A_slider_1PB_res.value(), self.main.para_C_slider_1PB_res.value()/100.,self.main.para_F_slider_1PB_res.value()],
            f"{self.main.scattering_type_1PB_res.currentText()}"
            )
        
        self.x_values, self.y_values = xy_data[0], xy_data[1]
        
        plot_item = pg.PlotCurveItem()
        plot_item.setData(self.x_values, self.y_values, pen=self.pen_thermCond, clear=True)
        self.addInfo_graph.addItem(plot_item)
        
        legend =  self.addInfo_graph.addLegend()
        legend.addItem(plot_item, "\u03ba")
        
        addInfo_graph_xAxis = self.addInfo_graph.getAxis("bottom")
        addInfo_graph_yAxis = self.addInfo_graph.getAxis("left")
        addInfo_graph_xAxis.setStyle(tickTextOffset=10)
        addInfo_graph_yAxis.setStyle(tickTextOffset=10)
        addInfo_graph_xAxis.setHeight(h=60)
        addInfo_graph_yAxis.setWidth(w=60)
        addInfo_graph_xAxis.setLabel(
            '<p style="font-size:14px;color=white">Temperature [K]</p>')
        addInfo_graph_yAxis.setLabel(
            '<p style="font-size:14px;color=white">Thermal conductivity [W/(mK)]</p>')  
        return
    
    def print_button_clicked(self):
        if self.active_graph == "ChemPot":
            x_vals = np.insert(self.x_values, 0, 0)
            y_vals = np.insert(self.y_values, 0, self.main.fermi_slider_1PB_res.value())
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "Chemical_potential.txt", filter="Text file (*.txt);;All files (*.*)")
            dp.save_data(path = path, data = [x_vals, y_vals], x_legend = "Temperature [K]", y_legends = ["Chemical potential [K]"])
        elif self.active_graph == "ElecCond":
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "Electrical_conductivity.txt", filter="Text file (*.txt);;All files (*.*)")
            dp.save_data(path = path, data = [self.x_values, self.y_values], x_legend = "Temperature [K]", y_legends = ["Electrical conductivity [S/m]"])
        elif self.active_graph == "ThermCond":
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "Thermal conductivity.txt", filter="Text file (*.txt);;All files (*.*)")
            dp.save_data(path = path, data = [self.x_values, self.y_values], x_legend = "Temperature [K]", y_legends = ["Thermal conductivity [W/(mK)]"])
       
class AddGraphs_Window_1PB_Hall(QWidget):
    def __init__(self, MainWindow):
        super().__init__()
        self.main = MainWindow
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Additional graphs')
        self.setGeometry(500, 200, 800, 600)  # x, y, width, height
        
        # Create a grid layout
        self.gridLayout = QGridLayout()
        self.label_ind = QLabel("Additional graphs")
        self.label_ind.setProperty("class", "title")
        
        # Buttons for switching the graph to the wanted quantity
        self.print_button = QPushButton("Print to file")
        self.print_button.clicked.connect(self.print_button_clicked)

        self.plot_chemPot_button = QPushButton("Chemical potential")
        self.plot_chemPot_button.clicked.connect(self.plot_chempot_button_clicked)
        
        self.plot_elecCond_button = QPushButton("Electrical conductivity")
        self.plot_elecCond_button.clicked.connect(self.plot_elecCond_button_clicked)
        
        self.plot_thermCond_button = QPushButton("Thermal conductivity")
        self.plot_thermCond_button.clicked.connect(self.plot_thermCond_button_clicked)

        # Main graph for the depiction of the individual contributions
        self.addInfo_graph = pg.PlotWidget()
        self.addInfo_graph.setBackground((255, 255, 255))
        self.addInfo_graph.setTitle(
            "<span style=\"color:white;font-size:12pt;font:Lato;font-weight:semi-bold\">Calculated Seebeck coefficient</span>")
        self.addInfo_graph.setProperty("class", "seebeck")
        self.pen_chemPot = pg.mkPen("#008200", width=3, style=Qt.SolidLine)
        self.pen_elecCond = pg.mkPen("#008200", width=3, style=Qt.SolidLine)
        self.pen_thermCond = pg.mkPen("#008200", width=3, style=Qt.SolidLine)

        # Show chemical potential when window is opened
        self.plot_chempot_button_clicked()

        self.gridLayout.addWidget(self.label_ind, 0, 0)
        self.gridLayout.addWidget(self.plot_chemPot_button, 4, 0)
        self.gridLayout.addWidget(self.plot_elecCond_button, 5, 0)
        self.gridLayout.addWidget(self.plot_thermCond_button, 6, 0)
        self.gridLayout.addWidget(self.print_button, 0, 3)
        self.setLayout(self.gridLayout)
        
        self.show()
    
    def reset_graph_window(self):
        widget = self.gridLayout.itemAtPosition(1, 1)
        if(widget != None):
            self.gridLayout.removeWidget(widget.widget())
        
        self.addInfo_graph = pg.PlotWidget()
        self.addInfo_graph.setBackground((255, 255, 255))
        self.addInfo_graph.setProperty("class", "seebeck")
        
        self.gridLayout.addWidget(self.addInfo_graph, 1, 1, 10, 5)
    
    def plot_chempot_button_clicked(self):
        self.active_graph = "ChemPot"
        self.reset_graph_window()
        
        xy_data = TE.spb_chemPot_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            self.main.mass_slider_1PB_hall.value()/100., 
            self.main.fermi_slider_1PB_hall.value()
            )
        
        self.x_values, self.y_values = xy_data[0], xy_data[1]
        
        plot_item = pg.PlotCurveItem()
        plot_item.setData(self.x_values, self.y_values, pen=self.pen_chemPot, clear=True)
        self.addInfo_graph.addItem(plot_item)
        
        legend = self.addInfo_graph.addLegend()
        legend.addItem(plot_item, "\u03b7")
        
        addInfo_graph_xAxis = self.addInfo_graph.getAxis("bottom")
        addInfo_graph_yAxis = self.addInfo_graph.getAxis("left")
        addInfo_graph_xAxis.setStyle(tickTextOffset=10)
        addInfo_graph_yAxis.setStyle(tickTextOffset=10)
        addInfo_graph_xAxis.setHeight(h=60)
        addInfo_graph_yAxis.setWidth(w=60)
        addInfo_graph_xAxis.setLabel(
            '<p style="font-size:14px;color=white">Temperature [K]</p>')
        addInfo_graph_yAxis.setLabel(
            '<p style="font-size:14px;color=white">Chemical potential [K]</p>')  
        return
    
    def plot_elecCond_button_clicked(self):
        self.active_graph = "ElecCond"
        self.reset_graph_window()
        
        xy_data = TE.spb_elecCond_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            self.main.mass_slider_1PB_hall.value()/100., 
            self.main.fermi_slider_1PB_hall.value(),
            [self.main.para_A_slider_1PB_hall.value(), self.main.para_C_slider_1PB_hall.value()/100.,self.main.para_F_slider_1PB_hall.value()],
            f"{self.main.scattering_type_1PB_hall.currentText()}"
            )
        
        self.x_values, self.y_values = xy_data[0], xy_data[1]
        
        plot_item = pg.PlotCurveItem()
        plot_item.setData(self.x_values, self.y_values, pen=self.pen_elecCond, clear=True)
        self.addInfo_graph.addItem(plot_item)
        
        legend = self.addInfo_graph.addLegend()
        legend.addItem(plot_item, "\u03c3")
        
        addInfo_graph_xAxis = self.addInfo_graph.getAxis("bottom")
        addInfo_graph_yAxis = self.addInfo_graph.getAxis("left")
        addInfo_graph_xAxis.setStyle(tickTextOffset=10)
        addInfo_graph_yAxis.setStyle(tickTextOffset=10)
        addInfo_graph_xAxis.setHeight(h=60)
        addInfo_graph_yAxis.setWidth(w=60)
        addInfo_graph_xAxis.setLabel(
            '<p style="font-size:14px;color=white">Temperature [K]</p>')
        addInfo_graph_yAxis.setLabel(
            '<p style="font-size:14px;color=white">Electrical conductivity [S/m]</p>')  
        return
   
    def plot_thermCond_button_clicked(self):
        self.active_graph = "ThermCond"
        self.reset_graph_window()
        
        xy_data = TE.spb_thermCond_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            self.main.mass_slider_1PB_hall.value()/100., 
            self.main.fermi_slider_1PB_hall.value(),
            [self.main.para_A_slider_1PB_hall.value(), self.main.para_C_slider_1PB_hall.value()/100.,self.main.para_F_slider_1PB_hall.value()],
            f"{self.main.scattering_type_1PB_hall.currentText()}"
            )
        
        self.x_values, self.y_values = xy_data[0], xy_data[1]
        
        plot_item = pg.PlotCurveItem()
        plot_item.setData(self.x_values, self.y_values, pen=self.pen_thermCond, clear=True)
        self.addInfo_graph.addItem(plot_item)
        
        legend = self.addInfo_graph.addLegend()
        legend.addItem(plot_item, "\u03ba")
        
        addInfo_graph_xAxis = self.addInfo_graph.getAxis("bottom")
        addInfo_graph_yAxis = self.addInfo_graph.getAxis("left")
        addInfo_graph_xAxis.setStyle(tickTextOffset=10)
        addInfo_graph_yAxis.setStyle(tickTextOffset=10)
        addInfo_graph_xAxis.setHeight(h=60)
        addInfo_graph_yAxis.setWidth(w=60)
        addInfo_graph_xAxis.setLabel(
            '<p style="font-size:14px;color=white">Temperature [K]</p>')
        addInfo_graph_yAxis.setLabel(
            '<p style="font-size:14px;color=white">Thermal conductivity [W/(mK)]</p>')  
        return
    
    def print_button_clicked(self):
        if self.active_graph == "ChemPot":
            x_vals = np.insert(self.x_values, 0, 0)
            y_vals = np.insert(self.y_values, 0, self.main.fermi_slider_1PB_hall.value())
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "Chemical_potential.txt", filter="Text file (*.txt);;All files (*.*)")
            dp.save_data(path = path, data = [x_vals, y_vals], x_legend = "Temperature [K]", y_legends = ["Chemical potential [K]"])
        elif self.active_graph == "ElecCond":
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "Electrical_conductivity.txt", filter="Text file (*.txt);;All files (*.*)")
            dp.save_data(path = path, data = [self.x_values, self.y_values], x_legend = "Temperature [K]", y_legends = ["Electrical conductivity [S/m]"])
        elif self.active_graph == "ThermCond":
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "Thermal conductivity.txt", filter="Text file (*.txt);;All files (*.*)")
            dp.save_data(path = path, data = [self.x_values, self.y_values], x_legend = "Temperature [K]", y_legends = ["Thermal conductivity [W/(mK)]"])
       
class AddGraphs_Window_2PB_See(QWidget):
    def __init__(self, MainWindow):
        super().__init__()
        self.main = MainWindow
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Additional graphs')
        self.setGeometry(500, 200, 800, 600)  # x, y, width, height
        
        # Create a grid layout
        self.gridLayout = QGridLayout()
        self.label_ind = QLabel("Additional graphs")
        self.label_ind.setProperty("class", "title")
        
        # Create various buttons
        self.print_button = QPushButton("Print to file")
        self.print_button.clicked.connect(self.print_button_clicked)

        self.plot_ind_see_button = QPushButton("Seebeck coefficient")
        self.plot_ind_see_button.clicked.connect(self.plot_ind_see_button_clicked)
        
        self.plot_ind_charCon_button = QPushButton("Charge carrier conc.")
        self.plot_ind_charCon_button.clicked.connect(self.plot_ind_charCon_button_clicked)

        self.plot_chempot_button = QPushButton("Chemical potential")
        self.plot_chempot_button.clicked.connect(self.plot_chemPot_button_clicked)

        # Main graph for the depiction of the individual contributions
        self.addInfo_graph = pg.PlotWidget()
        self.addInfo_graph.setBackground((255, 255, 255))
        self.addInfo_graph.setTitle(
            "<span style=\"color:white;font-size:12pt;font:Lato;font-weight:semi-bold\">Calculated Seebeck coefficient</span>")
        self.addInfo_graph.setProperty("class", "seebeck")
        self.pen_ind1 = pg.mkPen('r', width=3, style=Qt.SolidLine)
        self.pen_ind2 = pg.mkPen("b", width=3, style=Qt.SolidLine)
        self.pen_tot = pg.mkPen("#606060", width=3, style=Qt.SolidLine)
        self.pen_chemPot = pg.mkPen('#008200', width=3, style=Qt.SolidLine)

        # Show chemical potential when window is opened
        self.plot_chemPot_button_clicked()

        self.gridLayout.addWidget(self.label_ind, 0, 0)
        self.gridLayout.addWidget(self.plot_chempot_button, 4, 0)
        self.gridLayout.addWidget(self.plot_ind_charCon_button, 5, 0)
        self.gridLayout.addWidget(self.plot_ind_see_button, 6, 0)
        self.gridLayout.addWidget(self.print_button, 0, 3)
        self.setLayout(self.gridLayout)
        
        self.show()
    
    def reset_graph_window(self):
        widget = self.gridLayout.itemAtPosition(1, 1)
        if(widget != None):
            self.gridLayout.removeWidget(widget.widget())
        
        self.addInfo_graph = pg.PlotWidget()
        self.addInfo_graph.setBackground((255, 255, 255))
        self.addInfo_graph.setProperty("class", "seebeck")
        
        self.gridLayout.addWidget(self.addInfo_graph, 1, 1, 10, 5)
               
    def plot_ind_see_button_clicked(self):
        self.active_graph = "See"
        self.reset_graph_window()
        
        xy_ind_data = TE.dpb_ind_see_seeOnly_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            [self.main.mass_slider.value()/100., self.main.bandgap_slider.value(), self.main.fermi_slider.value()], 
            float(self.main.Nv2_value.text())/float(self.main.Nv1_value.text())
            )
        
        self.x_values = xy_ind_data[0]
        self.y1_values, self.y2_values = xy_ind_data[1], xy_ind_data[2]
        
        xy_tot_data = TE.dpb_see_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            [self.main.mass_slider.value()/100., self.main.bandgap_slider.value(), self.main.fermi_slider.value()], 
            float(self.main.Nv2_value.text())/float(self.main.Nv1_value.text())
            )
        
        self.y_values = xy_tot_data[1]
         
        plot_item1 = pg.PlotCurveItem()
        plot_item1.setData(self.x_values, self.y1_values, pen=self.pen_ind1, clear=True)
        plot_item2 = pg.PlotCurveItem()
        plot_item2.setData(self.x_values, self.y2_values, pen=self.pen_ind2, clear=True)
        plot_item3 = pg.PlotCurveItem()
        plot_item3.setData(self.x_values, self.y_values, pen=self.pen_tot, clear=True)
        self.addInfo_graph.addItem(plot_item1)
        self.addInfo_graph.addItem(plot_item2)
        self.addInfo_graph.addItem(plot_item3)
        
        legend = self.addInfo_graph.addLegend()
        legend.addItem(plot_item1, "S1")
        legend.addItem(plot_item2, "S2")
        legend.addItem(plot_item3, "S total")
        
        addInfo_graph_xAxis = self.addInfo_graph.getAxis("bottom")
        addInfo_graph_yAxis = self.addInfo_graph.getAxis("left")
        addInfo_graph_xAxis.setStyle(tickTextOffset=10)
        addInfo_graph_yAxis.setStyle(tickTextOffset=10)
        addInfo_graph_xAxis.setHeight(h=60)
        addInfo_graph_yAxis.setWidth(w=60)
        addInfo_graph_xAxis.setLabel(
            '<p style="font-size:14px;color=white">Temperature [K]</p>')
        addInfo_graph_yAxis.setLabel(
            '<p style="font-size:14px;color=white">Seebeck coefficient [V/K]</p>')

        return
    
    def plot_ind_charCon_button_clicked(self):
        self.active_graph = "CharCon"
        self.reset_graph_window()
        
        xy_ind_data = TE.dpb_ind_carCon_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            [self.main.mass_slider.value()/100., self.main.bandgap_slider.value(), self.main.fermi_slider.value()], 
            float(self.main.Nv2_value.text())/float(self.main.Nv1_value.text())
            )
        
        self.x_values = xy_ind_data[0]
        self.y1_values, self.y2_values = xy_ind_data[1], xy_ind_data[2]
        
        plot_item1 = pg.PlotCurveItem()
        plot_item1.setData(self.x_values, self.y1_values, pen=self.pen_ind1, clear=True)
        plot_item2 = pg.PlotCurveItem()
        plot_item2.setData(self.x_values, self.y2_values, pen=self.pen_ind2, clear=True)
        self.addInfo_graph.addItem(plot_item1)
        self.addInfo_graph.addItem(plot_item2)
        
        legend = self.addInfo_graph.addLegend()
        legend.addItem(plot_item1, "n1")
        legend.addItem(plot_item2, "n2")
        
        addInfo_graph_xAxis = self.addInfo_graph.getAxis("bottom")
        addInfo_graph_yAxis = self.addInfo_graph.getAxis("left")
        addInfo_graph_xAxis.setStyle(tickTextOffset=10)
        addInfo_graph_yAxis.setStyle(tickTextOffset=10)
        addInfo_graph_xAxis.setHeight(h=60)
        addInfo_graph_yAxis.setWidth(w=60)
        addInfo_graph_xAxis.setLabel(
            '<p style="font-size:14px;color=white">Temperature [K]</p>')
        addInfo_graph_yAxis.setLabel(
            '<p style="font-size:14px;color=white">Charge carrier concentration [1/m^3]</p>')

        return
    
    def plot_chemPot_button_clicked(self):
        self.active_graph = "ChemPot"
        self.reset_graph_window()
        
        xy_data = TE.dpb_chemPot_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            [self.main.mass_slider.value()/100., self.main.bandgap_slider.value(), self.main.fermi_slider.value()], 
            float(self.main.Nv2_value.text())/float(self.main.Nv1_value.text())
            )
        
        self.x_values, self.y_values = xy_data[0], xy_data[1]
        
        plot_item = pg.PlotCurveItem()
        plot_item.setData(self.x_values, self.y_values, pen=self.pen_chemPot, clear=True)
        self.addInfo_graph.addItem(plot_item)
        
        legend = self.addInfo_graph.addLegend()
        legend.addItem(plot_item, "\u03b7")
        
        addInfo_graph_xAxis = self.addInfo_graph.getAxis("bottom")
        addInfo_graph_yAxis = self.addInfo_graph.getAxis("left")
        addInfo_graph_xAxis.setStyle(tickTextOffset=10)
        addInfo_graph_yAxis.setStyle(tickTextOffset=10)
        addInfo_graph_xAxis.setHeight(h=60)
        addInfo_graph_yAxis.setWidth(w=60)
        addInfo_graph_xAxis.setLabel(
            '<p style="font-size:14px;color=white">Temperature [K]</p>')
        addInfo_graph_yAxis.setLabel(
            '<p style="font-size:14px;color=white">Chemical potential [K]</p>')  

        return
    
    def print_button_clicked(self):
        if self.active_graph == "ChemPot":
            x_vals = np.insert(self.x_values, 0, 0)
            y_vals = np.insert(self.y_values, 0, self.main.fermi_slider.value())
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "Chemical_potential.txt", filter="Text file (*.txt);;All files (*.*)")
            dp.save_data(path = path, data = [x_vals, y_vals], x_legend = "Temperature [K]", y_legends = ["Chemical potential [K]"])
        elif self.active_graph == "See":
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "Seebeck_coefficient.txt", filter="Text file (*.txt);;All files (*.*)")
            dp.save_data(path = path, data = [self.x_values, self.y1_values, self.y2_values, self.y_values], x_legend = "Temperature [K]", y_legends = ["Seebeck coefficient band 1 [V/K]", "Seebeck coefficient band 2 [V/K]", "Total Seebeck coefficient [V/K]"])
        elif self.active_graph == "CharCon":
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "Charge_carrier_concentration.txt", filter="Text file (*.txt);;All files (*.*)")
            dp.save_data(path = path, data = [self.x_values, self.y1_values, self.y2_values], x_legend = "Temperature [K]", y_legends = ["Charge carrier concentration band 1 [1/m^3]", "Charge carrier concentration band 2 [1/m^3]"])

class AddGraphs_Window_2PB_Res(QWidget):
    def __init__(self, MainWindow):
        super().__init__()
        self.main = MainWindow
        # print("MainWindow passed to ThirdWindow:", self.main)
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Additional graphs')
        self.setGeometry(500, 200, 800, 600)  # x, y, width, height
        
        # Create a grid layout
        self.gridLayout = QGridLayout()
        self.label_ind = QLabel("Additional graphs")
        self.label_ind.setProperty("class", "title")
        
        # Create various buttons
        self.print_button = QPushButton("Print to file")
        self.print_button.clicked.connect(self.print_button_clicked)

        self.plot_ind_see_button = QPushButton("Seebeck coefficient")
        self.plot_ind_see_button.clicked.connect(self.plot_ind_see_button_clicked)
        
        self.plot_ind_charCon_button = QPushButton("Charge carrier conc.")
        self.plot_ind_charCon_button.clicked.connect(self.plot_ind_charCon_button_clicked)

        self.plot_chempot_button = QPushButton("Chemical potential")
        self.plot_chempot_button.clicked.connect(self.plot_chemPot_button_clicked)
        
        self.plot_ind_elecCond_button = QPushButton("Electrical conductivity")
        self.plot_ind_elecCond_button.clicked.connect(self.plot_ind_elecCond_button_clicked)
        
        self.plot_ind_thermCond_button = QPushButton("Thermal conductivity")
        self.plot_ind_thermCond_button.clicked.connect(self.plot_thermCond_button_clicked)

        # Main graph for the depiction of the individual contributions
        self.addInfo_graph = pg.PlotWidget()
        self.addInfo_graph.setBackground((255, 255, 255))
        self.addInfo_graph.setTitle(
            "<span style=\"color:white;font-size:12pt;font:Lato;font-weight:semi-bold\">Calculated Seebeck coefficient</span>")
        self.addInfo_graph.setProperty("class", "seebeck")
        self.pen_ind1 = pg.mkPen('r', width=3, style=Qt.SolidLine)
        self.pen_ind2 = pg.mkPen("b", width=3, style=Qt.SolidLine)
        self.pen_tot = pg.mkPen("#606060", width=3, style=Qt.SolidLine)
        self.pen_chemPot = pg.mkPen('#008200', width=3, style=Qt.SolidLine)

        # Show chemical potential when window is opened
        self.plot_chemPot_button_clicked()

        self.gridLayout.addWidget(self.label_ind, 0, 0)
        self.gridLayout.addWidget(self.plot_chempot_button, 4, 0)
        self.gridLayout.addWidget(self.plot_ind_charCon_button, 5, 0)
        self.gridLayout.addWidget(self.plot_ind_see_button, 6, 0)
        self.gridLayout.addWidget(self.plot_ind_elecCond_button, 7, 0)
        self.gridLayout.addWidget(self.plot_ind_thermCond_button, 8, 0)
        self.gridLayout.addWidget(self.print_button, 0, 3)
        self.setLayout(self.gridLayout)
        
        self.show()
    
    def reset_graph_window(self):
        widget = self.gridLayout.itemAtPosition(1, 1)
        if(widget != None):
            self.gridLayout.removeWidget(widget.widget())
        
        self.addInfo_graph = pg.PlotWidget()
        self.addInfo_graph.setBackground((255, 255, 255))
        self.addInfo_graph.setProperty("class", "seebeck")
        
        self.gridLayout.addWidget(self.addInfo_graph, 1, 1, 10, 5)
               
    def plot_ind_see_button_clicked(self):
        self.active_graph = "See"
        self.reset_graph_window()
        
        xy_ind_data = TE.dpb_ind_see_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            [self.main.mass_slider_res.value()/100., self.main.bandgap_slider_res.value(), self.main.fermi_slider_res.value()],
            [self.main.para_A_slider_res.value(), self.main.para_B_slider_res.value()/100., self.main.para_C_slider_res.value(), self.main.para_D_slider_res.value(), self.main.para_F_slider_res.value(), self.main.para_G_slider_res.value()/100.],
            float(self.main.Nv2_value_res.text())/float(self.main.Nv1_value_res.text()),
            f"{self.main.scattering_type_1PB_res.currentText()}"
            )
        
        self.x_values = xy_ind_data[0]
        self.y1_values, self.y2_values = xy_ind_data[1], xy_ind_data[2]
        
        xy_tot_data = TE.dpb_see_with_scatter_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            [self.main.mass_slider_res.value()/100., self.main.bandgap_slider_res.value(), self.main.fermi_slider_res.value()],
            [self.main.para_A_slider_res.value(), self.main.para_B_slider_res.value()/100., self.main.para_C_slider_res.value(), self.main.para_D_slider_res.value(), self.main.para_F_slider_res.value(), self.main.para_G_slider_res.value()/100.],
            float(self.main.Nv2_value_res.text())/float(self.main.Nv1_value_res.text()),
            f"{self.main.scattering_type_1PB_res.currentText()}"
            )
        
        self.y_values = xy_tot_data[1]
         
        plot_item1 = pg.PlotCurveItem()
        plot_item1.setData(self.x_values, self.y1_values, pen=self.pen_ind1, clear=True)
        plot_item2 = pg.PlotCurveItem()
        plot_item2.setData(self.x_values, self.y2_values, pen=self.pen_ind2, clear=True)
        plot_item3 = pg.PlotCurveItem()
        plot_item3.setData(self.x_values, self.y_values, pen=self.pen_tot, clear=True)
        self.addInfo_graph.addItem(plot_item1)
        self.addInfo_graph.addItem(plot_item2)
        self.addInfo_graph.addItem(plot_item3)
        
        legend = self.addInfo_graph.addLegend()
        legend.addItem(plot_item1, "S1")
        legend.addItem(plot_item2, "S2")
        legend.addItem(plot_item3, "S total")
        
        addInfo_graph_xAxis = self.addInfo_graph.getAxis("bottom")
        addInfo_graph_yAxis = self.addInfo_graph.getAxis("left")
        addInfo_graph_xAxis.setStyle(tickTextOffset=10)
        addInfo_graph_yAxis.setStyle(tickTextOffset=10)
        addInfo_graph_xAxis.setHeight(h=60)
        addInfo_graph_yAxis.setWidth(w=60)
        addInfo_graph_xAxis.setLabel(
            '<p style="font-size:14px;color=white">Temperature [K]</p>')
        addInfo_graph_yAxis.setLabel(
            '<p style="font-size:14px;color=white">Seebeck coefficient [V/K]</p>')

        return
    
    def plot_ind_charCon_button_clicked(self):
        self.active_graph = "CharCon"
        self.reset_graph_window()
        
        xy_ind_data = TE.dpb_ind_carCon_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            [self.main.mass_slider_res.value()/100., self.main.bandgap_slider_res.value(), self.main.fermi_slider_res.value()],
            float(self.main.Nv2_value_res.text())/float(self.main.Nv1_value_res.text()),
            )
        
        self.x_values = xy_ind_data[0]
        self.y1_values, self.y2_values = xy_ind_data[1], xy_ind_data[2]
        
        plot_item1 = pg.PlotCurveItem()
        plot_item1.setData(self.x_values, self.y1_values, pen=self.pen_ind1, clear=True)
        plot_item2 = pg.PlotCurveItem()
        plot_item2.setData(self.x_values, self.y2_values, pen=self.pen_ind2, clear=True)
        self.addInfo_graph.addItem(plot_item1)
        self.addInfo_graph.addItem(plot_item2)
        
        legend = self.addInfo_graph.addLegend()
        legend.addItem(plot_item1, "n1")
        legend.addItem(plot_item2, "n2")
        
        addInfo_graph_xAxis = self.addInfo_graph.getAxis("bottom")
        addInfo_graph_yAxis = self.addInfo_graph.getAxis("left")
        addInfo_graph_xAxis.setStyle(tickTextOffset=10)
        addInfo_graph_yAxis.setStyle(tickTextOffset=10)
        addInfo_graph_xAxis.setHeight(h=60)
        addInfo_graph_yAxis.setWidth(w=60)
        addInfo_graph_xAxis.setLabel(
            '<p style="font-size:14px;color=white">Temperature [K]</p>')
        addInfo_graph_yAxis.setLabel(
            '<p style="font-size:14px;color=white">Charge carrier concentration [1/m^3]</p>')

        return
    
    def plot_chemPot_button_clicked(self):
        self.active_graph = "ChemPot"
        self.reset_graph_window()
        
        xy_data = TE.dpb_chemPot_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            [self.main.mass_slider_res.value()/100., self.main.bandgap_slider_res.value(), self.main.fermi_slider_res.value()], 
            float(self.main.Nv2_value_res.text())/float(self.main.Nv1_value_res.text())
            )
        
        self.x_values, self.y_values = xy_data[0], xy_data[1]
        
        plot_item = pg.PlotCurveItem()
        plot_item.setData(self.x_values, self.y_values, pen=self.pen_chemPot, clear=True)   
        self.addInfo_graph.addItem(plot_item)
        
        legend = self.addInfo_graph.addLegend()
        legend.addItem(plot_item, "\u03b7")

        addInfo_graph_xAxis = self.addInfo_graph.getAxis("bottom")
        addInfo_graph_yAxis = self.addInfo_graph.getAxis("left")
        addInfo_graph_xAxis.setStyle(tickTextOffset=10)
        addInfo_graph_yAxis.setStyle(tickTextOffset=10)
        addInfo_graph_xAxis.setHeight(h=60)
        addInfo_graph_yAxis.setWidth(w=60)
        addInfo_graph_xAxis.setLabel(
            '<p style="font-size:14px;color=white">Temperature [K]</p>')
        addInfo_graph_yAxis.setLabel(
            '<p style="font-size:14px;color=white">Chemical potential [K]</p>')  

        return
    
    def plot_ind_elecCond_button_clicked(self):
        self.active_graph = "ElecCond"
        self.reset_graph_window()
        
        xy_ind_data = TE.dpb_ind_elecCond_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            [self.main.mass_slider_res.value()/100., self.main.bandgap_slider_res.value(), self.main.fermi_slider_res.value()],
            [self.main.para_A_slider_res.value(), self.main.para_B_slider_res.value()/100., self.main.para_C_slider_res.value(), self.main.para_D_slider_res.value(), self.main.para_F_slider_res.value(), self.main.para_G_slider_res.value()/100.],
            float(self.main.Nv2_value_res.text())/float(self.main.Nv1_value_res.text()),
            f"{self.main.scattering_type_1PB_res.currentText()}"
            )
        
        self.x_values = xy_ind_data[0]
        self.y1_values, self.y2_values = xy_ind_data[1], xy_ind_data[2]
        
        xy_tot_data = TE.dpb_elecCond_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            [self.main.mass_slider_res.value()/100., self.main.bandgap_slider_res.value(), self.main.fermi_slider_res.value()],
            [self.main.para_A_slider_res.value(), self.main.para_B_slider_res.value()/100., self.main.para_C_slider_res.value(), self.main.para_D_slider_res.value(), self.main.para_F_slider_res.value(), self.main.para_G_slider_res.value()/100.],
            float(self.main.Nv2_value_res.text())/float(self.main.Nv1_value_res.text()),
            f"{self.main.scattering_type_1PB_res.currentText()}"
            )
        
        self.y_values = xy_tot_data[1]

        plot_item1 = pg.PlotCurveItem()
        plot_item1.setData(self.x_values, self.y1_values, pen=self.pen_ind1, clear=True)
        plot_item2 = pg.PlotCurveItem()
        plot_item2.setData(self.x_values, self.y2_values, pen=self.pen_ind2, clear=True)
        plot_item3 = pg.PlotCurveItem()
        plot_item3.setData(self.x_values, self.y_values, pen=self.pen_tot, clear=True) 
        self.addInfo_graph.addItem(plot_item1)
        self.addInfo_graph.addItem(plot_item2)
        self.addInfo_graph.addItem(plot_item3)
        
        legend = self.addInfo_graph.addLegend()
        legend.addItem(plot_item1, "\u03c31")
        legend.addItem(plot_item2, "\u03c32")
        legend.addItem(plot_item3, "\u03c3 total")
        
        addInfo_graph_xAxis = self.addInfo_graph.getAxis("bottom")
        addInfo_graph_yAxis = self.addInfo_graph.getAxis("left")
        addInfo_graph_xAxis.setStyle(tickTextOffset=10)
        addInfo_graph_yAxis.setStyle(tickTextOffset=10)
        addInfo_graph_xAxis.setHeight(h=60)
        addInfo_graph_yAxis.setWidth(w=60)
        addInfo_graph_xAxis.setLabel(
            '<p style="font-size:14px;color=white">Temperature [K]</p>')
        addInfo_graph_yAxis.setLabel(
            '<p style="font-size:14px;color=white">Electrical conductivity [S/m]</p>')

        return
   
    def plot_thermCond_button_clicked(self):
        self.active_graph = "ThermCond"
        self.reset_graph_window()
        
        xy_data = TE.dpb_thermCond_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            [self.main.mass_slider_res.value()/100., self.main.bandgap_slider_res.value(), self.main.fermi_slider_res.value()],
            [self.main.para_A_slider_res.value(), self.main.para_B_slider_res.value()/100., self.main.para_C_slider_res.value(), self.main.para_D_slider_res.value(), self.main.para_F_slider_res.value(), self.main.para_G_slider_res.value()/100.],
            float(self.main.Nv2_value_res.text())/float(self.main.Nv1_value_res.text()),
            f"{self.main.scattering_type_1PB_res.currentText()}"
            )
        
        self.x_values, self.y_values = xy_data[0], xy_data[1]

        plot_item = pg.PlotCurveItem()
        plot_item.setData(self.x_values, self.y_values, pen=self.pen_chemPot, clear=True)
        self.addInfo_graph.addItem(plot_item)

        legend = self.addInfo_graph.addLegend()
        legend.addItem(plot_item, "\u03ba")
        
        addInfo_graph_xAxis = self.addInfo_graph.getAxis("bottom")
        addInfo_graph_yAxis = self.addInfo_graph.getAxis("left")
        addInfo_graph_xAxis.setStyle(tickTextOffset=10)
        addInfo_graph_yAxis.setStyle(tickTextOffset=10)
        addInfo_graph_xAxis.setHeight(h=60)
        addInfo_graph_yAxis.setWidth(w=60)
        addInfo_graph_xAxis.setLabel(
            '<p style="font-size:14px;color=white">Temperature [K]</p>')
        addInfo_graph_yAxis.setLabel(
            '<p style="font-size:14px;color=white">Thermal conductivity [W/(mK)]</p>')

        return
    
    def print_button_clicked(self):
        if self.active_graph == "ChemPot":
            x_vals = np.insert(self.x_values, 0, 0)
            y_vals = np.insert(self.y_values, 0, self.main.fermi_slider_res.value())
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "Chemical_potential.txt", filter="Text file (*.txt);;All files (*.*)")
            dp.save_data(path = path, data = [x_vals, y_vals], x_legend = "Temperature [K]", y_legends = ["Chemical potential [K]"])
        elif self.active_graph == "See":
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "Seebeck_coefficient.txt", filter="Text file (*.txt);;All files (*.*)")
            dp.save_data(path = path, data = [self.x_values, self.y1_values, self.y2_values, self.y_values], x_legend = "Temperature [K]", y_legends = ["Seebeck coefficient band 1 [V/K]", "Seebeck coefficient band 2 [V/K]", "Total Seebeck coefficient [V/K]"])
        elif self.active_graph == "CharCon":
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "Charge_carrier_concentration.txt", filter="Text file (*.txt);;All files (*.*)")
            dp.save_data(path = path, data = [self.x_values, self.y1_values, self.y2_values], x_legend = "Temperature [K]", y_legends = ["Charge carrier concentration band 1 [1/m^3]", "Charge carrier concentration band 2 [1/m^3]"])
        elif self.active_graph == "ElecCond":
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "Electrical_conductivity.txt", filter="Text file (*.txt);;All files (*.*)")
            dp.save_data(path = path, data = [self.x_values, self.y1_values, self.y2_values, self.y_values], x_legend = "Temperature [K]", y_legends = ["Electrical conductivity band 1 [S/m]", "Electrical conductivity band 2 [S/m]", "Total electrical conductivity [S/m]"])
        elif self.active_graph == "ThermCond":
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "Thermal_conductivity.txt", filter="Text file (*.txt);;All files (*.*)")
            dp.save_data(path = path, data = [self.x_values, self.y_values], x_legend = "Temperature [K]", y_legends = ["Thermal conductivity [W/(mK)]"])

class AddGraphs_Window_2PB_Hall(QWidget):
    def __init__(self, MainWindow):
        super().__init__()
        self.main = MainWindow
        # print("MainWindow passed to ThirdWindow:", self.main)
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Additional graphs')
        self.setGeometry(500, 200, 800, 600)  # x, y, width, height
        
        # Create a grid layout
        self.gridLayout = QGridLayout()
        self.label_ind = QLabel("Additional graphs")
        self.label_ind.setProperty("class", "title")
        
        # Create various buttons
        self.print_button = QPushButton("Print to file")
        self.print_button.clicked.connect(self.print_button_clicked)

        self.plot_ind_see_button = QPushButton("Seebeck coefficient")
        self.plot_ind_see_button.clicked.connect(self.plot_ind_see_button_clicked)
        
        self.plot_ind_charCon_button = QPushButton("Charge carrier conc.")
        self.plot_ind_charCon_button.clicked.connect(self.plot_ind_charCon_button_clicked)

        self.plot_chempot_button = QPushButton("Chemical potential")
        self.plot_chempot_button.clicked.connect(self.plot_chemPot_button_clicked)
        
        self.plot_ind_elecCond_button = QPushButton("Electrical conductivity")
        self.plot_ind_elecCond_button.clicked.connect(self.plot_ind_elecCond_button_clicked)
        
        self.plot_ind_thermCond_button = QPushButton("Thermal conductivity")
        self.plot_ind_thermCond_button.clicked.connect(self.plot_thermCond_button_clicked)

        # Main graph for the depiction of the individual contributions
        self.addInfo_graph = pg.PlotWidget()
        self.addInfo_graph.setBackground((255, 255, 255))
        self.addInfo_graph.setTitle(
            "<span style=\"color:white;font-size:12pt;font:Lato;font-weight:semi-bold\">Calculated Seebeck coefficient</span>")
        self.addInfo_graph.setProperty("class", "seebeck")
        self.pen_ind1 = pg.mkPen('r', width=3, style=Qt.SolidLine)
        self.pen_ind2 = pg.mkPen("b", width=3, style=Qt.SolidLine)
        self.pen_tot = pg.mkPen("#606060", width=3, style=Qt.SolidLine)
        self.pen_chemPot = pg.mkPen('#008200', width=3, style=Qt.SolidLine)

        # Show chemical potential when window is opened
        self.plot_chemPot_button_clicked()

        self.gridLayout.addWidget(self.label_ind, 0, 0)
        self.gridLayout.addWidget(self.plot_chempot_button, 4, 0)
        self.gridLayout.addWidget(self.plot_ind_charCon_button, 5, 0)
        self.gridLayout.addWidget(self.plot_ind_see_button, 6, 0)
        self.gridLayout.addWidget(self.plot_ind_elecCond_button, 7, 0)
        self.gridLayout.addWidget(self.plot_ind_thermCond_button, 8, 0)
        self.gridLayout.addWidget(self.print_button, 0, 3)
        self.setLayout(self.gridLayout)
        
        self.show()
    
    def reset_graph_window(self):
        widget = self.gridLayout.itemAtPosition(1, 1)
        if(widget != None):
            self.gridLayout.removeWidget(widget.widget())
        
        self.addInfo_graph = pg.PlotWidget()
        self.addInfo_graph.setBackground((255, 255, 255))
        self.addInfo_graph.setProperty("class", "seebeck")
        
        self.gridLayout.addWidget(self.addInfo_graph, 1, 1, 10, 5)
               
    def plot_ind_see_button_clicked(self):
        self.active_graph = "See"
        self.reset_graph_window()
        
        xy_ind_data = TE.dpb_ind_see_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            [self.main.mass_slider_hall.value()/100., self.main.bandgap_slider_hall.value(), self.main.fermi_slider_hall.value()],
            [self.main.para_A_slider_hall.value(), self.main.para_B_slider_hall.value()/100., self.main.para_C_slider_hall.value(), self.main.para_D_slider_hall.value(), self.main.para_F_slider_hall.value(), self.main.para_G_slider_hall.value()/100.],
            float(self.main.Nv2_value_hall.text())/float(self.main.Nv1_value_hall.text()),
            f"{self.main.scattering_type_1PB_hall.currentText()}"
            )
        
        self.x_values = xy_ind_data[0]
        self.y1_values, self.y2_values = xy_ind_data[1], xy_ind_data[2]
        
        xy_tot_data = TE.dpb_see_with_scatter_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            [self.main.mass_slider_hall.value()/100., self.main.bandgap_slider_hall.value(), self.main.fermi_slider_hall.value()],
            [self.main.para_A_slider_hall.value(), self.main.para_B_slider_hall.value()/100., self.main.para_C_slider_hall.value(), self.main.para_D_slider_hall.value(), self.main.para_F_slider_hall.value(), self.main.para_G_slider_hall.value()/100.],
            float(self.main.Nv2_value_hall.text())/float(self.main.Nv1_value_hall.text()),
            f"{self.main.scattering_type_1PB_hall.currentText()}"
            )
        
        self.y_values = xy_tot_data[1]
         
        plot_item1 = pg.PlotCurveItem()
        plot_item1.setData(self.x_values, self.y1_values, pen=self.pen_ind1, clear=True)
        plot_item2 = pg.PlotCurveItem()
        plot_item2.setData(self.x_values, self.y2_values, pen=self.pen_ind2, clear=True)
        plot_item3 = pg.PlotCurveItem()
        plot_item3.setData(self.x_values, self.y_values, pen=self.pen_tot, clear=True)
        self.addInfo_graph.addItem(plot_item1)
        self.addInfo_graph.addItem(plot_item2)
        self.addInfo_graph.addItem(plot_item3)

        legend = self.addInfo_graph.addLegend()
        legend.addItem(plot_item1, "S1")
        legend.addItem(plot_item2, "S2")
        legend.addItem(plot_item3, "S total")
        
        addInfo_graph_xAxis = self.addInfo_graph.getAxis("bottom")
        addInfo_graph_yAxis = self.addInfo_graph.getAxis("left")
        addInfo_graph_xAxis.setStyle(tickTextOffset=10)
        addInfo_graph_yAxis.setStyle(tickTextOffset=10)
        addInfo_graph_xAxis.setHeight(h=60)
        addInfo_graph_yAxis.setWidth(w=60)
        addInfo_graph_xAxis.setLabel(
            '<p style="font-size:14px;color=white">Temperature [K]</p>')
        addInfo_graph_yAxis.setLabel(
            '<p style="font-size:14px;color=white">Seebeck coefficient [V/K]</p>')

        return
    
    def plot_ind_charCon_button_clicked(self):
        self.active_graph = "CharCon"
        self.reset_graph_window()
        
        xy_ind_data = TE.dpb_ind_carCon_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            [self.main.mass_slider_hall.value()/100., self.main.bandgap_slider_hall.value(), self.main.fermi_slider_hall.value()],
            float(self.main.Nv2_value_hall.text())/float(self.main.Nv1_value_hall.text()),
            )
        
        self.x_values = xy_ind_data[0]
        self.y1_values, self.y2_values = xy_ind_data[1], xy_ind_data[2]
        
        plot_item1 = pg.PlotCurveItem()
        plot_item1.setData(self.x_values, self.y1_values, pen=self.pen_ind1, clear=True)
        plot_item2 = pg.PlotCurveItem()
        plot_item2.setData(self.x_values, self.y2_values, pen=self.pen_ind2, clear=True) 
        self.addInfo_graph.addItem(plot_item1)
        self.addInfo_graph.addItem(plot_item2)
        
        legend = self.addInfo_graph.addLegend()
        legend.addItem(plot_item1, "n1")
        legend.addItem(plot_item2, "n2")

        
        addInfo_graph_xAxis = self.addInfo_graph.getAxis("bottom")
        addInfo_graph_yAxis = self.addInfo_graph.getAxis("left")
        addInfo_graph_xAxis.setStyle(tickTextOffset=10)
        addInfo_graph_yAxis.setStyle(tickTextOffset=10)
        addInfo_graph_xAxis.setHeight(h=60)
        addInfo_graph_yAxis.setWidth(w=60)
        addInfo_graph_xAxis.setLabel(
            '<p style="font-size:14px;color=white">Temperature [K]</p>')
        addInfo_graph_yAxis.setLabel(
            '<p style="font-size:14px;color=white">Charge carrier concentration [1/m^3]</p>')

        return
    
    def plot_chemPot_button_clicked(self):
        self.active_graph = "ChemPot"
        self.reset_graph_window()
        
        xy_data = TE.dpb_chemPot_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            [self.main.mass_slider_hall.value()/100., self.main.bandgap_slider_hall.value(), self.main.fermi_slider_hall.value()], 
            float(self.main.Nv2_value_hall.text())/float(self.main.Nv1_value_hall.text())
            )
        
        self.x_values, self.y_values = xy_data[0], xy_data[1]
        
        plot_item = pg.PlotCurveItem()
        plot_item.setData(self.x_values, self.y_values, pen=self.pen_chemPot, clear=True)  
        self.addInfo_graph.addItem(plot_item)
        
        legend = self.addInfo_graph.addLegend()
        legend.addItem(plot_item, "\u03b7")
        
        addInfo_graph_xAxis = self.addInfo_graph.getAxis("bottom")
        addInfo_graph_yAxis = self.addInfo_graph.getAxis("left")
        addInfo_graph_xAxis.setStyle(tickTextOffset=10)
        addInfo_graph_yAxis.setStyle(tickTextOffset=10)
        addInfo_graph_xAxis.setHeight(h=60)
        addInfo_graph_yAxis.setWidth(w=60)
        addInfo_graph_xAxis.setLabel(
            '<p style="font-size:14px;color=white">Temperature [K]</p>')
        addInfo_graph_yAxis.setLabel(
            '<p style="font-size:14px;color=white">Chemical potential [K]</p>')  

        return
    
    def plot_ind_elecCond_button_clicked(self):
        self.active_graph = "ElecCond"
        self.reset_graph_window()
        
        xy_ind_data = TE.dpb_ind_elecCond_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            [self.main.mass_slider_hall.value()/100., self.main.bandgap_slider_hall.value(), self.main.fermi_slider_hall.value()],
            [self.main.para_A_slider_hall.value(), self.main.para_B_slider_hall.value()/100., self.main.para_C_slider_hall.value(), self.main.para_D_slider_hall.value(), self.main.para_F_slider_hall.value(), self.main.para_G_slider_hall.value()/100.],
            float(self.main.Nv2_value_hall.text())/float(self.main.Nv1_value_hall.text()),
            f"{self.main.scattering_type_1PB_hall.currentText()}"
            )
        
        self.x_values = xy_ind_data[0]
        self.y1_values, self.y2_values = xy_ind_data[1], xy_ind_data[2]
        
        xy_tot_data = TE.dpb_elecCond_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            [self.main.mass_slider_hall.value()/100., self.main.bandgap_slider_hall.value(), self.main.fermi_slider_hall.value()],
            [self.main.para_A_slider_hall.value(), self.main.para_B_slider_hall.value()/100., self.main.para_C_slider_hall.value(), self.main.para_D_slider_hall.value(), self.main.para_F_slider_hall.value(), self.main.para_G_slider_hall.value()/100.],
            float(self.main.Nv2_value_hall.text())/float(self.main.Nv1_value_hall.text()),
            f"{self.main.scattering_type_1PB_hall.currentText()}"
            )
        
        self.y_values = xy_tot_data[1]
         
        plot_item1 = pg.PlotCurveItem()
        plot_item1.setData(self.x_values, self.y1_values, pen=self.pen_ind1, clear=True)
        plot_item2 = pg.PlotCurveItem()
        plot_item2.setData(self.x_values, self.y2_values, pen=self.pen_ind2, clear=True)
        plot_item3 = pg.PlotCurveItem()
        plot_item3.setData(self.x_values, self.y_values, pen=self.pen_tot, clear=True)     
        self.addInfo_graph.addItem(plot_item1)
        self.addInfo_graph.addItem(plot_item2)
        self.addInfo_graph.addItem(plot_item3)
        
        legend = self.addInfo_graph.addLegend()
        legend.addItem(plot_item1, "\u03c31")
        legend.addItem(plot_item2, "\u03c32")
        legend.addItem(plot_item3, "\u03c3 total")
        
        addInfo_graph_xAxis = self.addInfo_graph.getAxis("bottom")
        addInfo_graph_yAxis = self.addInfo_graph.getAxis("left")
        addInfo_graph_xAxis.setStyle(tickTextOffset=10)
        addInfo_graph_yAxis.setStyle(tickTextOffset=10)
        addInfo_graph_xAxis.setHeight(h=60)
        addInfo_graph_yAxis.setWidth(w=60)
        addInfo_graph_xAxis.setLabel(
            '<p style="font-size:14px;color=white">Temperature [K]</p>')
        addInfo_graph_yAxis.setLabel(
            '<p style="font-size:14px;color=white">Electrical conductivity [S/m]</p>')

        return
   
    def plot_thermCond_button_clicked(self):
        self.active_graph = "ThermCond"
        self.reset_graph_window()
        
        xy_data = TE.dpb_thermCond_calc(
            self.main.minT, 
            self.main.maxT, 
            50, 
            [self.main.mass_slider_hall.value()/100., self.main.bandgap_slider_hall.value(), self.main.fermi_slider_hall.value()],
            [self.main.para_A_slider_hall.value(), self.main.para_B_slider_hall.value()/100., self.main.para_C_slider_hall.value(), self.main.para_D_slider_hall.value(), self.main.para_F_slider_hall.value(), self.main.para_G_slider_hall.value()/100.],
            float(self.main.Nv2_value_hall.text())/float(self.main.Nv1_value_hall.text()),
            f"{self.main.scattering_type_1PB_hall.currentText()}"
            )
        
        self.x_values, self.y_values = xy_data[0], xy_data[1]
         
        plot_item = pg.PlotCurveItem()
        plot_item.setData(self.x_values, self.y_values, pen=self.pen_chemPot, clear=True)
        self.addInfo_graph.addItem(plot_item)

        legend = self.addInfo_graph.addLegend()
        legend.addItem(plot_item, "\u03ba")
        
        addInfo_graph_xAxis = self.addInfo_graph.getAxis("bottom")
        addInfo_graph_yAxis = self.addInfo_graph.getAxis("left")
        addInfo_graph_xAxis.setStyle(tickTextOffset=10)
        addInfo_graph_yAxis.setStyle(tickTextOffset=10)
        addInfo_graph_xAxis.setHeight(h=60)
        addInfo_graph_yAxis.setWidth(w=60)
        addInfo_graph_xAxis.setLabel(
            '<p style="font-size:14px;color=white">Temperature [K]</p>')
        addInfo_graph_yAxis.setLabel(
            '<p style="font-size:14px;color=white">Thermal conductivity [W/(mK)]</p>')

        return
    
    def print_button_clicked(self):
        if self.active_graph == "ChemPot":
            x_vals = np.insert(self.x_values, 0, 0)
            y_vals = np.insert(self.y_values, 0, self.main.fermi_slider_hall.value())
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "Chemical_potential.txt", filter="Text file (*.txt);;All files (*.*)")
            dp.save_data(path = path, data = [x_vals, y_vals], x_legend = "Temperature [K]", y_legends = ["Chemical potential [K]"])
        elif self.active_graph == "See":
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "Seebeck_coefficient.txt", filter="Text file (*.txt);;All files (*.*)")
            dp.save_data(path = path, data = [self.x_values, self.y1_values, self.y2_values, self.y_values], x_legend = "Temperature [K]", y_legends = ["Seebeck coefficient band 1 [V/K]", "Seebeck coefficient band 2 [V/K]", "Total Seebeck coefficient [V/K]"])
        elif self.active_graph == "CharCon":
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "Charge_carrier_concentration.txt", filter="Text file (*.txt);;All files (*.*)")
            dp.save_data(path = path, data = [self.x_values, self.y1_values, self.y2_values], x_legend = "Temperature [K]", y_legends = ["Charge carrier concentration band 1 [1/m^3]", "Charge carrier concentration band 2 [1/m^3]"])
        elif self.active_graph == "ElecCond":
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "Electrical_conductivity.txt", filter="Text file (*.txt);;All files (*.*)")
            dp.save_data(path = path, data = [self.x_values, self.y1_values, self.y2_values, self.y_values], x_legend = "Temperature [K]", y_legends = ["Electrical conductivity band 1 [S/m]", "Electrical conductivity band 2 [S/m]", "Total electrical conductivity [S/m]"])
        elif self.active_graph == "ThermCond":
            path, _ = QFileDialog.getSaveFileName(self, "Save as", directory = "Thermal_conductivity.txt", filter="Text file (*.txt);;All files (*.*)")
            dp.save_data(path = path, data = [self.x_values, self.y_values], x_legend = "Temperature [K]", y_legends = ["Thermal conductivity [W/(mK)]"])
        
'''
Main
'''

# Main to start the program

if __name__ == "__main__":

    TE = Fit_class_final.PB_fit()
    # TE.change_lambda(0)

    app = QApplication(sys.argv)
    with open("stylesheet.css", "r") as style:
        app.setStyleSheet(style.read())
    w = MainWindow()
    w.show()
    app.exec()