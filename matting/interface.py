import functools
import glob
import os
import os.path as pth
import pickle
import shutil
import sys
from functools import partial
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2
from skimage.measure import label as labeling
import sklearn.cluster as clustering
from sklearn.covariance import EllipticEnvelope

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from multi_material import MultiMaterial
from matting.patchmatch import inpaint
from matting.matting import svbrdf_matting, svbrdf_matting_with_mask, refined_matting

from matting.utils import center_crop_resize, mask_binarization, save_masks_as_image, load_masks, expand_dim_to_3, normalize
from matting.utils import normal2height, smooth_height_map, find_bounding_box, reduce_dim
from utils import save_image, load_image


def show_msg_box(title, text):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()


class Scroller(QScrollArea):
    def __init__(self):
        QScrollArea.__init__(self)

    def wheelEvent(self, event):
        if event.type() == QEvent.Wheel:
            event.ignore()


class MattingBase(QMainWindow):
    def __init__(self):
        super().__init__()

        self.root_dir = None
        self.work_dir = None

        # albedo/normal/roughness placeholders
        self.albedo = None
        self.normal = None
        self.roughness = None
        self.label_albedo = QLabel()
        self.label_albedo.setFixedSize(512, 512)
        self.label_albedo.setFrameShape(QFrame.Box)
        self.label_normal = QLabel()
        self.label_normal.setFixedSize(512, 512)
        self.label_normal.setFrameShape(QFrame.Box)
        self.label_roughness = QLabel()
        self.label_roughness.setFixedSize(512, 512)
        self.label_roughness.setFrameShape(QFrame.Box)

        """
        GUI layout definition
        """
        self.container = QWidget()
        self.layout = QGridLayout()
        self.scrollArea = Scroller()
        self.row_index = 3
        self.num_fixed_widget = 0

        # status bar
        self.status = self.statusBar()
        self.state = None
        self.scale_ratio = 1.0
        self.max_scale_ratio = 10.0

        # drawing flag
        self.drawing = False
        # QPoint object to tract drawing
        self.lastDrawingPoint = QPoint()
        # default color
        self.activeColor = 'blue'
        self.brushColor = Qt.blue
        # default brush size
        self.brushSize = 12
        # multiple layers based on the brush color
        self.layers = {}
        # foreground and background
        self.colors = {
            'blue': [0, 0, 255],
            'green': [0, 255, 0]
        }

        # panning flag
        self.panning = False
        # QPoint object to tract panning
        self.lastPanningPoint = QPoint()
        # offset of the current movement
        self.offset = QPoint(0, 0)

        # initialize undo buffer
        self.undoBuffer = []

    def updateLayout(self, widget, row, col, row_span=1, col_span=1, alignment=Qt.AlignCenter):
        self.layout.addWidget(widget, row, col, row_span, col_span, alignment)

    def clearLayout(self):
        while self.layout.count() > self.num_fixed_widget:
            child = self.layout.takeAt(self.num_fixed_widget)
            if child.widget():
                child.widget().deleteLater()
        # reset row index
        self.row_index = 3

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def show_status(self):
        self.status.showMessage('{:.2f}% | {}'.format(self.scale_ratio * 100, self.state))

    def mapFromMain(self, pos):
        pos = self.scrollArea.mapFromParent(pos)
        label = self.container.childAt(pos)
        if label is not None and (label == self.label_albedo or label == self.label_normal or label == self.label_roughness):
            pos = label.mapFromParent(pos)
            return pos
        else:
            return None

    def wheelEvent(self, event: QWheelEvent) -> None:
        local_pos = self.mapFromMain(event.pos())
        if local_pos is not None:
            if event.angleDelta().y() > 0:
                self.scale_ratio *= 1.1
                self.scale_ratio = min(self.max_scale_ratio, self.scale_ratio)
            else:
                self.scale_ratio *= 0.9
                self.scale_ratio = max(1.0, self.scale_ratio)
        self.show_status()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.albedo is None or self.normal is None or self.roughness is None:
            return

        # if left mouse button is pressed
        if event.button() == Qt.LeftButton:
            local_pos = self.mapFromMain(event.pos())
            if local_pos is None:
                return

            # make drawing flag true
            self.drawing = True
            # make last point to the point of cursor
            self.lastDrawingPoint = local_pos
        elif event.button() == Qt.RightButton:
            local_pos = self.mapFromMain(event.pos())
            if local_pos is None:
                return

            # make panning flag true
            self.panning = True
            # make last point to the point of cursor
            self.lastPanningPoint = local_pos
            QApplication.setOverrideCursor(Qt.OpenHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        # check if left button is pressed and drawing flag is true
        if (event.buttons() and Qt.LeftButton) and self.drawing:
            # compute local position
            local_pos = self.mapFromMain(event.pos())
            if local_pos is None:
                self.drawing = False
                return
            # creating painter object
            painter = QPainter(self.layers[self.activeColor])
            # compute the top left corner of the scaled canvas
            width = self.layers[self.activeColor].width()
            height = self.layers[self.activeColor].height()
            topLeft = QPoint((width - int(width / self.scale_ratio)) / 2, (height - int(height / self.scale_ratio)) / 2)
            topLeft -= self.offset
            transformed_local_pos = local_pos / self.scale_ratio + topLeft
            transformed_last_point = self.lastDrawingPoint / self.scale_ratio + topLeft
            # set the pen of the painter
            painter.setPen(QPen(self.brushColor, self.brushSize / self.scale_ratio, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))

            # draw line from the last point of cursor to the current point
            painter.drawLine(transformed_last_point, transformed_local_pos)
            # change the last point
            self.lastDrawingPoint = local_pos
            # update
            self.update()
        elif (event.buttons() and Qt.RightButton) and self.panning:
            # compute local position
            local_pos = self.mapFromMain(event.pos())
            if local_pos is None:
                self.drawing = False
                return

            self.offset += (local_pos - self.lastPanningPoint) / self.scale_ratio

            # change the last point
            self.lastPanningPoint = local_pos
            # update
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            # make drawing flag false
            self.drawing = False
            # push the current state into the undo buffer
            self.undoBuffer.append({key: value.copy() for (key, value) in self.layers.items()})
        elif event.button() == Qt.RightButton:
            self.panning = False
            QApplication.setOverrideCursor(Qt.ArrowCursor)

    def paintOnImage(self, im: QPixmap) -> QPixmap:
        image = QPixmap(im.size())
        image.fill(Qt.transparent)

        painter = QPainter(image)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        # scale image
        scaled_width = int(im.width() / self.scale_ratio)
        scaled_height = int(im.height() / self.scale_ratio)
        # compute the initial cropped area
        topLeft = QPoint((im.width() - scaled_width) / 2, (im.height() - scaled_height) / 2)
        if abs(self.offset.x()) > (im.width() - scaled_width) / 2:
            if self.offset.x() > 0:
                self.offset.setX((im.width() - scaled_width) / 2)
            else:
                self.offset.setX(-1 * (im.width() - scaled_width) / 2)

        if abs(self.offset.y()) > (im.height() - scaled_height) / 2:
            if self.offset.y() > 0:
                self.offset.setY((im.height() - scaled_height) / 2)
            else:
                self.offset.setY(-1 * (im.height() - scaled_height) / 2)
        # add offset to the cropped area
        topLeft -= self.offset
        area = QRect(topLeft.x(), topLeft.y(), scaled_width, scaled_height)
        # crop image
        cropped_im = im.copy(area)
        cropped_im = cropped_im.scaled(im.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter.drawPixmap(im.rect(), cropped_im)

        # render multiple pixel maps
        for layer in self.layers.values():
            cropped_layer = layer.copy(area)
            cropped_layer = cropped_layer.scaled(im.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(im.rect(), cropped_layer)
        return image

    def paintEvent(self, event: QPaintEvent) -> None:
        if self.albedo is not None:
            im = self.paintOnImage(self.albedo)
            self.label_albedo.setPixmap(im)

        if self.normal is not None:
            im = self.paintOnImage(self.normal)
            self.label_normal.setPixmap(im)

        if self.roughness is not None:
            im = self.paintOnImage(self.roughness)
            self.label_roughness.setPixmap(im)

    def undo(self):
        if len(self.undoBuffer) > 1:
            # if the undo buffer contains at least two items (the last state and the current state), pop the current state and retrieve the last one
            self.undoBuffer.pop()
            lastState = self.undoBuffer[-1]
            # change the QPixmaps according to the last state we get from the undo buffer
            for color in self.colors.keys():
                self.layers[color] = lastState[color].copy()
            self.update()

    def clear(self):
        self.layers[self.activeColor] = QPixmap(self.albedo.size())
        self.layers[self.activeColor].fill(Qt.transparent)
        self.undoBuffer.append({key: value.copy() for (key, value) in self.layers.items()})
        # update
        self.update()

    def clearAll(self):
        for color in self.colors.keys():
            self.layers[color] = QPixmap(self.albedo.size())
            self.layers[color].fill(Qt.transparent)
        self.undoBuffer.append({key: value.copy() for (key, value) in self.layers.items()})
        # update
        self.update()

    def reset(self):
        self.scale_ratio = 1.0
        self.offset = QPoint(0, 0)
        self.show_status()

    @staticmethod
    def to_pickleable(layers_raw, undoBuffer_raw):
        pickleable_layers = {}
        for (key, layer) in layers_raw.items():
            qbyte_array = QByteArray()
            stream = QDataStream(qbyte_array, QIODevice.WriteOnly)
            stream << layer
            pickleable_layers[key] = qbyte_array

        pickleable_undoBuffer = []
        for layers in undoBuffer_raw:
            pickleable = {}
            for (key, layer) in layers.items():
                qbyte_array = QByteArray()
                stream = QDataStream(qbyte_array, QIODevice.WriteOnly)
                stream << layer
                pickleable[key] = qbyte_array

            pickleable_undoBuffer.append(pickleable)

        dat = {'layers': pickleable_layers, 'undoBuffer': pickleable_undoBuffer}

        return dat

    def init_scribbles(self, preload=True, filename='scribbles.pkl'):
        success = preload and self.load_scribbles(filename=filename)
        if not success:
            # initialize scribble layers
            for color in self.colors:
                self.layers[color] = QPixmap(512, 512)  # FIXED SIZE
                self.layers[color].fill(Qt.transparent)
            # push the initial state into the undo buffer
            self.undoBuffer = []
            self.undoBuffer.append({key: value.copy() for (key, value) in self.layers.items()})
        self.update()

    def save_scribbles(self, filename='scribbles.pkl', show_warning=False):
        dat = self.to_pickleable(self.layers, self.undoBuffer)
        save_filename = pth.join(self.work_dir, filename)
        try:
            with open(save_filename, "wb") as f:
                pickle.dump(dat, f)
        except Exception as e:
            print(e)
            if show_warning:
                show_msg_box('Failed to save scribbles', 'Failed to save scribbles to file {}.'.format(save_filename))
            return False

        print('Scribbles saved to {}.'.format(save_filename))
        return True

    @staticmethod
    def from_pickleable(dat):
        pickleable_layers = dat['layers']
        pickleable_undoBuffer = dat['undoBuffer']

        layers = {}
        for (key, buffer) in pickleable_layers.items():
            qpixmap = QPixmap()
            stream = QDataStream(buffer, QIODevice.ReadOnly)
            stream >> qpixmap
            layers[key] = qpixmap

        undoBuffer = []
        for pickleable_layers in pickleable_undoBuffer:
            undo_layers = {}
            for (key, buffer) in pickleable_layers.items():
                qpixmap = QPixmap()
                stream = QDataStream(buffer, QIODevice.ReadOnly)
                stream >> qpixmap
                undo_layers[key] = qpixmap
            undoBuffer.append(undo_layers)

        return layers, undoBuffer

    def load_scribbles(self, filename='scribbles.pkl', show_warning=False):
        save_filename = pth.join(self.work_dir, filename)
        if not pth.exists(save_filename):
            if show_warning:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Failed to load scribbles")
                msg.setText("Cannot find file {}.".format(save_filename))
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            return False

        try:
            with open(save_filename, "rb") as f:
                dat = pickle.load(f)
        except:
            if show_warning:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setWindowTitle("Failed to load scribbles")
                msg.setText("Failed to load scribbles from file {}.".format(save_filename))
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
            return False

        try:
            layers, undoBuffer = self.from_pickleable(dat)
        except:
            return False

        self.layers, self.undoBuffer = layers, undoBuffer
        print('Scribbles loaded from {}'.format(save_filename))

        return True

    def set_brush_size(self, size):
        self.brushSize = size

    def set_brush_color(self, name):
        self.activeColor = name
        color = self.colors[name]
        self.brushColor = QColor(color[0], color[1], color[2])

    @staticmethod
    def convert_to_numpy_array(pixmap: QPixmap) -> np.ndarray:
        channel_count = 4
        image = pixmap.toImage()
        buffer = image.constBits()
        buffer.setsize(image.width() * image.height() * channel_count)
        bgra_array = np.frombuffer(buffer, dtype=np.uint8).reshape((pixmap.width(), pixmap.height(), channel_count))
        # remove the alpha channel
        rgb_array = np.delete(bgra_array.copy(), 3, axis=2)
        # convert bgr to rgb
        rgb_array[..., [0, 2]] = rgb_array[..., [2, 0]]
        return rgb_array

    @staticmethod
    def convert_to_pixmap(array: np.ndarray) -> QPixmap:
        width, height, channel = array.shape
        assert (channel == 3)
        bytesPerLine = channel * width
        image = QImage(array.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return QPixmap.fromImage(image)

    def load(self):
        raise NotImplementedError()

    def segment(self):
        raise NotImplementedError()

    def visualize(self):
        raise NotImplementedError()


class SVBRDFMatting(MattingBase):
    def __init__(self):
        super().__init__()
        # setting title
        self.setWindowTitle("SVBRDF Matting")
        self.setWindowFlags(Qt.MSWindowsFixedSizeDialogHint)

        # setting geometry to main window
        self.resize(512 * 3, 512)
        self.center()

        # store binary masks for visualization
        self.bmasks = None
        self.prev_bmask = None

        # MultiMaterial
        self.material = None

        """
        GUI layout definition
        """
        self.param_layout = QGridLayout()
        self.param_layout.addWidget(QLabel('KNN Matting'), 0, 0, 1, -1, Qt.AlignCenter)
        self.param_layout.addWidget(QLabel('Spectra Weright'), 1, 0)
        self.spectra_weight = QDoubleSpinBox(value=0.5)
        self.spectra_weight.setSingleStep(0.01)
        self.param_layout.addWidget(self.spectra_weight, 1, 1)
        self.param_layout.addWidget(QLabel('Instance-based Decomposition'), 2, 0, 1, -1, Qt.AlignCenter)
        self.param_layout.addWidget(QLabel('Length Threshold'), 3, 0)
        self.length_threshold = QDoubleSpinBox(value=10.0)
        self.length_threshold.setSingleStep(0.01)
        self.param_layout.addWidget(self.length_threshold, 3, 1)
        self.param_layout.addWidget(QLabel('Min Num of Samples'), 4, 0)
        self.min_num_of_samples = QSpinBox(value=10)
        self.param_layout.addWidget(self.min_num_of_samples, 4, 1)
        self.param_layout.addWidget(QLabel('Discard Outliers'), 5, 0)
        self.discard_outliers = QCheckBox(checked=False)
        self.param_layout.addWidget(self.discard_outliers, 5, 1)
        self.param_layout.addWidget(QLabel('Distance Threshold'), 6, 0)
        self.distance_threshold = QDoubleSpinBox(value=1.5)
        self.distance_threshold.setSingleStep(0.01)
        self.param_layout.addWidget(self.distance_threshold, 6, 1)
        self.param_layout.addWidget(QLabel('PatchMatch'), 7, 0, 1, -1, Qt.AlignCenter)
        self.param_layout.addWidget(QLabel('Erode Ratio'), 8, 0)
        self.erode_ratio = QDoubleSpinBox(value=0.03)
        self.erode_ratio.setSingleStep(0.01)
        self.erode_ratio.setRange(0.0, 1.0)
        self.param_layout.addWidget(self.erode_ratio, 8, 1)

        self.layout.addLayout(self.param_layout, 0, 0, -1, 1, Qt.AlignTop)
        self.updateLayout(QLabel("Albedo", self), 0, 1)
        self.updateLayout(QLabel("Normal", self), 0, 2)
        self.updateLayout(QLabel("Roughness", self), 0, 3)
        self.updateLayout(self.label_albedo, 1, 1)
        self.updateLayout(self.label_normal, 1, 2)
        self.updateLayout(self.label_roughness, 1, 3)
        self.num_fixed_widget = 7
        self.container.setLayout(self.layout)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.container)
        self.scrollArea.setMinimumWidth(self.container.sizeHint().width() + self.scrollArea.verticalScrollBar().sizeHint().width() + 10)
        self.scrollArea.setMinimumHeight(self.container.sizeHint().height() + self.scrollArea.horizontalScrollBar().sizeHint().height())
        self.setCentralWidget(self.scrollArea)
        # the starting row for the next widget to be added
        self.row_index = 2

        """
        menu bar layout definition
        """
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("File")
        editMenu = mainMenu.addMenu("Edit")
        self.opMenu = mainMenu.addMenu("Operation")
        sizeMenu = mainMenu.addMenu("Brush Size")
        colorMenu = mainMenu.addMenu("Brush Type")
        optionMenu = mainMenu.addMenu("Options")

        ################################################################################################################
        openAction = QAction("Open File Folder", self)
        openAction.setShortcut("Ctrl+Alt+O")
        fileMenu.addAction(openAction)
        openAction.triggered.connect(self.open_file_folder)

        closeAction = QAction('Exit', self)
        closeAction.setShortcut('Ctrl+Q')
        fileMenu.addAction(closeAction)
        closeAction.triggered.connect(self.close)

        ################################################################################################################
        self.create_op_menu()
        ################################################################################################################
        segmentAction = QAction("Segment", self)
        segmentAction.setShortcut("Ctrl+S")
        editMenu.addAction(segmentAction)
        segmentAction.triggered.connect(self.segment)

        extractInstanceAction = QAction("Extract Instance", self)
        extractInstanceAction.setShortcut("Ctrl+E")
        editMenu.addAction(extractInstanceAction)
        extractInstanceAction.triggered.connect(self.extract_instance)

        inpaintAction = QAction("Inpaint", self)
        inpaintAction.setShortcut("Ctrl+I")
        editMenu.addAction(inpaintAction)
        inpaintAction.triggered.connect(self.inpaint)

        # additional options
        self.regularize_button = QAction("Mask regularization", self, checkable=True, checked=False)
        editMenu.addAction(self.regularize_button)

        editMenu.addSeparator()

        undoAction = QAction("Undo", self)
        undoAction.setShortcut("Ctrl+Z")
        editMenu.addAction(undoAction)
        undoAction.triggered.connect(self.undo)

        clearAction = QAction("Clear", self)
        clearAction.setShortcut("Ctrl+C")
        editMenu.addAction(clearAction)
        clearAction.triggered.connect(self.clear)

        clearAllAction = QAction("Clear All", self)
        clearAllAction.setShortcut("Ctrl+A")
        editMenu.addAction(clearAllAction)
        clearAllAction.triggered.connect(self.clearAll)

        resetAction = QAction("Reset View", self)
        resetAction.setShortcut("Ctrl+R")
        editMenu.addAction(resetAction)
        resetAction.triggered.connect(self.reset)

        # creating options for brush sizes
        pix_3 = QAction("3 px", self, checkable=True, checked=False)
        sizeMenu.addAction(pix_3)
        pix_3.triggered.connect(lambda: self.set_brush_size(3))

        pix_6 = QAction("6 px", self, checkable=True, checked=False)
        sizeMenu.addAction(pix_6)
        pix_6.triggered.connect(lambda: self.set_brush_size(6))

        pix_9 = QAction("9 px", self, checkable=True, checked=False)
        sizeMenu.addAction(pix_9)
        pix_9.triggered.connect(lambda: self.set_brush_size(9))

        pix_12 = QAction("12 px", self, checkable=True, checked=True)
        sizeMenu.addAction(pix_12)
        pix_12.triggered.connect(lambda: self.set_brush_size(12))

        pix_15 = QAction("15 px", self, checkable=True, checked=False)
        sizeMenu.addAction(pix_15)
        pix_15.triggered.connect(lambda: self.set_brush_size(15))

        pix_18 = QAction("18 px", self, checkable=True, checked=False)
        sizeMenu.addAction(pix_18)
        pix_18.triggered.connect(lambda: self.set_brush_size(18))

        pix_21 = QAction("21 px", self, checkable=True, checked=False)
        sizeMenu.addAction(pix_21)
        pix_21.triggered.connect(lambda: self.set_brush_size(21))

        sizeGroup = QActionGroup(self)
        sizeGroup.addAction(pix_3)
        sizeGroup.addAction(pix_6)
        sizeGroup.addAction(pix_9)
        sizeGroup.addAction(pix_12)
        sizeGroup.addAction(pix_15)
        sizeGroup.addAction(pix_18)
        sizeGroup.addAction(pix_21)

        # creating options for brush colors
        foreground = QAction("Layer 0", self, checkable=True, checked=True)
        colorMenu.addAction(foreground)
        foreground.triggered.connect(lambda: self.set_brush_color('blue'))

        background = QAction("Layer 1", self, checkable=True, checked=False)
        colorMenu.addAction(background)
        background.triggered.connect(lambda: self.set_brush_color('green'))

        colorGroup = QActionGroup(self)
        colorGroup.addAction(foreground)
        colorGroup.addAction(background)

        # additional options
        self.trim_button = QAction("Trim Mask", self, checkable=True, checked=False)
        optionMenu.addAction(self.trim_button)

        del_button = QAction("Delete Cache", self)
        optionMenu.addAction(del_button)
        del_button.triggered.connect(self.delete_all_data)

        self.autosave_button = QAction("Auto Save/Load Scribbles", self, checkable=True, checked=True)
        optionMenu.addAction(self.autosave_button)

        load_scribbles = QAction("Load Previous Scribbles", self)
        optionMenu.addAction(load_scribbles)
        load_scribbles.triggered.connect(functools.partial(self.load_scribbles, show_warning=True))

        save_scribbles = QAction("Save Scribbles", self)
        optionMenu.addAction(save_scribbles)
        save_scribbles.triggered.connect(functools.partial(self.save_scribbles, show_warning=True))

        refine_masks = QAction("Refine Masks", self)
        optionMenu.addAction(refine_masks)
        refine_masks.triggered.connect(self.go_to_refinement_stage)

    def create_op_menu(self):
        self.opMenu.clear()
        if self.bmasks is None and self.root_dir is None:
            dummy_action = QAction("No available operation", self)
            self.opMenu.addAction(dummy_action)

        if self.bmasks is not None:
            for i in range(self.bmasks.shape[0]):
                next_action = QAction('Segment Layer {}'.format(i), self)
                next_action.setShortcut('Ctrl+N+{}'.format(i))
                self.opMenu.addAction(next_action)
                next_action.triggered.connect(partial(self.next_layer, index=i))
        if self.root_dir is not None and self.root_dir != self.work_dir:
            prev_action = QAction("Return to previous stage", self)
            prev_action.setShortcut("Ctrl+P")
            self.opMenu.addAction(prev_action)
            prev_action.triggered.connect(self.prev_layer)

        if self.work_dir is not None and pth.exists(pth.join(self.work_dir, 'inpainted')):
            self.opMenu.addSeparator()
            load_inpainted_action = QAction('Load inpainted', self)
            load_inpainted_action.triggered.connect(partial(self.load_inpainted_maps, is_inpainted=True))
            load_inpainted_action.setCheckable(True)
            load_original_action = QAction('Load original', self)
            load_original_action.triggered.connect(partial(self.load_inpainted_maps, is_inpainted=False))
            load_original_action.setCheckable(True)
            self.opMenu.addAction(load_inpainted_action)
            self.opMenu.addAction(load_original_action)
            load_action = QActionGroup(self)
            load_action.addAction(load_inpainted_action)
            load_action.addAction(load_original_action)
            if self.material.is_inpainted():
                load_inpainted_action.setChecked(True)
            else:
                load_original_action.setChecked(True)

    # preprocess a single image
    def preprocess(self, image_filename, save_filename, to_gray=False, is_normal=False):
        if pth.exists(save_filename) is True:
            return

        # resize to 512
        preprocess = functools.partial(center_crop_resize, shape=(512, 512), resample=Image.NEAREST)
        image = preprocess(Image.open(image_filename))
        if not is_normal:
            if to_gray and image.mode == 'RGB':
                image = image.convert('L')
            elif not to_gray and image.mode == 'RGBA':
                image = image.convert('RGB')
            image.save(save_filename)
        else:
            image = image.convert('RGB')
            image = np.asarray(image, dtype=np.float32) / 255.0
            height_ = normal2height(image)
            if self.trim_button.isChecked():
                height = smooth_height_map(height_)
            else:
                height = height_
            save_image(normalize(height), save_filename)
            np.save(pth.splitext(save_filename)[0] + '.npy', height)

    # preprocess an SVBRDF
    def preprocess_svbrdf(self, image_filename, albedo_filename_preprocessed,
                          normal_filename_preprocessed, roughness_filename_preprocessed):
        if pth.exists(albedo_filename_preprocessed) and pth.exists(normal_filename_preprocessed) and \
                pth.exists(roughness_filename_preprocessed):
            return True

        image = Image.open(image_filename)
        w, h = image.size
        if w != h * 4:
            show_msg_box('Error', 'Incorrect format')
            return False

        normal = image.crop((0, 0, h, h))
        albedo = image.crop((h, 0, h * 2, h))
        roughness = image.crop((h * 2, 0, h * 3, h))

        # resize to 512
        preprocess = functools.partial(center_crop_resize, shape=(512, 512))
        normal = preprocess(normal)
        albedo = preprocess(albedo)
        roughness = preprocess(roughness)

        albedo.save(albedo_filename_preprocessed)
        roughness = roughness.convert('L')
        roughness.save(roughness_filename_preprocessed)

        normal = np.asarray(normal, dtype=np.float32) / 255.0
        height_ = normal2height(normal)
        if self.trim_button.isChecked():
            height = smooth_height_map(height_)
        else:
            height = height_
        save_image(normalize(height), normal_filename_preprocessed)
        np.save(pth.splitext(normal_filename_preprocessed)[0] + '.npy', height)

        return True

    @staticmethod
    def check_exist(filename):
        if pth.exists(filename) is True:
            return True
        else:
            return False

    def open_file_folder(self):
        # save previous scribbles
        if len(self.layers) > 0 and len(self.undoBuffer) > 0 and self.autosave_button.isChecked():
            self.save_scribbles()

        init_path = './samples'
        work_dir = QFileDialog.getExistingDirectory(self, 'Open SVBRDF Maps', init_path)
        if len(work_dir) == 0:
            return

        basename = pth.basename(work_dir)
        albedo_filename = pth.join(work_dir, basename + '_albedo.png')
        normal_filename = pth.join(work_dir, basename + '_normal.png')
        height_filename = pth.join(work_dir, basename + '_height.png')
        roughness_filename = pth.join(work_dir, basename + '_roughness.png')

        if self.check_exist(albedo_filename) is False:
            show_msg_box("Cannot find input image", "Missing input image {}.".format(albedo_filename))
            return
        if self.check_exist(normal_filename) is False and self.check_exist(height_filename) is False:
            show_msg_box("Cannot find input image", "Missing input image {} or {}.".format(normal_filename, height_filename))
            return
        if self.check_exist(roughness_filename) is False:
            show_msg_box("Cannot find input image", "Missing input image {}.".format(roughness_filename))
            return

        # now load new svbrdf maps
        self.work_dir = pth.join(work_dir, "results")
        self.root_dir = self.work_dir
        os.makedirs(self.work_dir, exist_ok=True)

        self.state = 'Loading SVBRDF maps'
        self.show_status()

        albedo_filename_preprocessed = pth.join(self.work_dir, 'albedo.png')
        normal_filename_preprocessed = pth.join(self.work_dir, 'normal.png')
        roughness_filename_preprocessed = pth.join(self.work_dir, 'roughness.png')
        prev_bmask_filename = pth.join(self.work_dir, 'prev_bmask.png')

        self.preprocess(albedo_filename, albedo_filename_preprocessed)
        if pth.exists(height_filename):  # if height map exists
            self.preprocess(height_filename, normal_filename_preprocessed, to_gray=True)
        else:
            self.preprocess(normal_filename, normal_filename_preprocessed, is_normal=True)
        self.preprocess(roughness_filename, roughness_filename_preprocessed, to_gray=True)
        prev_bmask = np.ones((512, 512), dtype=bool)
        save_image(prev_bmask, prev_bmask_filename)

        self.load()
        self.load_material()
        self.init_precomputed_data()
        self.create_op_menu()

    def load(self, target_dir=None):
        if target_dir is None:
            target_dir = self.work_dir
        self.clearLayout()
        self.reset()

        albedo_filename = pth.join(target_dir, 'albedo.png')
        normal_filename = pth.join(target_dir, 'normal.png')
        roughness_filename = pth.join(target_dir, 'roughness.png')
        prev_bmask_filename = pth.join(target_dir, 'prev_bmask.png')

        self.albedo = QPixmap(albedo_filename)
        self.normal = QPixmap(normal_filename)
        self.roughness = QPixmap(roughness_filename)
        self.prev_bmask = load_image(prev_bmask_filename).astype(bool)

        self.label_albedo.setPixmap(self.albedo)
        self.label_normal.setPixmap(self.normal)
        self.label_roughness.setPixmap(self.roughness)

        self.adjustSize()
        self.state = 'SVBRDF loaded'
        self.show_status()

    def init_precomputed_data(self):
        valid = self.material.check_valid()
        if valid:
            self.init_scribbles(preload=True)
            try:
                bmasks = load_masks(self.work_dir, -1, prefix="bmask")
            except RuntimeError:
                self.bmasks = None
            else:
                self.bmasks = bmasks
                self.visualize()
        else:
            # if mask map is invalid, delete all cache data
            self.delete_all_data()
            self.delete_inpaint_data()
            self.remove_file_with_prefix('joint_spectra')  # delete spectrum data

            if self.work_dir != self.root_dir:
                self.remove_file_with_prefix('scribbles')
                self.init_scribbles(preload=False)
            else:
                self.init_scribbles(preload=True)
            self.bmasks = None

    def load_material(self):
        material_filename = pth.join(self.root_dir, 'material.pkl')
        if pth.exists(material_filename):
            try:
                with open(material_filename, "rb") as f:
                    self.material = pickle.load(f)
                self.material.reset()  # reset current node to root node
            except Exception as e:
                print(e)
                show_msg_box('Error', 'Failed to load material data. Create a new one.')
                svbrdf = self.extract_svbrdf_numpy(reformat=True)
                self.material = MultiMaterial(svbrdf)
        else:
            svbrdf = self.extract_svbrdf_numpy(reformat=True)
            self.material = MultiMaterial(svbrdf)

    def save_material(self, show_warning=False):
        material_filename = pth.join(self.root_dir, 'material.pkl')
        try:
            with open(material_filename, "wb") as f:
                pickle.dump(self.material, f)
        except Exception as e:
            print(e)
            if show_warning:
                show_msg_box('Failed to save material data', 'Failed to save material data to file {}.'.format(material_filename))
            return False

        print('Material saved to {}.'.format(material_filename))

    def extract_svbrdf_numpy(self, reformat=False):
        albedo = self.convert_to_numpy_array(self.albedo)
        normal = self.convert_to_numpy_array(self.normal)
        roughness = self.convert_to_numpy_array(self.roughness)
        if not reformat:
            return albedo, normal, roughness
        else:
            return albedo / 255.0, normal[..., 0] / 255.0, roughness[..., 0] / 255.0

    def build_next_layer(self, index):
        next_dir = pth.join(self.work_dir, 'layer{}'.format(index))
        os.makedirs(next_dir, exist_ok=True)

        albedo_filename = pth.join(next_dir, 'albedo.png')
        normal_filename = pth.join(next_dir, 'normal.png')
        roughness_filename = pth.join(next_dir, 'roughness.png')
        prev_bmask_filename = pth.join(next_dir, 'prev_bmask.png')
        if pth.exists(albedo_filename) and pth.exists(normal_filename) and pth.exists(roughness_filename) \
                and pth.exists(prev_bmask_filename) and self.material.check_valid():
            self.albedo = QPixmap(albedo_filename)
            self.normal = QPixmap(normal_filename)
            self.roughness = QPixmap(roughness_filename)
            self.prev_bmask = load_image(prev_bmask_filename)
            self.prev_bmask = self.prev_bmask.astype(bool)
        else:
            # compute and store SVBRDF maps
            mask = self.bmasks[index]
            self.prev_bmask = mask
            albedo, normal, roughness = self.extract_svbrdf_numpy()
            # compute the masked SVBRDF maps
            mask_albedo = mask[..., np.newaxis] * albedo
            mask_normal = mask[..., np.newaxis] * normal
            mask_roughness = mask[..., np.newaxis] * roughness
            # convert numpy arrays to QPixmaps
            self.albedo = self.convert_to_pixmap(mask_albedo)
            self.normal = self.convert_to_pixmap(mask_normal)
            self.roughness = self.convert_to_pixmap(mask_roughness)
            # save masked SVBRDF maps
            save_image(mask_albedo / 255.0, albedo_filename)
            save_image(mask_normal[..., 0] / 255.0, normal_filename)
            save_image(mask_roughness[..., 0] / 255.0, roughness_filename)
            save_image(self.prev_bmask, prev_bmask_filename)

        return next_dir

    def next_layer(self, index):
        if self.work_dir is None:
            return

        # check if current image has been segmented
        if self.bmasks is None:
            show_msg_box("Unsegmented", "This material has not been segmented yet. Please segment it first.")
            return

        if self.autosave_button.isChecked():
            self.save_scribbles()

        self.material.next_layer(index)
        self.work_dir = self.build_next_layer(index)
        self.load()
        self.init_precomputed_data()
        self.create_op_menu()

    # working on this one
    def prev_layer(self):
        if self.work_dir is None:
            return

        if self.work_dir == self.root_dir:
            show_msg_box("Top layer", "This is the top layer.")
            return

        if self.autosave_button.isChecked():
            self.save_scribbles()

        self.work_dir = pth.dirname(self.work_dir)
        self.material.prev_layer()
        self.load()
        self.init_precomputed_data()
        self.create_op_menu()

    def remove_file_with_prefix(self, prefix, ext=''):
        file_list = glob.glob(pth.join(self.work_dir, prefix + '*' + ext))
        for f in file_list:
            os.remove(f)

    def delete_all_data(self):
        self.delete_matting_data()
        self.delete_refined_data()

    def delete_matting_data(self):
        self.remove_file_with_prefix('bmask', '.png')
        self.remove_file_with_prefix('label_map', '.png')

    def delete_refined_data(self):
        self.remove_file_with_prefix('refined_spectra')
        self.remove_file_with_prefix('scribbles_ref')
        self.remove_file_with_prefix('fmask', '.png')

    def delete_inpaint_data(self):
        original_results_data_path = pth.join(self.work_dir, 'original')
        inpainted_data_path = pth.join(self.work_dir, 'inpainted')

        if pth.exists(original_results_data_path):
            shutil.rmtree(original_results_data_path)
        if pth.exists(inpainted_data_path):
            shutil.rmtree(inpainted_data_path)

    def segment(self):
        if self.work_dir is None:
            return

        if self.autosave_button.isChecked():
            self.save_scribbles()

        self.delete_all_data()

        # do segmentation
        self.state = 'Segmenting'
        self.show_status()

        scribbles = []
        labels = []

        # key is color
        for key in self.layers.keys():
            # convert layers to numpy arrays
            array = self.convert_to_numpy_array(self.layers[key])
            mask = np.zeros(array.shape[:2], dtype=bool)
            strokes = np.any(array != [0, 0, 0], axis=-1)
            mask[strokes] = 1
            if np.any(mask):
                scribbles.append(mask)
                labels.append(key)

        if len(scribbles) == 0:
            return

        scribbles = np.stack(scribbles, axis=0)
        albedo_filename = pth.join(self.work_dir, 'albedo.png')
        normal_filename = pth.join(self.work_dir, 'normal.png')
        roughness_filename = pth.join(self.work_dir, 'roughness.png')
        albedo_map = expand_dim_to_3(load_image(albedo_filename))
        height_map = expand_dim_to_3(load_image(normal_filename))
        roughness_map = expand_dim_to_3(load_image(roughness_filename))
        svbrdf_maps = np.concatenate((albedo_map, height_map, roughness_map), axis=2)

        try:
            if np.array_equal(self.prev_bmask, np.ones((512, 512), dtype=bool)):
                alpha, masks = svbrdf_matting(svbrdf_maps, scribbles,
                                              spectra_weight=self.spectra_weight.value(), data_path=self.work_dir)
            else:
                alpha, masks = svbrdf_matting_with_mask(svbrdf_maps, self.prev_bmask, scribbles,
                                                        spectra_weight=self.spectra_weight.value(), data_path=self.work_dir)

        except Exception as e:
            print(e)
            show_msg_box("Error", "Failed to perform svbrdf matting")
            return

        alpha = np.transpose(alpha, axes=(2, 0, 1))
        masks = np.transpose(masks, axes=(2, 0, 1))
        masks = masks.astype(bool)
        self.bmasks = mask_binarization(alpha)  # (n, w, h)
        for i in range(self.bmasks.shape[0]):
            self.bmasks[i] *= self.prev_bmask
        label_map = self.compute_label_map(labels, self.bmasks)

        save_masks_as_image(self.bmasks, self.work_dir, prefix='bmask')
        save_image(label_map, pth.join(self.work_dir, 'label_map.png'))

        self.material.update_mask(self.bmasks)
        self.save_material(self.material)

        self.visualize()

        self.create_op_menu()

        # end segmentation
        self.state = 'Finished'
        self.show_status()

    def compute_label_map(self, labels, mask_map):
        label_map = np.zeros((*mask_map.shape[1:], 3), dtype=np.uint8)
        for i in range(len(labels)):
            mask = mask_map[i]
            mask = expand_dim_to_3(mask)
            area = np.any(mask == [True], axis=-1)
            label_map[area] = self.colors[labels[i]]
        return label_map

    def visualize(self):
        if self.bmasks is None:
            return

        # clear the previous result first
        self.clearLayout()

        albedo = self.convert_to_numpy_array(self.albedo)
        normal = self.convert_to_numpy_array(self.normal)
        roughness = self.convert_to_numpy_array(self.roughness)

        n_masks = self.bmasks.shape[0]
        for i in range(n_masks):
            self.updateLayout(QLabel("Foreground" if i == 0 else "Background", self), self.row_index, 1)
            mask = self.bmasks[i]
            # compute the masked images
            mask_albedo = mask[..., np.newaxis] * albedo
            mask_normal = mask[..., np.newaxis] * normal
            mask_roughness = mask[..., np.newaxis] * roughness
            # convert numpy arrays to QPixmaps
            mask_albedo = self.convert_to_pixmap(mask_albedo)
            mask_normal = self.convert_to_pixmap(mask_normal)
            mask_roughness = self.convert_to_pixmap(mask_roughness)
            # display these pixmaps
            label_albedo = QLabel(self)
            label_albedo.setFrameShape(QFrame.Box)
            label_albedo.setPixmap(mask_albedo)
            label_normal = QLabel(self)
            label_normal.setFrameShape(QFrame.Box)
            label_normal.setPixmap(mask_normal)
            label_roughness = QLabel(self)
            label_roughness.setFrameShape(QFrame.Box)
            label_roughness.setPixmap(mask_roughness)
            self.updateLayout(label_albedo, self.row_index + 1, 1)
            self.updateLayout(label_normal, self.row_index + 1, 2)
            self.updateLayout(label_roughness, self.row_index + 1, 3)
            self.row_index += 2

        self.adjustSize()

    def regularize_mask_map(self, iters=1, morph_iters=1):
        mask0 = self.prev_bmask.copy()
        mask0 = mask0.astype(np.uint8)
        for it in range(iters):
            kernel = np.ones((it + 2, it + 2), np.uint8)
            mask1 = cv2.morphologyEx(mask0, cv2.MORPH_OPEN, kernel, iterations=morph_iters)
            mask0 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel, iterations=morph_iters)

        return mask0

    # compute features for each instance
    @staticmethod
    def compute_local_features(svbrdfs, resized_shape, reduced_dim=32):
        w, h = resized_shape
        local_spectra = []
        local_histograms = []

        for svbrdf in svbrdfs:
            albedo, height, roughness = svbrdf[..., 0:3], svbrdf[..., 3], svbrdf[..., 4]

            albedo = cv2.resize(albedo, resized_shape)
            height = cv2.resize(height, resized_shape)
            roughness = cv2.resize(roughness, resized_shape)

            # computer spectrum features
            albedo_gray = cv2.cvtColor(albedo.astype(np.float32), cv2.COLOR_RGB2GRAY).astype(np.float64)
            albedo_spectrum = np.abs(fft2(albedo_gray))
            height_spectrum = np.abs(fft2(height))
            roughness_spectrum = np.abs(fft2(roughness))
            spectrum = np.stack((albedo_spectrum, height_spectrum, roughness_spectrum), axis=2)
            local_spectra.append(spectrum)

            # compute color histogram feature
            histogram = []
            for i in range(albedo.shape[2]):
                hist, _ = np.histogram(albedo[..., i], bins=10, range=(0, 1), density=True)
                histogram.append(hist)
            hist, _ = np.histogram(height, bins=10, range=(0, 1), density=True)
            histogram.append(hist)
            hist, _ = np.histogram(roughness, bins=10, range=(0, 1), density=True)
            histogram.append(hist)
            histogram = np.concatenate(histogram, axis=0)
            local_histograms.append(histogram)

        # reduce dim
        local_spectra = np.stack(local_spectra, axis=0).reshape(-1, w * h * 3)  # (n_samples, w*h*3)
        local_spectra = reduce_dim(local_spectra, target_dim=reduced_dim, normalized=True)

        local_histograms = np.stack(local_histograms, axis=0)  # (n_samples, 50)
        local_histograms = np.nan_to_num(local_histograms)
        local_histograms = reduce_dim(local_histograms, target_dim=reduced_dim, normalized=True)

        features = np.concatenate((local_spectra, local_histograms), axis=1)  # (n_samples, n_features)
        return features

    def extract_instance(self):
        if self.prev_bmask is None or np.all(self.prev_bmask):
            return

        self.state = 'Extracting Instances'
        self.show_status()

        self.remove_file_with_prefix('bmask', '.png')

        if self.regularize_button.isChecked():
            mask = self.regularize_mask_map(iters=1, morph_iters=1)
        else:
            mask = self.prev_bmask.astype(np.uint8)

        albedo, normal, roughness = self.extract_svbrdf_numpy(reformat=True)
        svbrdf = np.concatenate((albedo, normal[..., np.newaxis], roughness[..., np.newaxis]), axis=2)  # (w, h, 5)

        labels, num = labeling(mask, return_num=True)
        print(f'Found {num} labeled regions')

        sub_svbrdfs = []
        lengths = []

        for i in range(1, num + 1):
            idx = np.where(labels == i)
            label = np.zeros_like(labels)
            label[idx[0], idx[1]] = 1

            sx, sy, ex, ey = find_bounding_box(label)
            sub_svbrdf = svbrdf[sx:ex, sy:ey]
            l = min(ex - sx, ey - sy)

            sub_svbrdfs.append(sub_svbrdf)

            lengths.append(l)

        lengths = np.asarray(lengths)

        # thresholding on size of instances
        # TODO: expose this parameter: length_threshold
        length_threshold = self.length_threshold.value()  # 10
        valid_indices_s = np.where(lengths > length_threshold)[0]
        lengths = lengths[valid_indices_s]

        print(f'{len(valid_indices_s)} valid regions remain after filtering small regions (length less than {length_threshold})')
        if len(valid_indices_s) == 0:
            show_msg_box('Error', 'Failed to detect enough regions')
            self.state = 'Finished'
            self.show_status()
            return

        # TODO: Expose a parameter that let user to switch this step on or off
        # minimum required samples if lower than this threshold, outlier detection will be disabled
        min_num_of_samples = self.min_num_of_samples.value()  # 10
        discard_outliers = self.discard_outliers.isChecked  # False
        if len(valid_indices_s) > min_num_of_samples and discard_outliers:
            # outlier detection based on length
            outlier_model = EllipticEnvelope(contamination=0.25, random_state=0)  # 0.25
            valid = outlier_model.fit_predict(lengths.reshape(-1, 1))
        else:
            valid = np.ones((len(valid_indices_s),), dtype=np.uint8)

        valid_indices_o = np.where(valid == 1)[0]
        valid_lengths = lengths[valid_indices_o]
        print(f'{len(valid_indices_o)} valid regions remain after outlier detection')
        if len(valid_indices_o) == 0:
            show_msg_box('Error', 'Failed to detect enough regions')
            self.state = 'Finished'
            self.show_status()
            return

        # compute indices
        valid_indices = valid_indices_s[valid_indices_o]
        valid_sub_svbrdfs = [sub_svbrdfs[i] for i in valid_indices]

        max_len = np.max(valid_lengths)
        features = self.compute_local_features(valid_sub_svbrdfs, (max_len, max_len))

        # clustering
        # TODO: expose distance_threshold
        clustering_model = clustering.AgglomerativeClustering(n_clusters=None, distance_threshold=self.distance_threshold.value())  # 0.75|1|1.5

        cluster_labels = clustering_model.fit_predict(features)
        n_clusters = len(np.unique(cluster_labels))
        print(f'{n_clusters} clusters are generated')

        mask_maps = np.zeros((n_clusters, *svbrdf.shape[:2]), dtype=bool)
        for i, cluster_label in zip(valid_indices, cluster_labels):
            idx = np.where(labels == i + 1)
            mask_maps[cluster_label, idx[0], idx[1]] = True

        self.bmasks = mask_maps
        save_masks_as_image(self.bmasks, self.work_dir, prefix='bmask')

        self.material.update_mask(self.bmasks, is_instance=True)
        self.save_material(self.material)

        self.visualize()

        self.create_op_menu()

        self.state = 'Finished'
        self.show_status()

    def load_inpainted_maps(self, is_inpainted):
        target_dir = pth.join(self.work_dir, 'inpainted' if is_inpainted else 'original')
        self.copy_svbrdf_maps(target_dir, self.work_dir)
        self.load()

        if self.material.is_inpainted() != is_inpainted:
            self.material.set_valid(False)
            self.delete_all_data()
            self.remove_file_with_prefix('joint_spectra')

        self.material.use_inpaint(is_inpainted)

    @staticmethod
    def copy_svbrdf_maps(source_dir, target_dir):
        shutil.copyfile(pth.join(source_dir, 'albedo.png'), pth.join(target_dir, 'albedo.png'))
        shutil.copyfile(pth.join(source_dir, 'normal.png'), pth.join(target_dir, 'normal.png'))
        shutil.copyfile(pth.join(source_dir, 'roughness.png'), pth.join(target_dir, 'roughness.png'))
        shutil.copyfile(pth.join(source_dir, 'prev_bmask.png'), pth.join(target_dir, 'prev_bmask.png'))

    def inpaint(self):
        # no need to inpaint
        if self.prev_bmask is None or np.all(self.prev_bmask):
            return

        self.state = 'Inpainting'
        self.show_status()

        albedo, normal, roughness = self.extract_svbrdf_numpy(reformat=True)
        svbrdf = np.concatenate((albedo, normal[..., np.newaxis], roughness[..., np.newaxis]), axis=2)
        try:
            inpainted_svbrdf = inpaint(svbrdf, expand_dim_to_3(self.prev_bmask), erode_ratio=self.erode_ratio.value(), searchvoteiters=100, patchmatchiters=100, extrapass3x3=1)
        except Exception as e:
            print(e)
            show_msg_box("Error", "Failed to perform svbrdf inpainting")
            return

        # save original results and inpainted results
        original_results_data_path = pth.join(self.work_dir, 'original')
        inpainted_data_path = pth.join(self.work_dir, 'inpainted')
        os.makedirs(original_results_data_path, exist_ok=True)
        os.makedirs(inpainted_data_path, exist_ok=True)

        save_image(inpainted_svbrdf[..., 0:3], pth.join(inpainted_data_path, 'albedo.png'))
        save_image(inpainted_svbrdf[..., 3], pth.join(inpainted_data_path, 'normal.png'))
        save_image(inpainted_svbrdf[..., 4], pth.join(inpainted_data_path, 'roughness.png'))
        self.prev_bmask = np.ones((512, 512), dtype=bool)
        save_image(self.prev_bmask, pth.join(inpainted_data_path, 'prev_bmask.png'))

        self.copy_svbrdf_maps(self.work_dir, original_results_data_path)

        print('Patch Match finished')
        self.material.use_inpaint(False)
        self.save_material(self.material)

        self.create_op_menu()

        self.state = 'Finished'
        self.show_status()

    def go_to_refinement_stage(self):
        if len(self.work_dir) > 0:
            try:
                refinement = MattingRefinement(self.work_dir)
                refinement.show()
            except Exception as e:
                print(e)


class MattingRefinement(MattingBase):
    def __init__(self, work_dir):
        super().__init__()

        self.work_dir = work_dir

        self.setWindowTitle("Matting Refinement")
        self.setWindowFlags(Qt.MSWindowsFixedSizeDialogHint)

        self.resize(512 * 2, 512)
        self.center()

        # main image label
        self.image_label = QLabel()

        # create buttons
        self.btn_layout = QVBoxLayout()
        self.btn_intended = QRadioButton('Intended', self)
        self.btn_intended.setChecked(True)
        self.btn_intended.toggled.connect(lambda: self.set_brush_color('blue'))
        self.btn_layout.addWidget(self.btn_intended)

        self.btn_unwanted = QRadioButton('Unwanted', self)
        self.btn_unwanted.setChecked(False)
        self.btn_unwanted.toggled.connect(lambda: self.set_brush_color('green'))
        self.btn_layout.addWidget(self.btn_unwanted)

        self.spectra_weight = QDoubleSpinBox()
        self.spectra_weight.setRange(0, 10)
        self.spectra_weight.setSingleStep(0.01)
        self.spectra_weight.setValue(0.5)
        self.btn_layout.addWidget(self.spectra_weight)

        next_image_btn = QPushButton('Next Image', self)
        next_image_btn.clicked.connect(self.proc_next_image)
        self.btn_layout.addWidget(next_image_btn)
        prev_image_btn = QPushButton('Previous Image', self)
        prev_image_btn.clicked.connect(self.proc_previous_image)
        self.btn_layout.addWidget(prev_image_btn)

        self.btn_container = QWidget()
        self.btn_container.setLayout(self.btn_layout)

        # refined image label
        self.intended_label = QLabel()
        self.intended_label.setFixedSize(512, 512)
        self.intended_label.setFrameShape(QFrame.Box)
        self.placeholder = QPixmap()
        self.unwanted_label = QLabel()
        self.unwanted_label.setFixedSize(512, 512)
        self.unwanted_label.setFrameShape(QFrame.Box)

        self.layout.addWidget(self.label_albedo, 0, 0)
        self.layout.addWidget(self.btn_container, 0, 1)
        self.layout.addWidget(self.intended_label, 1, 0)
        self.layout.addWidget(self.unwanted_label, 1, 1)
        self.layout.addWidget(QLabel("Intended", self), 2, 0, Qt.AlignCenter)
        self.layout.addWidget(QLabel("Unwanted", self), 2, 1, Qt.AlignCenter)

        self.container.setLayout(self.layout)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.container)
        self.scrollArea.setMinimumWidth(self.container.sizeHint().width() + self.scrollArea.verticalScrollBar().sizeHint().width() + 10)
        self.scrollArea.setMinimumHeight(self.container.sizeHint().height() + self.scrollArea.horizontalScrollBar().sizeHint().height())
        self.setCentralWidget(self.scrollArea)

        # create menu
        mainMenu = self.menuBar()
        optionMenu = mainMenu.addMenu("Options")
        sizeMenu = mainMenu.addMenu("Brush Size")

        # creating options for brush sizes
        pix_3 = QAction("3 px", self, checkable=True, checked=False)
        sizeMenu.addAction(pix_3)
        pix_3.triggered.connect(lambda: self.set_brush_size(3))

        pix_6 = QAction("6 px", self, checkable=True, checked=False)
        sizeMenu.addAction(pix_6)
        pix_6.triggered.connect(lambda: self.set_brush_size(6))

        pix_9 = QAction("9 px", self, checkable=True, checked=False)
        sizeMenu.addAction(pix_9)
        pix_9.triggered.connect(lambda: self.set_brush_size(9))

        pix_12 = QAction("12 px", self, checkable=True, checked=True)
        sizeMenu.addAction(pix_12)
        pix_12.triggered.connect(lambda: self.set_brush_size(12))

        pix_15 = QAction("15 px", self, checkable=True, checked=False)
        sizeMenu.addAction(pix_15)
        pix_15.triggered.connect(lambda: self.set_brush_size(15))

        pix_18 = QAction("18 px", self, checkable=True, checked=False)
        sizeMenu.addAction(pix_18)
        pix_18.triggered.connect(lambda: self.set_brush_size(18))

        pix_21 = QAction("21 px", self, checkable=True, checked=False)
        sizeMenu.addAction(pix_21)
        pix_21.triggered.connect(lambda: self.set_brush_size(21))

        sizeGroup = QActionGroup(self)
        sizeGroup.addAction(pix_3)
        sizeGroup.addAction(pix_6)
        sizeGroup.addAction(pix_9)
        sizeGroup.addAction(pix_12)
        sizeGroup.addAction(pix_15)
        sizeGroup.addAction(pix_18)
        sizeGroup.addAction(pix_21)

        # additonal options
        segmentAction = QAction("Segment", self)
        optionMenu.addAction(segmentAction)
        segmentAction.triggered.connect(self.segment)

        optionMenu.addSeparator()

        undoAction = QAction("Undo", self)
        undoAction.setShortcut("Ctrl+Z")
        optionMenu.addAction(undoAction)
        undoAction.triggered.connect(self.undo)

        clearAction = QAction("Clear", self)
        clearAction.setShortcut("Ctrl+C")
        optionMenu.addAction(clearAction)
        clearAction.triggered.connect(self.clear)

        clearAllAction = QAction("Clear All", self)
        clearAllAction.setShortcut("Ctrl+A")
        optionMenu.addAction(clearAllAction)
        clearAllAction.triggered.connect(self.clearAll)

        self.autosave_button = QAction("Auto Save/Load Scribbles", self, checkable=True, checked=True)
        optionMenu.addAction(self.autosave_button)

        # load image and binary masks
        albedo_filename = pth.join(self.work_dir, 'albedo.png')
        normal_filename = pth.join(self.work_dir, 'normal.png')
        roughness_filename = pth.join(self.work_dir, 'roughness.png')

        albedo = expand_dim_to_3(load_image(albedo_filename))
        height = expand_dim_to_3(load_image(normal_filename))
        roughness = expand_dim_to_3(load_image(roughness_filename))
        self.material_maps = [albedo, height, roughness]
        self.bmasks = load_masks(self.work_dir, n_masks=-1, prefix="bmask")
        self.n_material_maps = len(self.material_maps)
        self.n_bmasks = self.bmasks.shape[0]
        self.num = self.n_material_maps * self.n_bmasks

        self.idx = 0
        self.masked_image_np = None

        self.load()
        self.init_scribbles(filename='scribbles_ref_{}.pkl'.format(self.idx))

    def proc_previous_image(self):
        if self.idx == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Previous Image")
            msg.setText("It's been the first image.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

        if self.autosave_button.isChecked():
            self.save_scribbles(filename='scribbles_ref_{}.pkl'.format(self.idx))

        self.idx -= 1
        self.load()
        self.init_scribbles(filename='scribbles_ref_{}.pkl'.format(self.idx))

    def proc_next_image(self):
        if self.idx == self.num - 1:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Next Image")
            msg.setText("It's been the last image.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

        if self.autosave_button.isChecked():
            self.save_scribbles(filename='scribbles_ref_{}.pkl'.format(self.idx))

        self.idx += 1
        self.load()
        self.init_scribbles(filename='scribbles_ref_{}.pkl'.format(self.idx))

    def load(self):
        i_mat = self.idx // self.n_bmasks
        i_mask = self.idx % self.n_bmasks

        material_map = self.material_maps[i_mat]
        mask_map = self.bmasks[i_mask]

        masked = material_map * mask_map[..., np.newaxis]
        if masked.shape[-1] != 3:
            masked = np.concatenate((masked, masked, masked), axis=2)

        self.masked_image_np = masked.copy()

        masked = (masked * 255.0).astype(np.uint8)

        # some hacks to make the code compatible with functions in the base class
        self.albedo = self.convert_to_pixmap(masked)
        self.label_albedo.setPixmap(self.albedo)
        self.normal = self.albedo.copy()
        self.roughness = self.albedo.copy()

        try:
            name = self.idx2name()
            previous_fmasks = load_masks(self.work_dir, -1, prefix="fmask{}".format(name))
        except RuntimeError:
            self.intended_label.setPixmap(self.albedo)
            self.unwanted_label.setPixmap(self.placeholder)
        else:
            # show on the screen
            intended_image = self.masked_image_np * previous_fmasks[0][..., np.newaxis]
            intended_image = (intended_image * 255.0).astype(np.uint8)
            unwanted_image = self.masked_image_np * previous_fmasks[1][..., np.newaxis]
            unwanted_image = (unwanted_image * 255.0).astype(np.uint8)
            intended_image_qpixmap = self.convert_to_pixmap(intended_image)
            unwanted_image_qpixmap = self.convert_to_pixmap(unwanted_image)
            self.intended_label.setPixmap(intended_image_qpixmap)
            self.unwanted_label.setPixmap(unwanted_image_qpixmap)

        self.state = 'SVBRDF Loaded'
        self.show_status()

    def idx2name(self):
        mat_name = ['Albedo', 'Normal', 'Roughness']
        i_mat = self.idx // self.n_bmasks
        i_mask = self.idx % self.n_bmasks
        return mat_name[i_mat] + str(i_mask)

    def segment(self):
        if self.autosave_button.isChecked():
            self.save_scribbles(filename='scribbles_ref_{}.pkl'.format(self.idx))

        self.state = 'Segmenting'
        self.show_status()

        scribbles = []
        for layer in self.layers.values():
            # convert layers to numpy arrays
            array = self.convert_to_numpy_array(layer)
            mask = np.zeros(array.shape[:2], dtype=bool)
            strokes = np.any(array != [0, 0, 0], axis=-1)
            mask[strokes] = 1
            if np.any(mask):
                scribbles.append(mask)

        if len(scribbles) < 2:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Warning")
            msg.setText("No need to segment this image.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
            return

        scribbles = np.stack(scribbles, axis=0)
        image = self.masked_image_np
        mask = self.bmasks[self.idx % self.n_bmasks]

        try:
            alpha, masks = refined_matting(image, mask, scribbles, spectra_weight=self.spectra_weight.value(),
                                           data_path=self.work_dir, idx=self.idx)
        except Exception as e:
            print(e)
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("Error")
            msg.setText("Failed to perform svbrdf matting")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

            return

        alpha = np.transpose(alpha, axes=(2, 0, 1))
        masks = np.transpose(masks, axes=(2, 0, 1))
        masks = masks.astype(bool)
        fmasks = mask_binarization(alpha)

        # save to disk
        name = self.idx2name()
        save_masks_as_image(fmasks, self.work_dir, prefix='fmask{}'.format(name))

        # show on the screen
        intended_image = self.masked_image_np * fmasks[0][..., np.newaxis]
        intended_image = (intended_image * 255.0).astype(np.uint8)
        unwanted_image = self.masked_image_np * fmasks[1][..., np.newaxis]
        unwanted_image = (unwanted_image * 255.0).astype(np.uint8)
        intended_image_qpixmap = self.convert_to_pixmap(intended_image)
        unwanted_image_qpixmap = self.convert_to_pixmap(unwanted_image)
        self.intended_label.setPixmap(intended_image_qpixmap)
        self.unwanted_label.setPixmap(unwanted_image_qpixmap)

        self.state = 'Finished'
        self.show_status()


def start_app():
    app = QApplication(sys.argv)
    window = SVBRDFMatting()
    window.show()
    app.exec_()
