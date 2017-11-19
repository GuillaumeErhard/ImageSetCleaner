import sys
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtWidgets import QWidget, QApplication, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel, QScrollArea, QAction, \
    QMainWindow, QStyle, qApp, QFileDialog, QSlider, QComboBox, QPushButton
from File_Processing import *
from Predicting import *

# https://stackoverflow.com/questions/3157766/how-can-i-achieve-layout-similar-to-google-image-search-in-qt-pyqt/3160725
# https://stackoverflow.com/questions/38223705/adding-items-from-a-text-file-to-qlistwidget-in-pyqt5
# http://pythoncentral.io/pyside-pyqt-tutorial-the-qlistwidget/
# http://pythoncentral.io/pyside-pyqt-tutorial-qlistview-and-qstandarditemmodel/
# https://stackoverflow.com/questions/46493125/pyqt-change-picture-with-keyboard-button

# TODO: Watch this
# https://stackoverflow.com/questions/8814452/pyqt-how-to-add-separate-ui-widget-to-qmainwindow
# /!\ Pour voir tous les icons dispo
# https://joekuan.wordpress.com/2015/09/23/list-of-qt-icons/


class MainWindow(QMainWindow):
    def __init__(self, image_dir, image_set, paths, clustering_method, architecture, pollution_dir, pollution_percent, parent=None):
        super(MainWindow, self).__init__(parent)

        self.image_dir = image_dir
        self.image_set = image_set
        self.architecture = architecture
        self.pollution_dir = pollution_dir
        self.paths_processed = []

        deleteAct = QAction(qApp.style().standardIcon(QStyle.SP_TrashIcon), 'Delete Selection', self)
        deleteAct.setStatusTip('Will delete the selected images')
        deleteAct.triggered.connect(self.delete_images)

        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(deleteAct)

        moveAct = QAction(qApp.style().standardIcon(QStyle.SP_FileDialogStart), 'Move Selection', self)
        moveAct.triggered.connect(self.move_images)

        self.toolbar.addAction(moveAct)

        self.classifier_combo = QComboBox(self)

        for method in CLUSTERING_METHODS:
            self.classifier_combo.addItem(method)

        self.classifier_combo.setCurrentIndex(CLUSTERING_METHODS.index(clustering_method))
        self.classifier_combo.currentIndexChanged.connect(self.restore_button)

        self.toolbar.addWidget(self.classifier_combo)

        self.pollution_slider = QSlider(QtCore.Qt.Horizontal, self)
        self.pollution_slider.setRange(0, 40)
        self.pollution_slider.setStyleSheet(self.stylesheet())
        self.pollution_slider.setValue(pollution_percent)
        # #pollution_slider.setGeometry(1, 1, 1, 1)
        self.pollution_slider.valueChanged.connect(self.restore_button)
        # pollution_slider.sliderReleased.connect(self.pollutionValueWanted)
        # #pollution_slider.sliderMoved.connect(self.pollutionValueWanted)
        # #pollution_slider.sliderPressed.connect(self.pressed)
        # pollution_slider.rangeChanged.connect(self.rangechanged)
        # pollution_slider.adjustSize()

        #pollution_slider = PollutionSelection(pollution_percent)

        self.toolbar.addWidget(self.pollution_slider)

        self.predictionButton = QPushButton('Predictions', self)
        self.predictionButton.clicked.connect(self.get_new_predictions)
        self.predictionButton.setDisabled(True)

        self.somethingChanged = False

        self.toolbar.addWidget(self.predictionButton)

        self.window = Window(paths)
        self.setCentralWidget(self.window)

        self.show()

    def delete_images(self):
        current_selection = self.window.get_selection()
        # TODO : Uncomment this
        #delete_images(current_selection)

        # TODO : Check if empty then do nothing. Peut être fenêtre qui s'ouvre ?

        # TODO : Diff entre changement de classifier donc un update ou reload ? / new_window_afterchangeclassif ?
        self.paths_processed += current_selection
        remaining_paths = self.window.paths

        for path in self.paths_processed:
            try:
                remaining_paths.remove(path)
            except ValueError:
                pass

        self.window = Window(remaining_paths)
        self.setCentralWidget(self.window)

    def move_images(self):
        current_selection = self.window.get_selection()
        # TODO : Check if empty then do nothing. Peut être fenêtre qui s'ouvre ?
        print(current_selection)

        dir_relocation = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        print(dir_relocation)

        move_images(dir_relocation, current_selection)

        self.paths_processed += current_selection
        remaining_paths = self.window.paths
        print(self.paths_processed)
        for path in self.paths_processed:
            try:
                remaining_paths.remove(path)
            except ValueError:
                pass

        self.window = Window(remaining_paths)
        self.setCentralWidget(self.window)

    def restore_button(self):
        if not self.somethingChanged:
            self.somethingChanged = True
            self.predictionButton.setDisabled(False)

    def stylesheet(self):
        return """
                   QSlider::groove:horizontal {
            border: 1px solid #999999;
            height: 8px; /* the groove expands to the size of the slider by default. by giving it a height, it has a fixed size */
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #B1B1B1, stop:1 #c4c4c4);
            margin: 2px 0;
        }

        QSlider::handle:horizontal {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
            border: 1px solid #5c5c5c;
            width: 18px;
            margin: -2px 0; /* handle is placed by default on the contents rect of the groove. Expand outside the groove */
            border-radius: 3px;
        }
        """

    def get_new_predictions(self):
        print('Boom', self.pollution_slider.value(), self.classifier_combo.currentText())

        predictions = semi_supervised_detection(self.image_set, self.classifier_combo.currentText(), self.architecture,
                                                self.pollution_dir, float(self.pollution_slider.value()) / 100)

        image_paths = get_image_paths(self.image_dir, predictions)
        # TODO : Check if path already used

        self.window = Window(image_paths)
        self.setCentralWidget(self.window)

        self.predictionButton.setDisabled(True)
        self.somethingChanged = False


class ClikableLabel(QLabel):

    def __init__(self, path):
        super(ClikableLabel, self).__init__()
        self.width = 180
        self.height = 200
        self.isChecked = False

        pixmap = QPixmap(path)
        pixmap = pixmap.scaled(self.width, self.height, QtCore.Qt.KeepAspectRatio)
        self.setPixmap(pixmap)

    def mousePressEvent(self, event):
        self.isChecked = not self.isChecked

        if self.isChecked:
            self.setStyleSheet("border: 5px inset red;")
        else:
            self.setStyleSheet("")


class Window(QScrollArea):
    def __init__(self, paths):
        QScrollArea.__init__(self)

        # scroll = QScrollArea()
        # scroll.setWidgetResizable(True)
        # scroll.setFixedHeight(400)
        widget = QWidget()
        self.layout = QGridLayout(widget)
        self.paths = paths
        self.all_labels = []
        self.nb_columns = 5

        self.populate_grid(self.paths)



        self.setWidget(widget)

        self.setWidgetResizable(True)
        self.setMinimumWidth(1000)
        self.setMinimumHeight(600)

    def populate_grid(self, paths):
        row = 0
        column = 0
        for idx, path in enumerate(paths):
            label = ClikableLabel(path)
            self.all_labels.append(label)
            self.layout.addWidget(self.all_labels[idx], row, column)

            column += 1
            if column % self.nb_columns == 0:
                row += 1
                column = 0

    def get_selection(self):
        selection = []
        for idx, path in enumerate(self.paths):
            if self.all_labels[idx].isChecked:
                selection.append(path)

        return selection


if __name__ == '__main__':

    img_list = ['./Test_cluster_1/img01.jpg', './Test_cluster_1/img02.jpg']
    path = './Test_cluster_1/'
    images = os.listdir(path)
    img_list = [os.path.join(path, image) for image in images]

    app = QApplication(sys.argv)
    window = MainWindow(img_list, 'kmeans', 20)
    # window.setGeometry(500, 300, 200, 200)
    # window.show()
    sys.exit(app.exec_())