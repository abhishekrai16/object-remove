#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import cv2
import os
from objRemove import ObjectRemove
from models.deepFill import Generator
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

##################################
#get image path from command line#
##################################
# parser = argparse.ArgumentParser()
# parser.add_argument("image")
# args = parser.parse_args()
# image_path = args.image

import tkinter as tk
from tkinter import filedialog

# Create a Tkinter window
root = tk.Tk()
root.withdraw()  # Hide the main window

# Open a file dialog to select an image
image_path = filedialog.askopenfilename(title="Select Image")

# Check if the user selected an image or canceled
if image_path:
    print("Selected image:", image_path)
else:
    print("No image selected.")





######################################################
#creating Mask-RCNN model and load pretrained weights#
######################################################
for f in os.listdir('src/models'):
    if f.endswith('.pth'):
        deepfill_weights_path = os.path.join('src/models', f)
print("Creating rcnn model")
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
transforms = weights.transforms()
rcnn = maskrcnn_resnet50_fpn(weights=weights, progress=False)
rcnn = rcnn.eval()

#########################
#create inaptining model#
#########################
print('Creating deepfil model')
deepfill = Generator(checkpoint=deepfill_weights_path, return_flow=True)
######################
#create ObjectRemoval#
######################
model = ObjectRemove(segmentModel=rcnn,
                        rcnn_transforms=transforms, 
                        inpaintModel=deepfill, 
                        image_path=image_path )
#####
#run#
#####
output = model.run()

#################
#display results#
#################
img = cv2.cvtColor(model.image_orig[0].permute(1,2,0).numpy(),cv2.COLOR_RGB2BGR)
boxed = cv2.rectangle(img, (model.box[0], model.box[1]),(model.box[2], model.box[3]), (0,255,0),2)
boxed = cv2.cvtColor(boxed,cv2.COLOR_BGR2RGB)

fig,axs = plt.subplots(1,3,layout='constrained')
axs[0].imshow(boxed)
axs[0].set_title('Original Image Bounding Box')
axs[1].imshow(model.image_masked.permute(1,2,0).detach().numpy())
axs[1].set_title('Masked Image')
axs[2].imshow(output)
axs[2].set_title('Inpainted Image')
plt.show()


import sys
import cv2
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
import matplotlib.pyplot as plt

from objRemove import ObjectRemove
from models.deepFill import Generator
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

class InpaintingApp(QMainWindow):
    def _init_(self):
        super()._init_()
        self.title = 'Image Inpainting'
        self.initUI()
        
        # Load model
        self.rcnn, self.transforms = self.load_rcnn_model()
        self.deepfill = self.load_deepfill_model()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        # Button for browsing images
        self.btnBrowse = QPushButton('Browse Image', self)
        self.btnBrowse.clicked.connect(self.openFileNameDialog)
        layout.addWidget(self.btnBrowse)

        # Label to display image
        self.label = QLabel(self)
        layout.addWidget(self.label)

        # Button to run inpainting
        self.btnRun = QPushButton('Run Inpainting', self)
        self.btnRun.clicked.connect(self.run_inpainting)
        layout.addWidget(self.btnRun)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.show()

    def openFileNameDialog(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files ();;JPEG (.jpg; .jpeg);;PNG (.png)", options=options)
        if fileName:
            self.image_path = fileName
            pixmap = QPixmap(fileName)
            self.label.setPixmap(pixmap.scaled(800, 600, aspectRatioMode=1))

    def load_rcnn_model(self):
        print("Creating rcnn model")
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        transforms = weights.transforms()
        rcnn = maskrcnn_resnet50_fpn(weights=weights, progress=False)
        rcnn = rcnn.eval()
        return rcnn, transforms

    def load_deepfill_model(self):
        print('Creating deepfil model')
        for f in os.listdir('src/models'):
            if f.endswith('.pth'):
                deepfill_weights_path = os.path.join('src/models', f)
        deepfill = Generator(checkpoint=deepfill_weights_path, return_flow=True)
        return deepfill

    def run_inpainting(self):
        if hasattr(self, 'image_path'):
            model = ObjectRemove(segmentModel=self.rcnn,
                                 rcnn_transforms=self.transforms,
                                 inpaintModel=self.deepfill,
                                 image_path=self.image_path)
            output = model.run()
            output_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            cv2.imwrite('temp_output_image.png', output_image)
            pixmap = QPixmap('temp_output_image.png')
            self.label.setPixmap(pixmap.scaled(800, 600, aspectRatioMode=1))
            os.remove('temp_output_image.png')


if _name_ == '_main_':
    app = QApplication(sys.argv)
    ex = InpaintingApp()
    sys.exit(app.exec_())
