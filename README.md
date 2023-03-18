# object-remove

An object removal from image system using deep learning image segmentation and inpainting techniques.

## Contents
1. [Overview](#overview)
2. [Source Code](src/)
3. [Report](object_remmove.pdf)
4. [Results](#results)

## Overview
 Object removal from image involves two separate tasks, object detection and object removal.

 The first task is handled by the user drawing a bounding box around an object of interest to be removed. We could then remove all pixels inside the bounding box, but this could lead to loss of valuable information from the pixels in the box that are not part of the object. Instead Mask-RCNN, a state of the art instance segmentation model is used to get the exact mask of the object.  

 Filling in the image is done using DeepFillv2, an image inpainting generative adversarial network which employs a gated convolution system.
 
 The result is a complete image with the object removed. 

 <p align ="center">
  <img src="/img/diagram.png" width="1000" />
  <em></em>
 </p>

## Results
The following are some results of the system. We have a green bounding box for the user object selection and the final inpainted results. 
<p align ="center">
  <img src="/img/example1.png" width="1000" />
  <em></em>
</p>
<p align ="center">
  <img src="/img/example2.png" width="1000" />
  <em></em>
</p>


