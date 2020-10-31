# RCNN Family

### Inference on pre-trained Detectron model
* Ran a video on a pre-trained Detectron2 network. Result was uploaded to youtube:
  [![Alt text](https://img.youtube.com/vi/vEUT4G0NxmE/0.jpg)](https://www.youtube.com/watch?v=vEUT4G0NxmE)

### Dataset Generation for PPE classes

* Generate datasets from the original PPE dataset images. Need the following datasets:
  1. Depth Images - This will be from MiDaS network [https://github.com/intel-isl/MiDaS]. MiDaS computes depth from a single image. We perform this on all the images and store it
  2. Planer Images - From PlaneR-CNN network [https://github.com/NVlabs/planercnn]. PlaneR-CNN detects arbitrary number of planes, and reconstructs piecewise planar surfaces from a single RGB image. It also generates depth images, which we will not be using as depth images from MiDaS are way better that PlaneR-CNN's output
  3. Bounding Boxes - Already collected by using YoloV3 annotation program
* Above generated dataset is collected and stored in a single drive folder [https://drive.google.com/drive/folders/1ms6H8JVcTzLD8INZHSQiIVxQ6WgSYpKW?usp=sharing]

