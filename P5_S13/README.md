# YOLO V2/3

* This session deals with training Yolo on Google Colab
* use YoloV3 annotation tool to get dataset annotated in the format required by the model
* Followed this repo to perform Yolo training on our custom dataset: https://github.com/theschoolofai/YoloV3

### Extract frames from Video using ffmpeg:
```
❯ ffmpeg -i YoloDatasetVideo.mp4 -r <number-of-fps> $filename%03d.jpg
```

### Merge frames to form Video using ffmpeg:
```
❯ ffmpeg -r <number-of-fps> -i %03d.jpg out.mp4
```
