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

### Sample Videos and their annotations
* Can be found in sample_videos folder
* Annotations are on Youtube:
  * video1:
  [![Alt text](https://img.youtube.com/vi/5q4j3JOMBtc/0.jpg)](https://www.youtube.com/watch?v=5q4j3JOMBtc)
  * video2:
  [![Alt text](https://img.youtube.com/vi/PN-TCIcZW5E/0.jpg)](https://www.youtube.com/watch?v=PN-TCIcZW5E)
