## Object Localisation

* To understand object localisation, following is done in this project:
  * Collect dataset of people wearing hardhat,mask,vest,boots
  * Annotations i.e. bounding boxes of above classes collected using VGG annotation tools
  * Peform K-means clustering on bounding boxes on above collected dataset

#### Understanding annotation format by VGG annotation tool
```
{
  "img001.jpg375173": {
    "filename": "img001.jpg",
    "size": 375173,
    "regions": [
      {
        "shape_attributes": {
          "name": "rect",
          "x": 164,
          "y": 258,
          "width": 66,
          "height": 45
        },
        "region_attributes": {
          "class": "hardhat"
        }
      },
      .
      .
      {
        "shape_attributes": {
          "name": "rect",
          "x": 134,
          "y": 603,
          "width": 96,
          "height": 80
        },
        "region_attributes": {
          "class": "boots"
        }
      }
    ],
    "file_attributes": {}
  },
```
* Above is a snippet from annotation JSON output of VGG annotation tool
* Each entry is of an image file and attributes associated with that image
* It is key-value pair, with key being filename concatenated with the size of the file. From above example, it is `img001.jpg375173`, where `img001.jpg` is filename and `375173` is file size in bytes
* The value of each entry contains attributes about the file like: `filename`, `size` in bytes, `regions`, `file_attributes`
* `regions` are bounding boxes of region of interest in the image. Their centroid (`x`, `y`) and their `width` and `height` are stored as part of `shape_attributes`. Each bounding box is labeled with one or more `region_attributes`. These `region_attributes` are key-value pairs used to store metadata about the selected boxes (boudning box) in the image
* `file_attributes` are key-value pairs used to store metadata about complete file itself
* Above annotation tool is mainly used to annotate (in above format) objects in the image and later use this to train the network to predict such bounding boxes on test/validation dataset
