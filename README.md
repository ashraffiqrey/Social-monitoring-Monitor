# Social-monitoring-Monitor
This it the tool used to monitor social distancing by inputing either image/video as input or from CCTV. JAVA, Deep Learning, and Computer Vision are used. The dataset being 
used are COCO dataset and VOC dataset. 

The pretrained model being used is YOLOV3 for image/video input and TINYYOLO for CCTV. The reason of using 2 pretrained model are 
because TINYYOLO is more suitable to be used for live video because TINYYOLO is much faster than YOLO although the accuracy is lesser than YOLO.

This tools works by detecting the object (which is person), draw a bounding box and find the centre for the bounding box, calculate the distance between each 
centre by using euclidean distance.
