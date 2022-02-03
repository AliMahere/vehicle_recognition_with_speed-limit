### assesmentTask
assessment task part of hiring process in A1Softic 
## work doen
1. first search for sutable and annotated dataset that can fit this task  
2. second train costum [yolo5] (https://github.com/ultralytics/yolov5) model 
3. seprate code needed for deploying model 
4. create **Detector class** that be inizlized and be use to detect **[ambulance , bus,car, motocycle, trank]**
5. create centroid tracker customized for this task 
6. functin to claculate pixile speed 
7. calculate avarge speed and check spped limit for each class 
# run the porject 
first install packges using 
`pip install -r requirements.txt`
###### second run TestingScript 
`python TestingScript.py`
> you can oppen it and modify it as you can 
## running sample

https://user-images.githubusercontent.com/43875252/152268658-31f4724c-c790-4294-b863-9709be34d01b.mp4


## dataset 
[Vehicles-OpenImages](https://public.roboflow.com/object-detection/vehicles-openimages) this dataset contains only 627 image and 1,194 annotation with
an implance class distribution biasd towords **Car** class
- Car 651
- Bus 141
- Motorcycle 140
- Truck 136
- Ambulance 126
> this project doesn't include any data augmentation or handling implance clasess 
 ## model 

