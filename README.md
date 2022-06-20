# robocup2022-cv-object-tracker
Yolov5 + StrongSORT with OSNet
## Installation

1. Clone this project
2. Download all files from [LINK TO DRIVE](https://drive.google.com/file/d/16yhkaBn8zH6o6qcHhJ5FfsZWkLvac7Sm/view?usp=sharing)
3. Copy and paste any files from .zip to root folder
4. Create conda environment
```
$ conda env create -f environment.yml
```
- Or Install requirements.txt
```
$ pip install -r requirements.txt
```

## Weight
You can change weight.pt by put a new weight in weight folder and change a parameter in object_tracker.py
- YOLO_WEIGHTS_PATH = WEIGHTS / 'custom-weight.pt' 
- STRONG_SORT_WEIGHTS = WEIGHTS / 'custom-strong_sort-weight'



