# Line detection

## Wireframe dataset

The Wireframe dataset consists of 5,462 images (5,000 for training, 462 for test) of indoor and outdoor man-made scenes. \
Dataset link:
[One Drive](https://1drv.ms/u/s!AqQBtmo8Qg_9uHpjzIybaIfyJ-Zf?e=Fofbch)

### Raw dataset

Each pickle (.pkl) file contains the ground truth of an image, as follow:

```bash
*.pkl  
    |-- imagename: the name of the image  
    |-- img: the image data  
    |-- points: the set of points in the wireframe, each point is represented by its (x,y)-coordinates in the image  
    |-- lines: the set of lines in the wireframe, each line is represented by the indices of its two end-points  
    |-- pointlines: the set of associated lines of each point        
    |-- pointlines_index:  line indexes of lines in 'pointlines'  
    |-- junction: the junction locations, derived from the 'points' and 'lines'  
    |-- theta: the angle values of branches of each junction
```

### Data preparation

The script [preparation_ds_wireframe](src/data/preparation_ds_wireframe.py) generates the the input for data loader. \
We will generate 2 JSON file for training process: **train.json**, **valid.json** with the structure as follow:

```bash
train.json 
    {
        "image": np.ndarray, # the image itself
        "filename": str, # Ex: 00031546.png
        "lines": List(), # list of all lines, each lines is represented as [x1, y1, x2, y2]
        "height": ..., # height of the image
        "width": ... # width of the image
    }
```

### Alternative dataset format

```bash
|── src/
    └── data/
        └── raw/
            └── train/
                └── images/
                    ├── 0000001.png\n
                    └── 0000002.png
                └── labels/
                    ├── 0000001.txt\n
                    └── 0000002.txt
            └── tests/
                └── images/
                    ├── 0000003.png\n
                    └── 0000004.png
                └── labels/
                    ├── 0000003.txt\n
                    └── 0000004.txt

Each line of the txt file is the coordinates of one line in the corresponding image.
Example:
    File 0000001.txt:
    372.43095088, 118.95949936, 374.10025597, 212.82363129
    435.50227356, 123.33520508, 505.86045074,  -7.40013885
    ....
Interpret: (x1, y1, x2, y2)
```

## Setup

> **_NOTE:_** Right now, the `dev` branch is the most up-to-date.

### Option 1: Install with poetry

Clone the repository and checkout dev branch:

```bash
git clone https://github.com/ToDucThanh/line_detection.git
cd line_detection
git checkout dev
```

Install the environments:

```bash
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools
pip install poetry
poetry install --no-root
```

If you only want to run demo, run:

```bash
poetry install --no-root --without dev
```

### Option 2: Install without poetry

Clone the repository and checkout dev branch:

```bash
git clone https://github.com/ToDucThanh/line_detection.git
cd line_detection
git checkout dev
```

Install the environments:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Demo

Input: image or video.
Output: list of numpy array: [np.array([x1, y1, x2, y2]), ...]

Run with one image:

```bash

from workflow.line_detection_dev import LineDetectionDemo

line_detector = LineDetectionDemo(
    image_path="src/workdir/experiments/00030043.jpg"
)
outputs = line_detector.run()
```

Run with one video:

```bash

from workflow.line_detection_dev import LineDetectionDemo

line_detector = LineDetectionDemo(
    video_path="src/videos/sample_2images_20fps.mp4"
)
outputs = line_detector.run()
```
