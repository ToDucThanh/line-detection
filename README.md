# Dataset

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
