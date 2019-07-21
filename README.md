# PyTorch-YOLOv3 For text Detection
The code was adopted from inilial code that can be found at https://github.com/eriklindernoren/PyTorch-YOLOv3

## The code was modified to work with Detext dataset and changes to code are required to work on different datasets

Modification from original code now supports Torch v 0.4.
Results now being logged to text files as well Visdom dashboard.
Utilizing visdom removed the need to use tensorboard and tensorflow, both packages no longer required.
The data loader was also modified to read files from directories without a need of txt log containing path and name for each file.

The work is taking place on this personal project to adapt YOLOv3 for text detection and changes will be regulary added

## Installation
##### Clone and install requirements
    $ git clone https://github.com/Andr1u5/PyTorch-YOLOv3_Text_Detection
    $ cd PyTorch-YOLOv3_Text_Detection/
    $ sudo pip3 install -r requirements.txt


## Train
```
$ train.py [-h] [--epochs EPOCHS] [--batch_size BATCH_SIZE]
                [--gradient_accumulations GRADIENT_ACCUMULATIONS]
                [--model_def MODEL_DEF] [--data_config DATA_CONFIG]
                [--pretrained_weights PRETRAINED_WEIGHTS] [--n_cpu N_CPU]
                [--img_size IMG_SIZE]
                [--checkpoint_interval CHECKPOINT_INTERVAL]
                [--evaluation_interval EVALUATION_INTERVAL]
                [--compute_map COMPUTE_MAP]
                [--multiscale_training MULTISCALE_TRAINING]
                [--log LOG_NAME]
```

#### Training log
training logs kept in the same format as initial implementation but there is no need to use terminaltables
```
---- [Epoch 7/100, Batch 7300/14658] ----
| -----------|--------------|--------------|------------- |
| Metrics    | YOLO Layer 0 | YOLO Layer 1 | YOLO Layer 2 |
| -----------|--------------|--------------|------------- |
| grid_size  | 16           | 32           | 64           |
| loss       | 1.554926     | 1.446884     | 1.427585     |
| x          | 0.028157     | 0.044483     | 0.051159     |
| y          | 0.040524     | 0.035687     | 0.046307     |
| w          | 0.078980     | 0.066310     | 0.027984     |
| h          | 0.133414     | 0.094540     | 0.037121     |
| conf       | 1.234448     | 1.165665     | 1.223495     |
| cls        | 0.039402     | 0.040198     | 0.041520     |
| cls_acc    | 44.44%       | 43.59%       | 32.50%       |
| recall50   | 0.361111     | 0.384615     | 0.300000     |
| recall75   | 0.222222     | 0.282051     | 0.300000     |
| precision  | 0.520000     | 0.300000     | 0.070175     |
| conf_obj   | 0.599058     | 0.622685     | 0.651472     |
| conf_noobj | 0.003778     | 0.004039     | 0.004044     |
| -----------|--------------|--------------|--------------|
Total Loss 4.429395
---- ETA 0:35:48.821929
```

#### Visdom
Track training progress in Visdom:
* Initialize visdom server
* Run the commands below
* $ python -m visdom.server
* The dashboard can now be accessed http://localhost:8097

## Credit
### Initial Code taken from
https://github.com/eriklindernoren/PyTorch-YOLOv3

### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
