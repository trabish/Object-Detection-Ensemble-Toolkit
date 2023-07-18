## Object Detection Ensemble Toolkit

Parameter search-based ensemble method for object detection

## Requirements

The code is built with following libraries:

- [mmdetection](http://github.com/open-mmlab/mmdetection) = 3.0.0
- [nni](https://github.com/microsoft/nni) = 2.10.1

## Usage examples

```shell
python nni_main.py
```

## TODO List
- [x] Parameter search based on [NNI](https://github.com/microsoft/nni)
- [ ] Multi-threading
- [ ] Support multiple detection frameworks
- [ ] More nms methods
- [ ] Visualization
- [ ] Instance segmentation


## Acknowledgements
This repository is based on [Weighted-Boxes-Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion).
