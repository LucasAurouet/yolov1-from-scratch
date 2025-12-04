# YOLOv1 From Scratch

Implementation from scratch of **YOLOv1** for object detection, with a demonstration on a playing cards dataset.

---

## Project Structure

```
yolov1-from-scratch/
│
├── src/
│ ├── init.py
│ ├── model.py
│ ├── loss.py
│ ├── utils.py
│ └── train.py
│
├── notebooks/
│ └── cards_demo.ipynb
├── outputs/
├── requirements.txt
└── README.md
```

---

## Features



* YOLOv1 model implemented from scratch in PyTorch. Modular code with separate model, loss, utils



* Configurable hyperparameters for grid size (S), bounding boxes (B), and loss weights



* Example dataset: playing cards from roboflow (https://universe.roboflow.com/augmented-startups/playing-cards-ow27d)



* Notebook demo for inference and visualization



---



## Installation



Clone the repository:



```

git clone https://github.com/username/yolov1-from-scratch.git

cd yolov1-from-scratch

```



Install dependencies:



```

pip install -r requirements.txt

```

---



## Notebook Demo



* Open notebooks/yolov1-card-detection.ipynb



* Demonstrates inference, visualization, and evaluation on the playing cards dataset.



## Structure Highlights



* src/: Python package for YOLOv1, easy to import in notebooks



* notebooks/: Examples and demos



* outputs/: Trained models and visualizations (ignored in Git)



## Notes



* Outputs (.pt models, logs) are excluded via .gitignore



* Designed to run on GPU
