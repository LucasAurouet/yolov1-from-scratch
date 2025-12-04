\# YOLOv1 From Scratch



Implementation from scratch of \*\*YOLOv1 (You Only Look Once)\*\* for object detection, with a demonstration on a playing cards dataset.



---



\## 📂 Project Structureyolov1-from-scratch/

│

├── src/                  # Source code (package)

│   ├── \_\_init\_\_.py

│   ├── yolo\_model.py          # YOLOv1 model architecture

│   ├── loss.py           # YOLOv1 loss function

│   ├── utils.py          # Utilities: IoU, indicator functions

│

│

├── notebooks/            # Jupyter notebooks

│   └── yolov1-from-scrach.ipynb  # Demo on playing cards detection

│

├── outputs/              # Trained models, logs, results (ignored in Git)

├── requirements.txt      # Python dependencies

└── README.md



---



\## Features



* YOLOv1 model implemented from scratch in PyTorch. Modular code with separate model, loss, utils



* Configurable hyperparameters for grid size (S), bounding boxes (B), and loss weights



* Example dataset: playing cards from roboflow (https://universe.roboflow.com/augmented-startups/playing-cards-ow27d)



* Notebook demo for inference and visualization



---



\## Installation



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



\## Notebook Demo



* Open notebooks/yolov1-card-detection.ipynb



* Demonstrates inference, visualization, and evaluation on the playing cards dataset.



\## Structure Highlights



* src/: Python package for YOLOv1, easy to import in notebooks



* notebooks/: Examples and demos



* outputs/: Trained models and visualizations (ignored in Git)



\## Notes



* Outputs (.pt models, logs) are excluded via .gitignore



* Designed to run on GPU
