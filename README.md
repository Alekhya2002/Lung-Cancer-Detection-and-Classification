# Lung Cancer Detection and Classification

Deep-learning classification of chest CT scan images into four classes:

- Adenocarcinoma
- Large cell carcinoma
- Squamous cell carcinoma
- Normal

> **Disclaimer:** this is an academic/research project. It is not a medical device and must not be used for diagnosis.

## Dataset

Chest CT scan images split into `train` (613 images), `valid` (72) and `test` (315), with one folder per class:
`adenocarcinoma`, `large.cell.carcinoma`, `normal`, `squamous.cell.carcinoma`.
The layout follows the public Kaggle "Chest CT-Scan images" dataset. The data is not included in this repository.

## Approach

| Notebook | Contents |
|---|---|
| [MP-code.ipynb](MP-code.ipynb) | Main model: EfficientNetB3 (ImageNet weights) fine-tuned at 300x440 input. Training-set class balancing via augmentation, a custom learning-rate-adjusting callback, evaluation with confusion matrix and classification report. |
| [Other.ipynb](Other.ipynb) | Baselines: small CNN, DenseNet201 and ResNet50 (transfer learning, 460x460 input). |

## Results (test set, 315 images)

| Model | Accuracy |
|---|---|
| Small CNN (baseline) | 46% |
| DenseNet201 | 76% |
| ResNet50 | 78% |
| **EfficientNetB3** | **94.3%** (18 errors) |

EfficientNetB3 per-class F1: adenocarcinoma 0.93, large cell 0.94, normal 0.99, squamous cell 0.93 (macro F1 0.95). Squamous cell recall is lowest at 88%.
ResNet50 struggles most with adenocarcinoma (recall 58%, often confused with squamous cell).

Caveats: single training run, small dataset, and the baselines use different input sizes and a limited `steps_per_epoch`, so the comparison is indicative rather than rigorous. Results were not validated on external data.

## GUI demo

[main.py](main.py) is a small Tkinter app: choose a CT image, click **Classify**, and see the predicted class plus a probability bar chart.

```bash
pip install -r requirements.txt
python main.py path/to/model.h5     # or: MODEL_PATH=path/to/model.h5 python main.py
```

The trained weights are not included in this repository. To produce them, run [MP-code.ipynb](MP-code.ipynb) on the dataset above; it saves a file named like `chest scans-4-(300 X 440)-94.29.h5`. Pass that file to `main.py`.

## Repository layout

```
MP-code.ipynb    main EfficientNetB3 pipeline
Other.ipynb      baseline models
MP-code.html     HTML export of the main notebook (with outputs)
main.py          Tkinter inference GUI
layers_plot.png  architecture plot of the EfficientNetB3 model
```
