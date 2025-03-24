# Aided Active Learning for Enhanced Critical Heat Flux Prediction

**NURETH-21: Aided Active Learning for Enhanced Critical Heat Flux Prediction**

---

## 📄 Paper

Nabila, U.M., Radaideh, M.I., Burnett, L.A., Lin, L., Radaideh, M.I. (2025). “Aided Active Learning for Enhanced Critical Heat Flux Prediction”, In: *21st International Topical Meeting on Nuclear Reactor Thermal Hydraulics (NURETH-21)*, Busan, Korea, August 31 – September 5, 2025.

## ⚙️Environment Installation

To set up the environment for this project, follow these steps:
```bash
# 1. Create a new conda environment with Python 3.11
conda create -n torchgpu python=3.11
# 2. Activate the environment
conda activate torchgpu
# 3. Install required libraries
pip install -r requirements.txt
```
Check whether Nvidia-cuda was installed using 
```bash
import torch
print(torch.cuda.is_available())
```
If this prints ```False```, you can download torch+cuda from [Pytorch](https://pytorch.org/get-started/locally/) website.

## 📊 How to Generate the Results

- The folder `data` contains input data files used by the model scripts to generate results.

-  Go to the folder `models` and run the desired script (e.g., `viAL.py`) to start the training or evaluation process.
```bash
python viAL.py
```

Results will be saved automatically in the `results` folder.

