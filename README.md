# CNET QoE Prediction

## Overview
This project implements a novel demographic-aware QoE (Quality of Experience) prediction system for adaptive video streaming using data augmentation techniques.

## Research Contributions
1. **Demographic Profiling**: Six distinct user profiles with varying QoE sensitivities
2. **Augmentation Algorithm**: Novel MOS adjustment based on demographic characteristics
3. **Performance Enhancement**: Improved prediction accuracy through augmented training data

## Project Structure
```
cnet-qoe/
│
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
│
├── paper/                       # Research papers and documentation
│   └── base_paper.pdf
│
├── src/                         # Source code modules
│   ├── augmentation/            # Data augmentation modules
│   │   └── demographic_augmentation.py
│   │
│   └── qoe/                     # QoE prediction modules
│       └── qoe_prediction.py
│
├── notebooks/                   # Jupyter notebooks for experiments
│   ├── demographic_augmentation.ipynb
│   └── QoE_prediction.ipynb
│
├── data/                        # Data directory (not tracked)
│
└── outputs/                     # Results and figures (not tracked)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cnet-qoe.git
cd cnet-qoe
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Demographic Augmentation
```python
from src.augmentation.demographic_augmentation import DemographicAugmentation

# Initialize augmentation
augmentor = DemographicAugmentation(df)

# Apply demographic-based augmentation
augmented_df = augmentor.augment()
```

### QoE Prediction
```python
from src.qoe.qoe_prediction import QoEPredictor

# Initialize predictor
predictor = QoEPredictor()

# Train and evaluate model
results = predictor.train_and_evaluate(X_train, y_train, X_test, y_test)
```

## Notebooks
- **demographic_augmentation.ipynb**: Implements demographic profiling and data augmentation
- **QoE_prediction.ipynb**: QoE prediction models and evaluation

## Dataset
- **Input**: Base QoE dataset (450 streaming sessions)
- **Process**: 6× augmentation with demographic-specific transformations
- **Output**: Enhanced dataset (2,700 samples)

## Results
The demographic augmentation approach improves model generalization and prediction accuracy for diverse user populations.

## License
This project is part of academic research. Please cite appropriately if you use this work.

## Contact
For questions or collaboration, please open an issue in this repository.
