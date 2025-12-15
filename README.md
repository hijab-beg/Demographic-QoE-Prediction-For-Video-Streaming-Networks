# Demographic-QoE-Prediction-For-Video-Streaming-Networks

A novel demographic-aware Quality of Experience (QoE) prediction system for adaptive video streaming networks using advanced data augmentation techniques and machine learning. 

## Overview

This project implements a state-of-the-art QoE prediction framework that incorporates demographic profiling to improve prediction accuracy for diverse user populations in video streaming networks. By leveraging demographic-specific data augmentation, the system enhances model generalization and provides more accurate QoE predictions across different user segments.

## Key Features

- **Demographic Profiling**: Six distinct user profiles with varying QoE sensitivities
- **Smart Data Augmentation**: Novel MOS (Mean Opinion Score) adjustment algorithm based on demographic characteristics
- **Enhanced Prediction Accuracy**: Improved model performance through augmented training data (450 → 2,700 samples)
- **Comprehensive Analysis**: End-to-end pipeline from data augmentation to model evaluation
- **Interactive Notebooks**: Jupyter notebooks for experimentation and visualization

## Project Structure

```
Demographic-QoE-Prediction-For-Video-Streaming-Networks/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore rules
│
├── data/                              # Dataset directory
│   └── (place your datasets here)
│
├── notebooks/                         # Jupyter notebooks
│   ├── demographic_augmentation.ipynb # Demographic profiling & augmentation
│   └── QoE_prediction.ipynb          # QoE prediction models & evaluation
│
└── src/                               # Source code modules
    ├── augmentation/
    │   └── demographic_augmentation.py
    └── qoe/
        └── qoe_prediction.py
```

## Methodology

### Demographic Profiles

The system defines six distinct user profiles based on sensitivity to QoE factors:
1. Quality-sensitive users
2. Latency-sensitive users
3. Bandwidth-constrained users
4. Mobile users
5. Average users
6. Tolerant users

### Augmentation Pipeline

1. **Base Dataset**: 450 streaming sessions with QoE metrics
2. **Demographic Transformation**: Apply profile-specific MOS adjustments
3. **Augmented Dataset**: 2,700 samples (6× expansion)
4. **Model Training**: Train on diverse demographic representations

### Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score
- Cross-validation performance

## Results

The demographic augmentation approach demonstrates: 
- Improved model generalization across user populations
- Enhanced prediction accuracy for diverse demographic segments
- Reduced overfitting through balanced training data
- Better handling of edge cases and minority user groups

## Technologies Used

- **Python**: Core programming language
- **Jupyter Notebook**: Interactive development environment
- **Pandas & NumPy**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning models and evaluation
- **Matplotlib & Seaborn**: Data visualization

## Key Contributions

1. Novel demographic profiling framework for QoE prediction
2. Data augmentation algorithm tailored to user characteristics
3. Empirical validation of demographic-aware model training
4. Open-source implementation for reproducible research

## Authors

- [hijab-beg](https://github.com/hijab-beg)
- [maryamss-hub](https://github.com/maryamss-hub)
- [zunaira-ahmd](https://github.com/zunaira-ahmd)

## License

Please cite appropriately if you use this work in your research or projects.

## Contact

For questions or collaborations:
- Open an issue in this repository
- Reach out via GitHub

## Acknowledgments

Special thanks to the research community working on QoE prediction and video streaming optimization.

---

If you find this project useful, please consider giving it a star!
