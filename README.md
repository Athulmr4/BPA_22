# Bacteria Lab Assistant Predictor

This project trains and runs a bacteria + fungi image analysis pipeline for lab assistance.

It performs:
- Organism type prediction (`bacteria` vs `fungi`)
- Gram-type prediction (internal classifier: gram-positive vs gram-negative, with fungi mapped to `fungi`)
- Species prediction (constrained by taxonomy group)
- Colony detection and measurement from input image
- Morphology prediction focused on `cocci`, `bacilli`, `spiral`, with fungal override
- Exact JSON output contracts in **basic** and **advanced** mode

> Dataset root expected: `Bacteria dataset/`

## Output contracts

### Basic output

```json
{
  "organism_type": "bacteria",
  "predicted_bacteria_name": "Bacillus subtilis",
  "bacteria_type": "gram_positive",
  "total_colonies_detected": 42,
  "dominant_shape": "cocci",
  "confidence": 0.81
}
```

### Advanced output

```json
{
  "organism_type": "bacteria",
  "predicted_bacteria_name": "Bacillus subtilis",
  "bacteria_type": "gram_positive",
  "total_colonies": 42,
  "colonies": [
    {
      "id": 1,
      "area": 109.5,
      "perimeter": 45.69,
      "circularity": 0.65,
      "aspect_ratio": 1.02,
      "solidity": 0.89,
      "equivalent_diameter": 11.8,
      "mean_intensity": 132.4,
      "predicted_shape": "cocci"
    }
  ],
  "final_morphology": {
    "dominant_shape": "cocci",
    "distribution": "clustered",
    "confidence": 0.87
  }
}
```

## Notes about spiral + fungi

The provided dataset contains bacteria mainly mapped to cocci/bacilli classes.
Spiral class is produced via morphology heuristics (`aspect_ratio` and `circularity`) when contours match spiral-like geometry.
Fungal samples override the dominant shape to `fungal` for clarity.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Train model

```bash
python train_model.py
```

Trained artifact is saved to:
- `artifacts/bacteria_models.joblib`
- `artifacts/bacteria_models.metrics.json`

## Run prediction

### Basic mode

```bash
python predict_bacteria.py --image "Bacteria dataset/Bacillus subtilis_gram stain/Bacillus subtilis_gram stain_1.png" --mode basic
```

### Advanced mode

```bash
python predict_bacteria.py --image "Bacteria dataset/Bacillus subtilis_gram stain/Bacillus subtilis_gram stain_1.png" --mode advanced
```

## Run simple PyQt5 UI

```bash
python bacteria_ui.py
```

UI features:
- Upload an image from dataset or your test folder
- Predict bacteria name + bacteria type + dominant morphology
- View JSON output directly in the app

## Run tests

```bash
python -m pytest -q
```

## Troubleshooting

- If prediction fails with missing model file, run training first.
- If images are not found, verify `dataset_full.csv` paths remain under `Bacteria dataset/`.
- For better accuracy, add more labeled spiral samples and retrain.
