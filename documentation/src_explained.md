# `src` Folder Explanation

This document explains what the code inside the `src/bacteria_assistant` package does and how the pipeline works end‑to‑end.

## Package entry point: `__init__.py`

**Purpose:** Provide clean import access for the package.

**How it works:**
- Exposes `train_models` and `predict_bacteria_image`.
- Allows other scripts (CLI, UI, tests) to do:
  - `from bacteria_assistant import train_models`
  - `from bacteria_assistant import predict_bacteria_image`

---

## Configuration: `config.py`

**Purpose:** Central source of truth for taxonomy and artifact paths.

**What it defines:**
- `ORGANISM_METADATA`: label metadata (organism type, gram label, morphology shape, taxonomy group).
- `SUPPORTED_SHAPES`: valid shape outputs.
- `ORGANISMS_BY_GROUP`: constraints for group‑specific species prediction.
- `MODEL_PATH`: where the trained model bundle is stored.

**How it’s used:**
- Training uses metadata to label dataset rows.
- Inference uses metadata to override model predictions when possible.

---

## Feature extraction: `features.py`

**Purpose:** Convert images into numeric features for ML training and prediction.

### 1) Global image features (used for species/gram/taxonomy)
**Function:** `extract_image_features(image)`

Produces:
- RGB / HSV mean and std
- Grayscale mean and std
- Laplacian variance (sharpness)
- Edge density (Canny)
- Color histograms (8 bins × 3 channels)
- Compact spatial signature (24×24 downsampled grayscale)

### 2) Colony segmentation & morphology features
**Functions:**
- `extract_colonies(image)`: thresholds image and finds colony contours.
- `_contour_to_measurement(contour, gray)`: geometric measurements.
- `colony_to_feature_dict(colony)`: numeric feature dict for ML.

Each colony measurement includes:
- Area, perimeter, circularity
- Aspect ratio, solidity
- Equivalent diameter
- Mean intensity

### 3) Distribution analysis
**Function:** `detect_distribution(colonies, image_shape)`

Classifies colony layout as:
- `isolated`, `clustered`, `mixed`, or `dispersed`

---

## Training pipeline: `training.py`

**Purpose:** Train the full hierarchical model bundle and save it.

**Main entry:** `train_models(dataset_csv, workspace_root, ...)`

### Steps
1. **Load and clean labels**
   - `_load_labeled_dataframe` maps dataset rows to taxonomy labels.

2. **Build image feature table**
   - `_build_image_feature_table` extracts global features per image.

3. **Train top‑level classifiers** (auto‑select best model)
   - Organism type (bacteria vs fungi)
   - Gram type (gram+ vs gram−)
   - Taxonomy group (e.g., gram+ cocci)
   - Species (all classes)

4. **Train group‑specific species models**
   - `_train_group_species_models` trains separate models per taxonomy group.
   - Improves accuracy by constraining class sets.

5. **Train colony morphology model**
   - `_build_colony_feature_table` builds colony features.
   - Trains a shape classifier (cocci/bacilli/fungal/spiral).

6. **Save artifacts**
   - Models + feature columns + metadata → `artifacts/bacteria_models.joblib`
   - Metrics also saved for reporting.

---

## Inference pipeline: `inference.py`

**Purpose:** Run predictions on a single image using trained models.

**Main entry:** `predict_bacteria_image(image_path, model_path=None, mode="basic")`

### Steps
1. **Load model bundle** with `load_models()`.
2. **Extract global features** from the image.
3. **Predict organism type** (bacteria vs fungi).
4. **Predict taxonomy group** (gram+/gram− subgroup).
5. **Predict species**
   - Uses group‑specific model if available.
   - Falls back to global species model otherwise.
6. **Predict gram label** (only for bacteria).
7. **Extract colonies** and predict shapes for each colony.
8. **Aggregate morphology**
   - Dominant shape + distribution + confidence.

### Output modes
- `mode="basic"`: minimal UI‑friendly output.
- `mode="advanced"`: full colony detail + morphology summary.

---

## End‑to‑end flow summary

1. **Train**
   - CSV → features → ML models → artifact bundle.

2. **Predict**
   - Image → features → organism + gram + morphology.

This lets the UI and CLI give structured outputs without re‑training each time.
