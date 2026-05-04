# final architecture
```md
[1] Input Image
      ↓
[2] Preprocessing
      ↓
[3] Global Feature Extraction   (image-level)
      ↓
[4] ML Hierarchy Inference
      ↓
      ├── organism_type
      ├── taxonomy_group
      ├── species_prediction
      └── gram_prediction

      ↓
[5] Colony Segmentation
      ↓
[6] Colony Feature Extraction
      ↓
[7] Morphology Classification (per colony)
      ↓
[8] Morphology Aggregation (image-level)
      ↓
[9] CONSISTENCY CHECK (NEW — inside current system)
      ↓
[10] Output Builder
```
------------------------------------------------
##  1) Global Feature Extraction   (image-level)
Global Feature Extraction means everything we compute from the entire image (not individual colonies) to help the model recognize species and gram type. Global Feature Extraction = converting the WHOLE IMAGE into a numerical vector that represents color, texture, structure, and overall patterns

## Global Features (Image-Level)
### 1. Color Statistics
These capture stain tone and color distribution.

- RGB mean + std
      - Average red, green, blue intensity
- HSV mean + std
      - Hue, saturation, brightness patterns
- Gray mean + std
      - Overall brightness + contrast

### 2. Texture / Sharpness
These capture how “rough” or “granular” the image appears.

- Laplacian variance
      - High value → sharper edges / more texture
- Edge density (Canny)
      - Fraction of pixels that are edges
### 3. Color Histograms
These describe color distribution shape.

- 8‑bin histogram for each RGB channel
(24 values total)
### 4. Spatial Signature (Morphology Layout)
This is the strongest discriminator we added.

- Downsample image to 24×24 grayscale
- Flatten into 576 numeric features
- Encodes rough colony arrangement and spatial texture

They help the model decide:

- Which organism group it belongs to
- Which species it resembles
- Gram/fungi differences

They are fast, lightweight, and work well with classical ML models.
```md
feature_vector = [
    rgb_mean,
    rgb_std,
    hsv_mean,
    hsv_std,
    gray_mean,
    gray_std,
    laplacian,
    edge_density,
    histogram_bins,
    spatial_signature (576 values)
]
```
------------------------------------------------
## 2) ML Hierarchy 
Overview

The system uses a hierarchical set of machine‑learning classifiers to reduce biologically invalid predictions and improve accuracy. Each stage narrows the decision space before the next stage runs.

### 1) Organism Type Classifier
Task: bacteria vs fungi
Purpose: Ensures fungi are not forced into Gram‑positive/Gram‑negative bacteria categories.

### 2) Taxonomy Group Classifier
Task:

- gram_positive_cocci
- gram_positive_bacilli
- gram_negative_bacilli
- fungi

Purpose: Constrains species prediction to the correct biological group.

### 3) Group‑Specific Species Classifier
- Task: Predict species within the chosen group only
- Purpose: Reduces cross‑group errors (e.g., predicting a Gram‑negative species inside Gram‑positive group).

### 4) Global Species Fallback
- Task: Predict among all 10 organisms
- Purpose: Used when a group‑specific model is unavailable or low‑confidence.

### 5) Gram Classifier (Bacteria Only)
- Task: gram_positive vs gram_negative
- Purpose: Adds Gram label to output and validates bacterial predictions.

| Benefit          | Why                    |
| ---------------- | ---------------------- |
| Better accuracy  | smaller decision space |
| Biological logic | matches real taxonomy  |
| Easier debugging | errors localized       |

------------------------------------------------

## 3)Colony Segmentation 
Purpose
Colony segmentation isolates individual bacterial colonies from the background so that morphology features (shape, size, texture) can be measured per colony.

### Pipeline Steps
### Grayscale conversion
- Converts the input image to grayscale to simplify intensity‑based segmentation.

### Gaussian blur
- Reduces noise and smooths small pixel variations.

### Otsu thresholding
- Automatically separates foreground (colonies) from background using a global threshold.

### Binary inversion (if needed)
- Ensures colonies are treated as foreground (white) for contour detection.

### Morphological opening & closing

- Opening: removes tiny noise pixels
- Closing: fills small holes inside colonies
- Contour extraction
### Detects connected components (potential colonies).

### Area filtering
- Removes objects that are too small or too large to be real colonies.

#### Output
- A list of valid colony contours used to compute colony morphology features such as:
```md
Area
Perimeter
Circularity
Aspect ratio
Solidity
Equivalent diameter
Mean intensity
```

------------------------------------------------

## 4) Colony Feature Extraction (Deep Explanation)

### Purpose
After segmentation, each detected colony is treated as an individual object.  
We extract **geometric and intensity features** so the model can describe colony morphology and classify shapes (cocci/bacilli/fungal).

---

### 1) Contour → Geometric Core
Each colony is represented by its contour points. From this, we compute:

- **Area (A)**  
  Pixel area inside the contour.  
  *Indicates colony size.*

- **Perimeter (P)**  
  Total contour length.  
  *Captures boundary complexity.*

- **Equivalent Diameter**  
  Diameter of a circle with the same area.  
  Formula:  
  `D = sqrt(4A / π)`  
  *Normalizes size into one comparable value.*

---

### 2) Shape Descriptors
These measure how round, elongated, or compact the colony is:

- **Circularity**  
  `4πA / P²`  
  - Close to 1 → perfect circle (cocci)  
  - Lower values → elongated/irregular (bacilli/fungal)

- **Aspect Ratio**  
  Width / Height of the bounding rectangle.  
  - ≈1 → round  
  - >1.5 → elongated/rod-like

- **Solidity**  
  `Area / ConvexHullArea`  
  - 1.0 → solid, compact  
  - Lower → rough or irregular boundary

---

### 3) Intensity Feature
Computed over pixels inside the contour:

- **Mean Intensity**  
  Average grayscale value of the colony region.  
  *Helps detect staining density or colony thickness.*

---

### 4) Why these features matter
These values are fed into the **colony morphology classifier** to decide:

- `cocci` → high circularity + low aspect ratio  
- `bacilli` → low circularity + high aspect ratio  
- `fungal` → irregular, low solidity patterns

---

### 5) Feature Vector (Per Colony)
Each colony becomes a numeric vector:

```
[area,
 perimeter,
 circularity,
 aspect_ratio,
 solidity,
 equivalent_diameter,
 mean_intensity]
```

---

### 6) Output Usage
These values are:
- **Aggregated** to compute dominant morphology  
- **Included** in advanced JSON output  
- **Used** to support morphology inference confidence

------------------------------------------------

## 5) Deep Dive: Morphology Classification Logic (Per Colony)

### Purpose
This step converts **per‑colony feature vectors** into a discrete **shape class**:
- `cocci`
- `bacilli`
- `fungal`

It is used for:
- **Per‑colony predictions** (advanced output)
- **Image‑level morphology aggregation**
- **Dominant shape decision**

---

### 1) Input Feature Vector (Per Colony)
Each colony is represented by:

```
[area,
 perimeter,
 circularity,
 aspect_ratio,
 solidity,
 equivalent_diameter,
 mean_intensity]
```

---

### 2) Classification Strategy (Current System)
The classifier is trained on labeled colony features derived from the dataset:
- `cocci` → round, high circularity, low aspect ratio
- `bacilli` → elongated, low circularity, high aspect ratio
- `fungal` → irregular edges, lower solidity, inconsistent shape

The model learns these patterns statistically rather than using fixed thresholds.

---

### 3) Learned Decision Patterns (Empirical)
Although the model is data‑driven, the following patterns dominate:

| Feature Pattern | Likely Class | Reason |
|---------------|--------------|--------|
| circularity ≈ 0.8–1.0 and aspect_ratio ≈ 1.0 | cocci | round colonies |
| circularity < 0.5 and aspect_ratio > 1.5 | bacilli | rod‑like colonies |
| solidity < 0.7 and irregular perimeter | fungal | hyphae/irregular edges |

---

### 4) Confidence Scoring
For each colony prediction, probability is extracted from the classifier.

- `predicted_shape` = class with highest probability
- `confidence` = max(class_probabilities)

These are later aggregated at image‑level.

---

### 5) Output (Per Colony)
Each colony produces:

```
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
```

---

### 6) Role in Final Output
Morphology predictions are used to compute:
- `dominant_shape`
- `distribution` (clustered vs scattered)
- `final_morphology.confidence`

```md 
                circularity ↑
                     |
     cocci           |        (high circularity)
                     |
                     |
------------------------------------→ aspect_ratio
                     |
                     |
                     |
   irregular         |      bacilli
 (low circularity)   |   (high aspect_ratio)
```

----------------------------------------

## 6) Deep Dive: Morphology Aggregation (Image‑Level)

### Purpose
This step converts **many per‑colony predictions** into a **single, reliable morphology summary** for the whole image.

It produces:
- `dominant_shape`
- `distribution`
- `final_morphology.confidence`

---

### 1) Inputs
From colony classification:
- `predicted_shape` for each colony
- optional per‑colony probabilities (if available)
- colony centroids (x, y) for spatial distribution

---

### 2) Dominant Shape Calculation
We count predicted shapes:

```
count_cocci, count_bacilli, count_fungal
```

Dominant shape is the **max count**.

---

### 3) Dominance Ratio
This measures how strong the dominant class is:

```
dominance_ratio = max(counts) / total_colonies
```

Examples:
- 0.85 → very consistent morphology
- 0.50 → mixed morphology

---

### 4) Morphology Confidence
Combined from:
- **dominance_ratio**
- **average per‑colony probability** (if available)

One simple blend:

```
confidence = 0.5 * dominance_ratio + 0.5 * mean_colony_confidence
```

If probabilities are unavailable:

```
confidence = dominance_ratio
```

---

### 5) Distribution (Clustered vs Scattered)
Uses colony centroids and nearest‑neighbor distances.

Steps:
1. Compute each colony’s nearest neighbor distance.
2. Find mean NN distance.
3. Compare to image size:

```
if mean_nn_distance < 0.03 * max(image_width, image_height):
    distribution = "clustered"
else:
    distribution = "scattered"
```

This produces the `final_morphology.distribution` field.

---

### 6) Final Output Block
This is appended to advanced JSON:

```
"final_morphology": {
  "dominant_shape": "cocci",
  "distribution": "clustered",
  "confidence": 0.87
}
```

---

### Why this matters
Aggregation stabilizes predictions by:
- reducing colony‑level noise
- giving a single image‑level interpretation
- capturing both shape consistency and spatial arrangement

--------------------------------------------------


## 7) Deep Dive: Consistency Check & Final Decision

### Purpose
This step **verifies biological and logical consistency** across predictions before the final output is built.  
It prevents impossible or conflicting results (e.g., fungi with Gram type).

---

### 1) Inputs Used
- `organism_type` (bacteria / fungi)
- `taxonomy_group`
- `predicted_bacteria_name`
- `bacteria_type` (gram_positive / gram_negative / non_bacterial_fungi)
- `dominant_shape`
- `confidence`
- optional per‑colony shape statistics

---

### 2) Core Consistency Rules

#### Rule A: Fungi override
If `organism_type = fungi`:
- Force `bacteria_type = non_bacterial_fungi`
- Force `dominant_shape = fungal`
- Species must be in fungi list

#### Rule B: Gram vs group coherence
If `taxonomy_group` implies Gram:
- `gram_positive_cocci` → `bacteria_type = gram_positive`
- `gram_positive_bacilli` → `bacteria_type = gram_positive`
- `gram_negative_bacilli` → `bacteria_type = gram_negative`

If mismatch is detected:
- prefer taxonomy group and override `bacteria_type`

#### Rule C: Species vs group coherence
If predicted species is **not in the selected group**:
- fallback to group‑specific model prediction (if available)
- else fallback to global species model
- if still mismatch → mark as `low_confidence`

#### Rule D: Shape vs group sanity check
If `dominant_shape` contradicts group:
- `gram_positive_cocci` must not end as `bacilli`
- `gram_negative_bacilli` must not end as `cocci`
If mismatch:
- reduce confidence
- optionally flag for review

---

### 3) Confidence Adjustment
Final confidence is adjusted if inconsistencies occur.

Example:
```
if conflict_detected:
    confidence = confidence * 0.75
```

---

### 4) Final Decision Logic
After checks:

1. Fix invalid fields (override inconsistent fields)
2. Down‑weight confidence if needed
3. Build final JSON output

---

### 5) Output Impact
The consistency check ensures:
- biologically valid labels
- stable final predictions
- safer output for lab use