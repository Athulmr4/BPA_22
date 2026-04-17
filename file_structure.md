```md
morphology_project/
│
├── data/                          # ALL DATA (never mix with code)
│   ├── raw/                       # Original images (never modify)
│   │   ├── petri_dish/
│   │   ├── microscopic_reference/
│   │   └── rejected/
│
│   ├── processed/                 # Intermediate outputs
│   │   ├── grayscale/
│   │   ├── blurred/
│   │   ├── threshold/
│   │   └── masks/
│
│   ├── colonies/                  # FINAL extracted colonies
│   │   ├── unlabelled/
│   │   ├── cocci/
│   │   ├── bacilli/
│   │   └── spirilla/
│
│   └── metadata/
│       ├── image_log.csv
│       └── annotations.csv
│
├── src/                           # CORE LOGIC (modular code)
│   ├── preprocessing/
│   │   └── preprocess.py
│
│   ├── segmentation/
│   │   └── segment.py
│
│   ├── extraction/
│   │   └── extract.py
│
│   ├── utils/
│   │   ├── io_utils.py
│   │   └── visualization.py
│
│   └── config.py
│
├── pipelines/                     # END-TO-END PIPELINES
│   └── run_pipeline.py
│
├── notebooks/                     # EXPERIMENTATION ONLY
│   └── test_segmentation.ipynb
│
├── output/                        # FINAL RESULTS / VISUALS
│   ├── debug/
│   └── overlays/
│
├── requirements.txt
└── README.md
```