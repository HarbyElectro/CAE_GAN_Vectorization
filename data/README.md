# Data Directory

This folder is reserved for local datasets used by the models in this repository.

Large datasets are **not included** in this GitHub repository because of file size limits and dataset licensing restrictions.

Users should download the datasets from their official sources and place them in this folder using the expected structure below.

---

## Expected Data Structure

```bash
data/
│
├── celeba/
│   └── img_align_celeba/
│       ├── image_000001.jpg
│       ├── image_000002.jpg
│       └── ...
│
├── cfpd/
│   └── Data/
│       └── Images/
│           ├── person_001/
│           ├── person_002/
│           └── ...
│
├── imdb_faces/
│   ├── class_1/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── class_2/
│       ├── img003.jpg
│       └── img004.jpg
│
├── flowers/
│   ├── class_1/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── class_2/
│       ├── img003.jpg
│       └── img004.jpg
│
├── mnist/
│
└── UCF101/
    ├── ApplyEyeMakeup/
    ├── Archery/
    ├── BabyCrawling/
    └── ...
