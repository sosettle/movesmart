# movesmart

```
movesmart/
│
├── data/
│   ├── raw/                # raw downloaded data
│   └── processed/          # master cleaned dataset
│
├── notebooks/
│   ├── 01_data.ipynb       # data cleaning & preprocessing
│   ├── 02_models.ipynb     # similarity model / clustering
│   └── 03_eda.ipynb        # exploratory data analysis
│
├── src/
│   ├── recommender.py      # recommendation logic
│   ├── clustering.py       # clustering logic
│   └── utils.py            # helper functions
│
├── models/                 # saved models / indexes
│
├── app.py                  # Streamlit app
│
├── requirements.txt        # project dependencies
│
└── README.md               # project documentation
```