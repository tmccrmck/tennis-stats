# ATP Tennis Analytics & Prediction Suite

A professional-grade tennis match prediction system and analytical toolkit powered by machine learning and 6 years of ATP match data (2019–2024).

## 🚀 Features
- **3-Model Weighted Ensemble**: Combines **Logistic Regression**, **XGBoost (Calibrated)**, and a custom **PyTorch Neural Network** for a consensus accuracy of ~66%.
- **Advanced Feature Engineering**: Incorporates surface-specific Elo, 14-day rolling fatigue, stylistic hold/break metrics, and "clutch" performance (BP conversion/saving).
- **Deep Analytics**: Non-linear stylistic mapping (UMAP), Tactical DNA analysis, and the "Iceman" mental strength index.
- **Modern Tech Stack**: Fully managed by `uv` for lightning-fast dependency handling and Python execution.

## 🛠 Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

1. **Install uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Run Training**:
   This will process the data, engineer features, and train all three models.
   ```bash
   uv run --with pandas --with numpy --with scikit-learn --with xgboost --with joblib --with torch python3 main.py
   ```

## 📈 Running Predictions

You can forecast upcoming matches using the `Predictor` class. We provide a sample script for quick testing:

```bash
uv run --with pandas --with numpy --with scikit-learn --with xgboost --with joblib --with torch python3 predict_match.py
```

### Example Usage:
```python
from prediction import Predictor

predictor = Predictor()
prob = predictor.predict("Jannik Sinner", "Novak Djokovic", surface="Hard", level="G")
print(f"Sinner Win Probability: {prob:.2%}")
```

## 📊 Analytics Suite

The `scripts/` folder contains a comprehensive engine for generating high-density visualizations. All scripts use `adjustText` for clean, readable labels.

| Script | Insight |
| :--- | :--- |
| `style_umap.py` | 2D Manifold of player "DNA" using non-linear UMAP. |
| `iceman_index.py` | Maps mental strength (BP Saved % vs BP Converted %). |
| `court_speed.py` | Ranks ATP tournaments by their statistical pace (SSI). |
| `tactical_dna_pca.py` | Clusters players by shot placement preference (Pattern vs. Chaos). |
| `short_player_compensation.py` | Reveals the "Survival Threshold" for shorter players. |

**Run any analytics script:**
```bash
uv run --with pandas --with matplotlib --with seaborn --with scikit-learn --with umap-learn --with adjustText python3 scripts/style_umap.py
```

## 📝 Technical Documentation
For a deep dive into the physics of player physique and style, see our technical blog post:
👉 [The 190cm Sweet Spot: Physics of Tennis](PHYSICS_OF_TENNIS_BLOG.md)

---
*Data sourced from Jeff Sackmann's `tennis_atp` and the Match Charting Project.*
