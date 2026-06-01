# Ethiopian RTA Severity Predictor — Full Stack Web App

A production-grade full stack application that predicts road traffic accident
severity in Addis Ababa, Ethiopia using a Stacking Ensemble ML model.

---

## Tech Stack

| Layer     | Technology                        |
|-----------|-----------------------------------|
| Frontend  | React 18, Recharts, Framer Motion |
| Backend   | FastAPI, Python 3.9+              |
| ML Model  | sklearn Stacking Ensemble         |
| Styling   | Custom CSS with design system     |

---

## Model Performance

| Metric            | Score  |
|-------------------|--------|
| Test Accuracy     | 90.82% |
| ROC-AUC           | 0.9822 |
| F1 (Weighted)     | 90.91% |
| F1 (Macro)        | 88.25% |
| Precision (Weight)| 91.14% |
| Recall (Weighted) | 90.82% |

---

## Quick Start

### Linux / macOS

```bash
chmod +x start.sh
./start.sh
```

### Windows

**Using Batch:**
```
Double-click start.bat
```

**Using PowerShell:**
```powershell
.\start.ps1
```

### Manual Start

**Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm start
```

---

## Project Structure

```
rta_webapp/
├── backend/
│   ├── main.py              # FastAPI app
│   ├── requirements.txt
│   └── model/
│       ├── best_rta_model.pkl
│       └── model_metadata.json
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js
│   │   ├── index.js
│   │   ├── index.css        # Global design system
│   │   ├── components/
│   │   │   ├── Navbar.js
│   │   │   └── Navbar.css
│   │   └── pages/
│   │       ├── Home.js      # Landing page with metrics
│   │       ├── Predict.js   # Interactive prediction form
│   │       ├── Dashboard.js # Charts and model analytics
│   │       └── About.js     # Technical documentation
│   └── package.json
├── start.sh                 # Linux/macOS launcher
├── start.bat                # Windows launcher
├── start.ps1                # PowerShell launcher
└── README.md
```

---

## API Endpoints

| Method | Endpoint         | Description                        |
|--------|------------------|------------------------------------|
| GET    | /                | API info                           |
| GET    | /health          | Model performance stats            |
| GET    | /features        | Valid values for all features      |
| GET    | /stats           | Dataset and algorithm summary      |
| POST   | /predict         | Single accident severity prediction|

### Example Request

```json
POST /predict
{
  "Time": "Night (22-6)",
  "Day_of_week": "Friday",
  "Age_band_of_driver": "18-30",
  "Sex_of_driver": "Male",
  "Educational_level": "High school",
  "Vehicle_driver_relation": "Employee",
  "Driving_experience": "No Licence",
  "Lanes_or_Medians": "Undivided Two way",
  "Types_of_Junction": "Y Shape",
  "Road_surface_type": "Asphalt roads",
  "Road_surface_conditions": "Wet or damp",
  "Light_conditions": "Darkness - no lighting",
  "Weather_conditions": "Raining and Windy",
  "Type_of_collision": "Rollover",
  "Number_of_vehicles_involved": 4,
  "Number_of_casualties": 5,
  "Vehicle_type": "Long lorry",
  "Cause_of_accident": "Drunk driving",
  "Pedestrian_movement": "Crossing from nearside",
  "Vehicle_movement": "Turnover",
  "Road_allignment": "Steep grade downward with mountainous terrain",
  "Area_accident_occured": "Outside Addis Ababa",
  "Sub_district": "Akaki Kaliti"
}
```

### Example Response

```json
{
  "predicted_class": 2,
  "predicted_severity": "Fatal injury",
  "confidence": 1.0,
  "probabilities": {
    "Slight Injury": 0.0,
    "Serious Injury": 0.0,
    "Fatal injury": 1.0
  },
  "risk_level": "HIGH",
  "risk_factors": [
    "Night driving",
    "Poor lighting",
    "Drunk driving",
    "Inexperienced driver",
    "Rollover collision",
    "Wet road surface",
    "5 casualties",
    "Outside city limits"
  ]
}
```

---

## App Pages

### Home (`/`)
Landing page with animated metric counters, feature highlights,
severity class breakdown, and CTA.

### Predict (`/predict`)
Interactive 23-feature form organized in 5 sections.
Dropdown menus for all categorical features.
Stepper inputs for numerical features.
Live prediction result with probability bars, risk factor detection,
and confidence score.

### Dashboard (`/dashboard`)
Full analytics dashboard with:
- Animated metric stat cards
- Severity distribution pie chart
- Algorithm comparison bar chart
- Performance radar chart
- Per-class metrics grouped bar chart
- Feature importance horizontal bar chart
- Regularization analysis chart
- Dataset summary table

### About (`/about`)
Technical documentation covering:
- Algorithm map with list numbers
- 6-step pipeline walkthrough
- Regularization strategy table
- Final performance results
- Dataset source details

---

## Requirements

- Python 3.9+
- Node.js 18+
- 2 GB RAM minimum (model is 277 MB)

---

## Algorithms Used (from Approved List)

| List # | Algorithm              | Role           |
|--------|------------------------|----------------|
| #2     | Logistic Regression    | Meta-Learner   |
| #7     | CART Decision Tree     | Base Learner 1 |
| #19    | Bagging                | Base Learner 3 |
| #21    | Extra Trees            | Base Learner 2 |
| #23    | Stacking               | Ensemble       |

---

*Dataset: Mendeley DOI 10.17632/xytv86278f.2 | License: CC BY 4.0*
