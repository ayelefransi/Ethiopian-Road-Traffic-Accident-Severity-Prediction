import React from 'react';
import { Brain, Database, Shield, Layers, GitBranch, Award } from 'lucide-react';
import './About.css';

const ALGORITHMS = [
  { num: '#2',  name: 'Logistic Regression',              role: 'Meta-Learner',  desc: 'L2 regularization (C=2.0). Learns optimal weights to combine base learner predictions.' },
  { num: '#7',  name: 'CART Decision Tree',               role: 'Base Learner 1',desc: 'Classification and Regression Tree. Deterministic splits; provides interpretable decision rules.' },
  { num: '#19', name: 'Bagging',                          role: 'Base Learner 3',desc: 'Bootstrap Aggregating over CART. max_samples=0.9, max_features=0.9 for variance reduction.' },
  { num: '#21', name: 'Extra Trees',                      role: 'Base Learner 2',desc: 'Extremely Randomized Trees. 150 estimators. Best individual CV at 90.02%.' },
  { num: '#23', name: 'Stacking',                         role: 'Ensemble',      desc: 'Combines CART + Extra Trees + Bagging via 3-fold internal CV. passthrough=True.' },
];

const PIPELINE_STEPS = [
  { step: '01', title: 'Data Ingestion',     desc: 'Merge Mendeley (12,316 rows) and Figshare (13,064 rows) datasets. Drop Source and Year metadata columns.' },
  { step: '02', title: 'Preprocessing',      desc: 'OrdinalEncoder for 21 categorical features. StandardScaler for 2 numerical features. Both inside ColumnTransformer.' },
  { step: '03', title: 'SMOTE',              desc: 'Synthetic Minority Oversampling applied only on training data inside the pipeline to prevent data leakage.' },
  { step: '04', title: 'Stacking Ensemble',  desc: 'Three base learners trained via 3-fold internal CV. Predictions fed to Logistic Regression meta-learner with passthrough.' },
  { step: '05', title: 'Evaluation',         desc: 'Stratified 3-fold CV for model selection. Single held-out test set (5,076 records) for final performance reporting.' },
  { step: '06', title: 'Deployment',         desc: 'FastAPI backend exposes /predict endpoint. React frontend provides interactive form and live dashboard.' },
];

const REGULARIZATION = [
  { component: 'CART',         technique: 'class_weight="balanced"',    effect: 'Upweights minority class loss during tree construction' },
  { component: 'Extra Trees',  technique: 'n_estimators=150, averaging', effect: 'Variance reduction by sqrt(150) across all trees' },
  { component: 'Bagging',      technique: 'max_samples=0.9',             effect: 'Each tree sees 90% of training data independently' },
  { component: 'Bagging',      technique: 'max_features=0.9',            effect: 'Each tree uses 90% of features — forces diverse splits' },
  { component: 'LR Meta',      technique: 'L2 penalty, C=2.0',           effect: 'Shrinks meta-weights toward zero, prevents over-reliance on one model' },
  { component: 'All',          technique: 'SMOTE, k=5',                  effect: 'Synthesizes minority class samples to prevent imbalance overfitting' },
];

export default function About() {
  return (
    <div className="about-page">
      <div className="about-inner">

        {/* Header */}
        <div className="about-hero">
          <div className="about-hero__badge">
            <Brain size={14} /> ML Engineering Documentation
          </div>
          <h1 className="about-hero__title">How This System Works</h1>
          <p className="about-hero__sub">
            A complete technical breakdown of the stacking ensemble, preprocessing pipeline,
            regularization strategy, and deployment architecture.
          </p>
        </div>

        {/* Algorithms */}
        <section className="about-section">
          <div className="section-label">
            <Layers size={14} /> ALGORITHM MAP
          </div>
          <h2 className="about-section__title">Algorithms from the Approved List</h2>
          <p className="about-section__desc">
            Every algorithm used in this system comes from the approved classical ML list.
            No deep learning, no XGBoost, no LightGBM.
          </p>
          <div className="algo-grid">
            {ALGORITHMS.map(({ num, name, role, desc }) => (
              <div key={num} className="algo-card">
                <div className="algo-card__num">{num}</div>
                <div className="algo-card__content">
                  <div className="algo-card__role">{role}</div>
                  <div className="algo-card__name">{name}</div>
                  <div className="algo-card__desc">{desc}</div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Pipeline */}
        <section className="about-section">
          <div className="section-label">
            <GitBranch size={14} /> PIPELINE
          </div>
          <h2 className="about-section__title">End-to-End ML Pipeline</h2>
          <p className="about-section__desc">
            Six sequential stages from raw CSV data to deployed FastAPI prediction endpoint.
          </p>
          <div className="pipeline-steps">
            {PIPELINE_STEPS.map(({ step, title, desc }, i) => (
              <div key={step} className="pipeline-step">
                <div className="pipeline-step__connector">
                  <div className="pipeline-step__num">{step}</div>
                  {i < PIPELINE_STEPS.length - 1 && <div className="pipeline-step__line" />}
                </div>
                <div className="pipeline-step__content">
                  <div className="pipeline-step__title">{title}</div>
                  <div className="pipeline-step__desc">{desc}</div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Regularization */}
        <section className="about-section">
          <div className="section-label">
            <Shield size={14} /> REGULARIZATION
          </div>
          <h2 className="about-section__title">Regularization Strategy</h2>
          <p className="about-section__desc">
            Six regularization techniques applied across different pipeline components
            to prevent overfitting on imbalanced data.
          </p>
          <div className="reg-table">
            <div className="reg-table__head">
              <span>Component</span>
              <span>Technique</span>
              <span>Effect</span>
            </div>
            {REGULARIZATION.map(({ component, technique, effect }, i) => (
              <div key={i} className="reg-table__row">
                <span className="reg-table__component">{component}</span>
                <span className="reg-table__technique">{technique}</span>
                <span className="reg-table__effect">{effect}</span>
              </div>
            ))}
          </div>
        </section>

        {/* Results */}
        <section className="about-section">
          <div className="section-label">
            <Award size={14} /> RESULTS
          </div>
          <h2 className="about-section__title">Final Performance</h2>
          <div className="results-grid">
            {[
              { label: 'Test Accuracy',    value: '90.82%', color: '#38bdf8', note: 'Exceeds 90% target' },
              { label: 'ROC-AUC (OvR)',   value: '0.9822',  color: '#8b5cf6', note: 'Near-perfect discrimination' },
              { label: 'F1 (Weighted)',   value: '90.91%',  color: '#22c55e', note: 'Strong across all classes' },
              { label: 'F1 (Macro)',      value: '88.25%',  color: '#f59e0b', note: 'Even per-class performance' },
              { label: 'Precision (Wtd)','value': '91.14%', color: '#ef4444', note: 'High prediction confidence' },
              { label: 'Recall (Wtd)',    value: '90.82%',  color: '#06b6d4', note: 'Captures most true cases' },
            ].map(({ label, value, color, note }) => (
              <div key={label} className="result-tile" style={{ '--rt': color }}>
                <div className="result-tile__value">{value}</div>
                <div className="result-tile__label">{label}</div>
                <div className="result-tile__note">{note}</div>
              </div>
            ))}
          </div>
        </section>

        {/* Dataset */}
        <section className="about-section">
          <div className="section-label">
            <Database size={14} /> DATASET
          </div>
          <h2 className="about-section__title">Dataset Overview</h2>
          <div className="dataset-cards">
            <div className="dataset-card">
              <div className="dataset-card__num">12,316</div>
              <div className="dataset-card__name">Mendeley</div>
              <div className="dataset-card__info">Addis Ababa Sub-city Police · 2017-2020 · CC BY 4.0</div>
            </div>
            <div className="dataset-card__plus">+</div>
            <div className="dataset-card">
              <div className="dataset-card__num">13,064</div>
              <div className="dataset-card__name">Figshare</div>
              <div className="dataset-card__info">Addis Ababa Police Dept · 2016-2022 · CC BY 4.0</div>
            </div>
            <div className="dataset-card__plus">=</div>
            <div className="dataset-card dataset-card--total">
              <div className="dataset-card__num">25,380</div>
              <div className="dataset-card__name">Combined</div>
              <div className="dataset-card__info">23 features · 3 severity classes · 0 missing values</div>
            </div>
          </div>
        </section>

      </div>
    </div>
  );
}
