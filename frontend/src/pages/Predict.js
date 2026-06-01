import React, { useState, useRef } from 'react';
import axios from 'axios';
import {
  AlertTriangle, CheckCircle, XCircle, ChevronDown,
  Loader, Send, RotateCcw, MapPin, Car, Cloud,
  Clock, User, Road
} from 'lucide-react';
import './Predict.css';

const CAT_OPTIONS = {
  Time: ['Afternoon (14-18)', 'Night (22-6)'],
  Day_of_week: ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
  Age_band_of_driver: ['18-30','31-50'],
  Sex_of_driver: ['Male','Female'],
  Educational_level: ['Elementary school','High school','Above high school','Unknown'],
  Vehicle_driver_relation: ['Employee','Owner','Other'],
  Driving_experience: ['2-5yr','Below 1yr','No Licence'],
  Lanes_or_Medians: ['Undivided Two way','Two-way (divided with broken lines road marking)','Double carriageway (median)','One way'],
  Types_of_Junction: ['No junction','Y Shape','T Shape','Crossing','Unknown'],
  Road_surface_type: ['Asphalt roads','Asphalt roads with some distress','Earth roads','Gravel roads'],
  Road_surface_conditions: ['Dry','Wet or damp'],
  Light_conditions: ['Daylight','Darkness - no lighting'],
  Weather_conditions: ['Normal','Raining','Raining and Windy','Cloudy'],
  Type_of_collision: ['Vehicle with vehicle collision','Rollover'],
  Vehicle_type: ['Automobile','Taxi','Bajaj','Motorcycle','Public (12 seats)','Lorry (41-100Q)','Long lorry'],
  Cause_of_accident: ['Other','Overspeed','Drunk driving'],
  Pedestrian_movement: ['Not a Pedestrian','Crossing from nearside','Crossing from driver near side','Unknown or other'],
  Vehicle_movement: ['Going straight','Stopping','U-Turn','Moving Backward','Turnover'],
  Road_allignment: ['Tangent road with flat terrain','Steep grade downward with mountainous terrain'],
  Area_accident_occured: ['Residential areas','Outside Addis Ababa'],
  Sub_district: ['Bole','Kirkos','Kolfe Keranio','Lideta','Nifas Silk-Lafto','Yeka','Akaki Kaliti','Addis Ketema','Arada','Gulele'],
};

const DEFAULT = {
  Time: 'Afternoon (14-18)',
  Day_of_week: 'Friday',
  Age_band_of_driver: '18-30',
  Sex_of_driver: 'Male',
  Educational_level: 'High school',
  Vehicle_driver_relation: 'Employee',
  Driving_experience: '2-5yr',
  Lanes_or_Medians: 'Undivided Two way',
  Types_of_Junction: 'No junction',
  Road_surface_type: 'Asphalt roads',
  Road_surface_conditions: 'Dry',
  Light_conditions: 'Daylight',
  Weather_conditions: 'Normal',
  Type_of_collision: 'Vehicle with vehicle collision',
  Number_of_vehicles_involved: 2,
  Number_of_casualties: 1,
  Vehicle_type: 'Automobile',
  Cause_of_accident: 'Other',
  Pedestrian_movement: 'Not a Pedestrian',
  Vehicle_movement: 'Going straight',
  Road_allignment: 'Tangent road with flat terrain',
  Area_accident_occured: 'Residential areas',
  Sub_district: 'Bole',
};

const SECTIONS = [
  {
    id: 'time', title: 'Time & Location', icon: Clock,
    fields: ['Time','Day_of_week','Area_accident_occured','Sub_district'],
  },
  {
    id: 'driver', title: 'Driver Information', icon: User,
    fields: ['Age_band_of_driver','Sex_of_driver','Educational_level','Vehicle_driver_relation','Driving_experience'],
  },
  {
    id: 'vehicle', title: 'Vehicle', icon: Car,
    fields: ['Vehicle_type','Vehicle_movement','Lanes_or_Medians','Types_of_Junction'],
  },
  {
    id: 'road', title: 'Road & Environment', icon: Cloud,
    fields: ['Road_surface_type','Road_surface_conditions','Light_conditions','Weather_conditions','Road_allignment'],
  },
  {
    id: 'accident', title: 'Accident Details', icon: AlertTriangle,
    fields: ['Type_of_collision','Cause_of_accident','Pedestrian_movement','Number_of_vehicles_involved','Number_of_casualties'],
  },
];

const FIELD_LABELS = {
  Time: 'Time of Day',
  Day_of_week: 'Day of Week',
  Age_band_of_driver: 'Driver Age Band',
  Sex_of_driver: 'Driver Sex',
  Educational_level: 'Educational Level',
  Vehicle_driver_relation: 'Driver-Vehicle Relation',
  Driving_experience: 'Driving Experience',
  Lanes_or_Medians: 'Lanes / Medians',
  Types_of_Junction: 'Junction Type',
  Road_surface_type: 'Road Surface Type',
  Road_surface_conditions: 'Road Surface Condition',
  Light_conditions: 'Light Conditions',
  Weather_conditions: 'Weather',
  Type_of_collision: 'Collision Type',
  Number_of_vehicles_involved: 'Vehicles Involved',
  Number_of_casualties: 'Casualties',
  Vehicle_type: 'Vehicle Type',
  Cause_of_accident: 'Cause of Accident',
  Pedestrian_movement: 'Pedestrian Movement',
  Vehicle_movement: 'Vehicle Movement',
  Road_allignment: 'Road Alignment',
  Area_accident_occured: 'Area',
  Sub_district: 'Sub-district',
};

const SEV_CONFIG = {
  'Slight Injury':  { color: '#22c55e', bg: 'rgba(34,197,94,0.08)',  border: 'rgba(34,197,94,0.3)',  icon: CheckCircle,    label: 'LOW RISK' },
  'Serious Injury': { color: '#f59e0b', bg: 'rgba(245,158,11,0.08)', border: 'rgba(245,158,11,0.3)', icon: AlertTriangle,  label: 'MEDIUM RISK' },
  'Fatal injury':   { color: '#ef4444', bg: 'rgba(239,68,68,0.08)',  border: 'rgba(239,68,68,0.3)',  icon: XCircle,        label: 'HIGH RISK' },
};

function SelectField({ name, value, onChange }) {
  const opts = CAT_OPTIONS[name] || [];
  return (
    <div className="field">
      <label className="field__label">{FIELD_LABELS[name] || name}</label>
      <div className="field__select-wrap">
        <select
          className="field__select"
          value={value}
          onChange={e => onChange(name, e.target.value)}
        >
          {opts.map(o => <option key={o} value={o}>{o}</option>)}
        </select>
        <ChevronDown size={14} className="field__chevron" />
      </div>
    </div>
  );
}

function NumberField({ name, value, onChange, min, max }) {
  return (
    <div className="field">
      <label className="field__label">{FIELD_LABELS[name] || name}</label>
      <div className="field__number-wrap">
        <button className="field__num-btn" onClick={() => onChange(name, Math.max(min, value - 1))}>−</button>
        <input
          type="number"
          className="field__number"
          value={value}
          min={min} max={max}
          onChange={e => onChange(name, Math.max(min, Math.min(max, parseInt(e.target.value) || min)))}
        />
        <button className="field__num-btn" onClick={() => onChange(name, Math.min(max, value + 1))}>+</button>
      </div>
    </div>
  );
}

function ProbBar({ label, value, color }) {
  const pct = (value * 100).toFixed(1);
  return (
    <div className="prob-bar">
      <div className="prob-bar__header">
        <span className="prob-bar__label">{label}</span>
        <span className="prob-bar__pct" style={{ color }}>{pct}%</span>
      </div>
      <div className="prob-bar__track">
        <div
          className="prob-bar__fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}

export default function Predict() {
  const [form, setForm]       = useState(DEFAULT);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);
  const resultRef             = useRef(null);

  const handleChange = (name, value) => {
    setForm(prev => ({ ...prev, [name]: value }));
    setResult(null);
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    try {
      const { data } = await axios.post('https://fransimengesha-rta-api.hf.space/predict', form);
      setResult(data);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
    } catch (e) {
      setError(e.response?.data?.detail || 'Prediction failed. Make sure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => { setForm(DEFAULT); setResult(null); setError(null); };

  const cfg = result ? SEV_CONFIG[result.predicted_severity] : null;

  return (
    <div className="predict-page">
      <div className="predict-page__inner">

        {/* Header */}
        <div className="predict-header">
          <h1 className="predict-header__title">Accident Severity Predictor</h1>
          <p className="predict-header__sub">
            Fill in the 23 accident features below. The stacking ensemble will classify severity in real time.
          </p>
        </div>

        <div className="predict-layout">
          {/* Form */}
          <div className="predict-form">
            {SECTIONS.map(({ id, title, icon: Icon, fields }) => (
              <div key={id} className="form-section">
                <div className="form-section__header">
                  <div className="form-section__icon"><Icon size={16} /></div>
                  <h3 className="form-section__title">{title}</h3>
                </div>
                <div className="form-section__grid">
                  {fields.map(f => {
                    if (f === 'Number_of_vehicles_involved')
                      return <NumberField key={f} name={f} value={form[f]} onChange={handleChange} min={1} max={5} />;
                    if (f === 'Number_of_casualties')
                      return <NumberField key={f} name={f} value={form[f]} onChange={handleChange} min={1} max={8} />;
                    return <SelectField key={f} name={f} value={form[f]} onChange={handleChange} />;
                  })}
                </div>
              </div>
            ))}

            {/* Actions */}
            <div className="form-actions">
              <button className="btn-predict" onClick={handleSubmit} disabled={loading}>
                {loading ? <><Loader size={18} className="spin" /> Predicting...</> : <><Send size={18} /> Predict Severity</>}
              </button>
              <button className="btn-reset" onClick={handleReset}>
                <RotateCcw size={16} /> Reset
              </button>
            </div>

            {error && (
              <div className="error-banner">
                <AlertTriangle size={16} />
                {error}
              </div>
            )}
          </div>

          {/* Result panel */}
          <div className="result-panel" ref={resultRef}>
            {!result && !loading && (
              <div className="result-empty">
                <div className="result-empty__icon">
                  <MapPin size={32} />
                </div>
                <div className="result-empty__title">No Prediction Yet</div>
                <div className="result-empty__sub">Fill in accident details and click Predict Severity</div>
              </div>
            )}

            {loading && (
              <div className="result-loading">
                <div className="result-loading__spinner" />
                <div className="result-loading__text">Running stacking ensemble...</div>
              </div>
            )}

            {result && cfg && (
              <div className="result-card" style={{ '--rc': cfg.color, '--rc-bg': cfg.bg, '--rc-border': cfg.border }}>
                {/* Severity badge */}
                <div className="result-card__badge">
                  <cfg.icon size={14} style={{ color: cfg.color }} />
                  {cfg.label}
                </div>

                {/* Main result */}
                <div className="result-card__main">
                  <div className="result-card__icon-wrap">
                    <cfg.icon size={40} style={{ color: cfg.color }} />
                  </div>
                  <h2 className="result-card__severity" style={{ color: cfg.color }}>
                    {result.predicted_severity}
                  </h2>
                  <div className="result-card__confidence">
                    Confidence: <strong>{(result.confidence * 100).toFixed(1)}%</strong>
                  </div>
                </div>

                {/* Probabilities */}
                <div className="result-card__probs">
                  <div className="result-card__probs-title">Class Probabilities</div>
                  {Object.entries(result.probabilities).map(([cls, prob]) => (
                    <ProbBar
                      key={cls}
                      label={cls}
                      value={prob}
                      color={SEV_CONFIG[cls]?.color || '#888'}
                    />
                  ))}
                </div>

                {/* Risk factors */}
                {result.risk_factors?.length > 0 && (
                  <div className="result-card__risks">
                    <div className="result-card__risks-title">
                      <AlertTriangle size={14} style={{ color: '#f59e0b' }} />
                      Risk Factors Detected
                    </div>
                    <div className="result-card__risks-list">
                      {result.risk_factors.map(r => (
                        <div key={r} className="risk-tag">
                          <span className="risk-tag__dot" />
                          {r}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Model info */}
                <div className="result-card__model">
                  <div className="result-meta">
                    <span className="result-meta__key">Model</span>
                    <span className="result-meta__val">Stacking Ensemble</span>
                  </div>
                  <div className="result-meta">
                    <span className="result-meta__key">Test Accuracy</span>
                    <span className="result-meta__val">90.82%</span>
                  </div>
                  <div className="result-meta">
                    <span className="result-meta__key">ROC-AUC</span>
                    <span className="result-meta__val">0.9822</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
