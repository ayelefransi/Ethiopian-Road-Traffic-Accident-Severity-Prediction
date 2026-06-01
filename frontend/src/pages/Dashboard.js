import React, { useEffect, useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  AreaChart, Area, Legend
} from 'recharts';
import './Dashboard.css';

const COLORS_SEV  = ['#22c55e','#f59e0b','#ef4444'];
const COLORS_ALGO = ['#38bdf8','#8b5cf6','#22c55e','#f59e0b','#ef4444'];

const SEVERITY_DIST = [
  { name: 'Slight Injury',  value: 15227, pct: 59.98 },
  { name: 'Serious Injury', value: 6345,  pct: 24.99 },
  { name: 'Fatal injury',   value: 3808,  pct: 15.03 },
];

const METRICS = [
  { label: 'Accuracy',   value: 90.82, target: 90, color: '#38bdf8' },
  { label: 'F1 (Wtd)',   value: 90.91, target: 90, color: '#8b5cf6' },
  { label: 'Precision',  value: 91.14, target: 90, color: '#22c55e' },
  { label: 'Recall',     value: 90.82, target: 90, color: '#f59e0b' },
  { label: 'ROC-AUC',   value: 98.22, target: 95, color: '#ef4444' },
  { label: 'F1 (Mac)',   value: 88.25, target: 85, color: '#06b6d4' },
];

const PER_CLASS = [
  { class: 'Slight Injury',  precision: 96, recall: 95, f1: 95 },
  { class: 'Serious Injury', precision: 79, recall: 86, f1: 82 },
  { class: 'Fatal injury',   precision: 91, recall: 83, f1: 87 },
];

const MODEL_COMPARISON = [
  { model: 'CART #7',      accuracy: 86.24, f1: 84.10 },
  { model: 'Extra Trees',  accuracy: 90.02, f1: 89.80 },
  { model: 'Bagging #19',  accuracy: 89.03, f1: 88.40 },
  { model: 'Stacking #23', accuracy: 90.82, f1: 90.91 },
];

const FEATURE_IMPORTANCE = [
  { feature: 'Cause_of_accident',           importance: 0.142 },
  { feature: 'Number_of_casualties',        importance: 0.128 },
  { feature: 'Light_conditions',            importance: 0.097 },
  { feature: 'Type_of_collision',           importance: 0.091 },
  { feature: 'Time',                        importance: 0.085 },
  { feature: 'Driving_experience',          importance: 0.078 },
  { feature: 'Road_surface_conditions',     importance: 0.064 },
  { feature: 'Number_of_vehicles_involved', importance: 0.058 },
  { feature: 'Road_allignment',             importance: 0.051 },
  { feature: 'Area_accident_occured',       importance: 0.047 },
].reverse();

const REGULARIZATION = [
  { config: 'CART (no reg)', train: 100, test: 85.89 },
  { config: 'CART depth=6',  train: 77.4, test: 78.5  },
  { config: 'Bagging 60',    train: 100,  test: 89.48 },
  { config: 'Extra (150)',   train: 100,  test: 90.11 },
  { config: 'Stacking',      train: null, test: 90.82 },
];

const RADAR_DATA = [
  { subject: 'Accuracy', value: 90.82 },
  { subject: 'Precision', value: 91.14 },
  { subject: 'Recall',    value: 90.82 },
  { subject: 'F1 Wtd',   value: 90.91 },
  { subject: 'ROC-AUC',  value: 98.22 },
  { subject: 'F1 Macro', value: 88.25 },
];

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background:'#111a2e', border:'1px solid rgba(99,179,237,0.2)', borderRadius:8, padding:'10px 14px' }}>
      <div style={{ color:'#94a3b8', fontSize:12, marginBottom:6 }}>{label}</div>
      {payload.map((p,i) => (
        <div key={i} style={{ color: p.color || '#e2e8f0', fontSize:13, fontWeight:600 }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(2) : p.value}
          {p.name?.includes('accuracy') || p.name?.includes('f1') || p.name?.includes('test') || p.name?.includes('train') ? '%' : ''}
        </div>
      ))}
    </div>
  );
};

function StatCard({ label, value, color, target }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    let frame;
    const duration = 1500;
    const start = Date.now();
    const tick = () => {
      const p = Math.min((Date.now() - start) / duration, 1);
      const e = 1 - Math.pow(1 - p, 3);
      setDisplay(e * value);
      if (p < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [value]);

  return (
    <div className="stat-card" style={{ '--sc': color }}>
      <div className="stat-card__value">{display.toFixed(2)}%</div>
      <div className="stat-card__label">{label}</div>
      <div className="stat-card__bar">
        <div className="stat-card__fill" style={{ width: `${display}%`, background: color }} />
      </div>
      <div className="stat-card__target">Target: {target}%</div>
    </div>
  );
}

export default function Dashboard() {
  return (
    <div className="dashboard">
      <div className="dashboard__inner">
        <div className="dash-header">
          <h1 className="dash-header__title">Model Performance Dashboard</h1>
          <p className="dash-header__sub">Stacking Ensemble · Addis Ababa RTA Dataset · 25,380 Records</p>
          <div className="dash-header__tags">
            {['CART #7','Extra Trees #21','Bagging #19','Stacking #23','LR Meta #2'].map(t => (
              <span key={t} className="dash-tag">{t}</span>
            ))}
          </div>
        </div>

        {/* Metric cards */}
        <div className="metrics-row">
          {METRICS.map(m => <StatCard key={m.label} {...m} />)}
        </div>

        {/* Row 1 */}
        <div className="chart-row">
          {/* Severity distribution */}
          <div className="chart-card chart-card--md">
            <div className="chart-card__header">
              <h3 className="chart-card__title">Severity Distribution</h3>
              <span className="chart-card__sub">25,380 total records</span>
            </div>
            <div className="chart-card__body">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={SEVERITY_DIST} cx="50%" cy="50%" innerRadius={60} outerRadius={90}
                    dataKey="value" nameKey="name" paddingAngle={3}
                    label={({ name, pct }) => `${pct}%`} labelLine={false}>
                    {SEVERITY_DIST.map((_, i) => <Cell key={i} fill={COLORS_SEV[i]} />)}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                  <Legend formatter={v => <span style={{color:'#94a3b8',fontSize:12}}>{v}</span>} />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Model comparison */}
          <div className="chart-card chart-card--lg">
            <div className="chart-card__header">
              <h3 className="chart-card__title">Algorithm Comparison</h3>
              <span className="chart-card__sub">Accuracy vs F1 Score</span>
            </div>
            <div className="chart-card__body">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={MODEL_COMPARISON} barSize={18} barGap={4}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="model" tick={{ fill:'#64748b', fontSize:11 }} axisLine={false} tickLine={false} />
                  <YAxis domain={[80,95]} tick={{ fill:'#64748b', fontSize:11 }} axisLine={false} tickLine={false} unit="%" />
                  <Tooltip content={<CustomTooltip />} cursor={{ fill:'rgba(255,255,255,0.03)' }} />
                  <Bar dataKey="accuracy" name="accuracy" fill="#38bdf8" radius={[4,4,0,0]} />
                  <Bar dataKey="f1"       name="f1"       fill="#8b5cf6" radius={[4,4,0,0]} />
                  <Legend formatter={v => <span style={{color:'#94a3b8',fontSize:12,textTransform:'capitalize'}}>{v}</span>} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Row 2 */}
        <div className="chart-row">
          {/* Radar */}
          <div className="chart-card chart-card--md">
            <div className="chart-card__header">
              <h3 className="chart-card__title">Performance Radar</h3>
              <span className="chart-card__sub">All metrics normalized</span>
            </div>
            <div className="chart-card__body">
              <ResponsiveContainer width="100%" height={240}>
                <RadarChart data={RADAR_DATA}>
                  <PolarGrid stroke="rgba(255,255,255,0.06)" />
                  <PolarAngleAxis dataKey="subject" tick={{ fill:'#64748b', fontSize:11 }} />
                  <Radar name="Score" dataKey="value" stroke="#38bdf8" fill="#38bdf8" fillOpacity={0.15} dot={{ fill:'#38bdf8', r:4 }} />
                  <Tooltip content={<CustomTooltip />} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Per-class metrics */}
          <div className="chart-card chart-card--lg">
            <div className="chart-card__header">
              <h3 className="chart-card__title">Per-Class Performance</h3>
              <span className="chart-card__sub">Precision · Recall · F1</span>
            </div>
            <div className="chart-card__body">
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={PER_CLASS} barSize={16} barGap={3}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="class" tick={{ fill:'#64748b', fontSize:10 }} axisLine={false} tickLine={false} />
                  <YAxis domain={[60,100]} tick={{ fill:'#64748b', fontSize:11 }} axisLine={false} tickLine={false} unit="%" />
                  <Tooltip content={<CustomTooltip />} cursor={{ fill:'rgba(255,255,255,0.03)' }} />
                  <Bar dataKey="precision" name="Precision" fill="#38bdf8" radius={[3,3,0,0]} />
                  <Bar dataKey="recall"    name="Recall"    fill="#22c55e" radius={[3,3,0,0]} />
                  <Bar dataKey="f1"        name="F1"        fill="#f59e0b" radius={[3,3,0,0]} />
                  <Legend formatter={v => <span style={{color:'#94a3b8',fontSize:12}}>{v}</span>} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Row 3: Feature Importance */}
        <div className="chart-row">
          <div className="chart-card chart-card--full">
            <div className="chart-card__header">
              <h3 className="chart-card__title">Feature Importance</h3>
              <span className="chart-card__sub">From Extra Trees base learner · mean Gini impurity decrease</span>
            </div>
            <div className="chart-card__body">
              <ResponsiveContainer width="100%" height={320}>
                <BarChart data={FEATURE_IMPORTANCE} layout="vertical" barSize={14}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" horizontal={false} />
                  <XAxis type="number" domain={[0,0.16]} tick={{ fill:'#64748b', fontSize:11 }} axisLine={false} tickLine={false} />
                  <YAxis type="category" dataKey="feature" width={220} tick={{ fill:'#94a3b8', fontSize:11 }} axisLine={false} tickLine={false} />
                  <Tooltip content={<CustomTooltip />} cursor={{ fill:'rgba(255,255,255,0.03)' }} />
                  <Bar dataKey="importance" name="Importance" radius={[0,4,4,0]}>
                    {FEATURE_IMPORTANCE.map((_, i) => (
                      <Cell key={i} fill={`hsl(${200 + i*8},80%,${55 + i*2}%)`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Row 4: Regularization */}
        <div className="chart-row">
          <div className="chart-card chart-card--full">
            <div className="chart-card__header">
              <h3 className="chart-card__title">Regularization Analysis — Train vs Test Accuracy</h3>
              <span className="chart-card__sub">Demonstrating overfitting without regularization</span>
            </div>
            <div className="chart-card__body">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={REGULARIZATION} barSize={28} barGap={6}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="config" tick={{ fill:'#64748b', fontSize:11 }} axisLine={false} tickLine={false} />
                  <YAxis domain={[70,105]} tick={{ fill:'#64748b', fontSize:11 }} axisLine={false} tickLine={false} unit="%" />
                  <Tooltip content={<CustomTooltip />} cursor={{ fill:'rgba(255,255,255,0.03)' }} />
                  <Bar dataKey="train" name="train" fill="#3b82f6" radius={[4,4,0,0]} opacity={0.7} />
                  <Bar dataKey="test"  name="test"  fill="#22c55e" radius={[4,4,0,0]} />
                  <Legend formatter={v => <span style={{color:'#94a3b8',fontSize:12,textTransform:'capitalize'}}>{v} Accuracy</span>} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Dataset info table */}
        <div className="info-table-wrap">
          <h3 className="info-table-title">Dataset Summary</h3>
          <div className="info-table">
            {[
              ['Total Records','25,380'],
              ['Training Set','20,304 (80%)'],
              ['Test Set','5,076 (20%)'],
              ['Features','23 (21 categorical, 2 numerical)'],
              ['Source 1','Mendeley — 12,316 records (2017-2020)'],
              ['Source 2','Figshare — 13,064 records (2016-2022)'],
              ['Imbalance Fix','SMOTE oversampling (training only)'],
              ['Cross-Validation','Stratified 3-Fold'],
            ].map(([k,v]) => (
              <div key={k} className="info-row">
                <span className="info-row__key">{k}</span>
                <span className="info-row__val">{v}</span>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
