import React, { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Activity, BarChart3, Brain, Shield, Zap, ChevronRight, TrendingUp } from 'lucide-react';
import './Home.css';

const METRICS = [
  { label: 'Test Accuracy',    value: '90.82%', icon: TrendingUp,  color: '#38bdf8' },
  { label: 'ROC-AUC Score',   value: '0.9822',  icon: BarChart3,   color: '#8b5cf6' },
  { label: 'F1 (Weighted)',    value: '90.4%',   icon: Activity,    color: '#22c55e' },
  { label: 'Total Records',   value: '25,380',   icon: Brain,       color: '#f59e0b' },
];

const FEATURES = [
  { icon: Zap,    title: 'Real-time Prediction', desc: 'Instant severity classification from 23 accident features in milliseconds.' },
  { icon: Brain,  title: 'Stacking Ensemble',    desc: 'CART + Extra Trees + Bagging base learners combined via Logistic Regression meta-learner.' },
  { icon: Shield, title: 'Regularized',           desc: 'L2 penalty, bootstrap subsampling, and SMOTE oversampling prevent overfitting.' },
  { icon: BarChart3, title: 'Interactive Dashboard', desc: 'Live performance metrics, ROC curves, and feature importance charts.' },
];

function AnimatedNumber({ target, suffix = '' }) {
  const [current, setCurrent] = useState(0);
  const ref = useRef(null);

  useEffect(() => {
    const num = parseFloat(target.replace(/[^0-9.]/g, ''));
    let start = null;
    const duration = 1800;
    const step = (ts) => {
      if (!start) start = ts;
      const progress = Math.min((ts - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setCurrent(eased * num);
      if (progress < 1) requestAnimationFrame(step);
    };
    const observer = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) { requestAnimationFrame(step); observer.disconnect(); }
    });
    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [target]);

  const raw = parseFloat(target.replace(/[^0-9.]/g, ''));
  const decimals = target.includes('.') ? (target.split('.')[1]?.replace(/[^0-9]/g,'').length || 0) : 0;
  const formatted = target.includes(',')
    ? Math.round(current).toLocaleString()
    : current.toFixed(decimals);

  const unit = target.replace(/[0-9.,]/g, '');
  return <span ref={ref}>{formatted}{unit}</span>;
}

export default function Home() {
  return (
    <div className="home">
      {/* Hero */}
      <section className="hero">
        <div className="hero__badge">
          <span className="hero__badge-dot" />
          Live ML System · Addis Ababa RTA Dataset
        </div>
        <h1 className="hero__title">
          Predict Road Accident
          <br />
          <span className="hero__title-gradient">Severity with AI</span>
        </h1>
        <p className="hero__subtitle">
          A production-grade stacking ensemble trained on 25,380 Addis Ababa
          police records. Classifies accidents as Slight, Serious, or Fatal
          with 90.82% accuracy.
        </p>
        <div className="hero__actions">
          <Link to="/predict" className="btn btn--primary">
            Start Prediction <ChevronRight size={18} />
          </Link>
          <Link to="/dashboard" className="btn btn--ghost">
            View Dashboard
          </Link>
        </div>

        {/* Floating badges */}
        <div className="hero__badges">
          {['CART #7','Extra Trees #21','Bagging #19','Stacking #23'].map((t,i) => (
            <div key={t} className="hero__tag" style={{ animationDelay: `${i*0.1}s` }}>{t}</div>
          ))}
        </div>
      </section>

      {/* Metrics strip */}
      <section className="metrics">
        <div className="metrics__inner">
          {METRICS.map(({ label, value, icon: Icon, color }, i) => (
            <div key={label} className="metric-card" style={{ animationDelay: `${i*0.1}s` }}>
              <div className="metric-card__icon" style={{ color, background: `${color}18` }}>
                <Icon size={20} />
              </div>
              <div className="metric-card__value">
                <AnimatedNumber target={value} />
              </div>
              <div className="metric-card__label">{label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section className="features">
        <div className="section-header">
          <h2 className="section-title">What Makes This System Unique</h2>
          <p className="section-sub">Built using only classical ML algorithms from an approved academic list.</p>
        </div>
        <div className="features__grid">
          {FEATURES.map(({ icon: Icon, title, desc }, i) => (
            <div key={title} className="feature-card" style={{ animationDelay: `${i*0.08}s` }}>
              <div className="feature-card__icon">
                <Icon size={22} />
              </div>
              <h3 className="feature-card__title">{title}</h3>
              <p className="feature-card__desc">{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Severity legend */}
      <section className="severity-section">
        <div className="section-header">
          <h2 className="section-title">Three Severity Classes</h2>
        </div>
        <div className="severity-cards">
          {[
            { label:'Slight Injury',  pct:'60%',  color:'var(--slight)',  desc:'Minor injuries, no hospitalization required. Most common category.' },
            { label:'Serious Injury', pct:'25%',  color:'var(--serious)', desc:'Requires hospitalization. Significant road safety concern.' },
            { label:'Fatal injury',   pct:'15%',  color:'var(--fatal)',   desc:'Death involved. Highest-consequence class requiring immediate response.' },
          ].map(({ label, pct, color, desc }) => (
            <div key={label} className="sev-card" style={{ '--sev-color': color }}>
              <div className="sev-card__indicator" />
              <div className="sev-card__pct">{pct}</div>
              <div className="sev-card__label">{label}</div>
              <div className="sev-card__desc">{desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="cta-section">
        <div className="cta-inner">
          <h2 className="cta-title">Ready to Run a Prediction?</h2>
          <p className="cta-sub">Enter accident details and get an instant severity classification with confidence scores.</p>
          <Link to="/predict" className="btn btn--primary btn--lg">
            Open Prediction Tool <ChevronRight size={20} />
          </Link>
        </div>
      </section>
    </div>
  );
}
