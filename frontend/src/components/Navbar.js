import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Activity, BarChart3, Home, Info, Menu, X } from 'lucide-react';
import './Navbar.css';

const links = [
  { to: '/',          label: 'Home',      icon: Home },
  { to: '/predict',   label: 'Predict',   icon: Activity },
  { to: '/dashboard', label: 'Dashboard', icon: BarChart3 },
  { to: '/about',     label: 'About',     icon: Info },
];

export default function Navbar() {
  const location = useLocation();
  const [scrolled, setScrolled]   = useState(false);
  const [menuOpen, setMenuOpen]   = useState(false);

  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', fn);
    return () => window.removeEventListener('scroll', fn);
  }, []);

  useEffect(() => { setMenuOpen(false); }, [location]);

  return (
    <nav className={`navbar ${scrolled ? 'navbar--scrolled' : ''}`}>
      <div className="navbar__inner">
        {/* Logo */}
        <Link to="/" className="navbar__logo">
          <div className="navbar__logo-icon">
            <span className="navbar__logo-pulse" />
            <Activity size={18} strokeWidth={2.5} />
          </div>
          <div className="navbar__logo-text">
            <span className="navbar__logo-title">RTA</span>
            <span className="navbar__logo-sub">Ethiopia · Severity AI</span>
          </div>
        </Link>

        {/* Desktop links */}
        <div className="navbar__links">
          {links.map(({ to, label, icon: Icon }) => (
            <Link
              key={to}
              to={to}
              className={`navbar__link ${location.pathname === to ? 'navbar__link--active' : ''}`}
            >
              <Icon size={15} />
              {label}
            </Link>
          ))}
        </div>

        {/* CTA */}
        <Link to="/predict" className="navbar__cta">
          Run Prediction
          <span className="navbar__cta-arrow">→</span>
        </Link>

        {/* Mobile menu button */}
        <button className="navbar__menu-btn" onClick={() => setMenuOpen(v => !v)}>
          {menuOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {/* Mobile drawer */}
      {menuOpen && (
        <div className="navbar__mobile">
          {links.map(({ to, label, icon: Icon }) => (
            <Link
              key={to}
              to={to}
              className={`navbar__mobile-link ${location.pathname === to ? 'navbar__mobile-link--active' : ''}`}
            >
              <Icon size={16} />
              {label}
            </Link>
          ))}
        </div>
      )}
    </nav>
  );
}
