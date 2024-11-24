import React from "react";
import "./Header.css";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faShieldAlt } from "@fortawesome/free-solid-svg-icons";


function Header() {
  return (
    <header className="nav-header">
    <div className="logo">
    <FontAwesomeIcon
        icon={faShieldAlt}
        color="#6c63ff"
        size="2x"
        className="logo-icon" 
    />
    <span className="logo-text">PhishShield</span>
    </div>
      <nav className="nav-menu">
        <a href="#products" className="nav-link">Products</a>
        <a href="#solutions" className="nav-link">Solutions</a>
        <a href="#developer" className="nav-link">Developer</a>
        <a href="#partner" className="nav-link">Partner</a>
        <a href="#pricing" className="nav-link">Pricing</a>
      </nav>
      <button className="get-started-button">Get Started</button>
    </header>
  );
}

export default Header;
