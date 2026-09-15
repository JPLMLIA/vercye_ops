
import { NavLink } from 'react-router-dom';
import nasaHarvestLogo from '@/assets/nasa-harvest.png';

const Header = () => {
  return (
    <div className="header">
      <div className="container-inner">
        <div className="title-container">
          <div className="title-left">
            <img src={nasaHarvestLogo} alt="NASA Harvest" className="header-logo" />
            <h1 className="title">VeRCYe: Yield-Prediction Dashboard</h1>
            <div className="help-icon">
              <button className="help-button">?</button>
              <div className="help-tooltip">Help coming soon.</div>
            </div>
          </div>
          <nav>
            <ul className="navbar-nav">
              <li className="nav-item">
                <NavLink to="/" end className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>Studies</NavLink>
              </li>
              <li className="nav-item">
                <NavLink to="/lai" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>LAI</NavLink>
              </li>
               <li className="nav-item">
                <NavLink to="/cropmasks" className={({isActive}) => `nav-link ${isActive ? 'active' : ''}`}>Cropmasks</NavLink>
              </li>
            </ul>
          </nav>
        </div>
        <p className="subtitle"></p>
      </div>
    </div>
  );
}

export default Header
