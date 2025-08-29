
import { NavLink } from 'react-router-dom';

const Header = () => {
  return (
    <div className="header">
      <div className="container-inner">
        <div className="title-container">
          <div className="title-left">
            <h1 className="title">Vercye Dashboard</h1>
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
