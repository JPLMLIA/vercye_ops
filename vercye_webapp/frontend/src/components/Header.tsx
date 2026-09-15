import { NavLink, useNavigate } from 'react-router-dom';
import nasaHarvestLogo from '@/assets/nasa-harvest.png';
import { ChartIcon, LogsIcon, MapIcon } from '@/components/Icons';

const HARVEST_SITE = 'https://nasaharvest.org';

/**
 * App shell: a fixed left rail plus the scrolling content column.
 *
 * Kept as `Header` so every page's `<Header />` keeps working, but it is the
 * sidebar from RAAPID-ORG/stacnotator rather than the old full-width banner -
 * white rail, hairline right border, wordmark over a muted "by NASA Harvest",
 * and nav items that tint brand-green when active instead of growing underlines.
 */
const Header = () => {
  const navigate = useNavigate();

  const item = ({ isActive }: { isActive: boolean }) => `nav-link ${isActive ? 'active' : ''}`;

  return (
    <aside className="sidebar">
      <div className="sidebar-head">
        <a href={HARVEST_SITE} target="_blank" rel="noreferrer" title="NASA Harvest">
          <img src={nasaHarvestLogo} alt="NASA Harvest" className="sidebar-logo" />
        </a>
        <span className="sidebar-wordmark">
          <button type="button" className="sidebar-name" onClick={() => navigate('/')}>
            VeRCYe
          </button>
          <span className="sidebar-by">
            by{' '}
            <a href={HARVEST_SITE} target="_blank" rel="noreferrer" className="sidebar-by-link">
              NASA Harvest
            </a>
          </span>
        </span>
      </div>

      <nav className="sidebar-nav">
        <NavLink to="/" end className={item}>
          <ChartIcon size={15} /> Studies
        </NavLink>
        <NavLink to="/lai" className={item}>
          <MapIcon size={15} /> LAI
        </NavLink>
        <NavLink to="/cropmasks" className={item}>
          <LogsIcon size={15} /> Cropmasks
        </NavLink>
      </nav>

      <div className="sidebar-foot">Yield-prediction dashboard</div>
    </aside>
  );
};

export default Header;
