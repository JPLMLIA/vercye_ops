import { Link } from 'react-router-dom';
import { BackIcon, ChartIcon } from '@/components/Icons';
import type { BasemapName } from './basemaps';

type Option = { value: string; label: string };

type Props = {
  studyId: string;
  runId?: string;
  years: Record<string, string[]>;
  year: string;
  timepoint: string;
  levels: string[];
  level: string;
  /** Which value colours the polygons. */
  metric: string;
  /** Every metric the map knows, with the levels that actually carry each one. */
  metrics: Array<{ key: string; label: string; available: boolean; whereAvailable: string[] }>;
  basemap: string;
  hasImagery: boolean;
  /** Single pixel layer: '' | 'yield' | 'apsim_yield' | 'cropmask'. */
  pixel: string;
  pixelOptions: Option[];
  opacity: number;
  onOpacity: (v: number) => void;
  search: string;
  onSearch: (v: string) => void;
  onSearchSubmit: () => void;
  statsOpen: boolean;
  onChange: (patch: Record<string, string | undefined>) => void;
};

const BASEMAPS: BasemapName[] = ['Light', 'Terrain', 'Satellite', 'Sentinel-2 imagery'];

/**
 * Map controls.
 *
 * Basemap and pixel layer are each a single choice rather than a set of checkboxes: both
 * occupy one slot on screen, so independent toggles only produced combinations where one
 * layer silently hid another. Sentinel-2 is a basemap because that is what it is - the
 * backdrop - and choosing it is what brings up the month scrubber.
 */
export default function MapToolbar({
  studyId,
  runId,
  years,
  year,
  timepoint,
  levels,
  level,
  metric,
  metrics,
  basemap,
  hasImagery,
  pixel,
  pixelOptions,
  opacity,
  onOpacity,
  search,
  onSearch,
  onSearchSubmit,
  statsOpen,
  onChange,
}: Props) {
  const yearList = Object.keys(years).sort();
  const timepoints = years[year] ?? [];
  const yearIndex = yearList.indexOf(year);

  const step = (delta: number) => {
    const next = yearList[yearIndex + delta];
    if (!next) return;
    const tps = years[next] ?? [];
    onChange({ year: next, timepoint: tps.includes(timepoint) ? timepoint : tps[0] });
  };

  return (
    <div className="map-toolbar">
      <Link to="/" className="btn btn-sm btn-secondary map-toolbar-back" title="Back to studies">
        <BackIcon size={15} />
      </Link>

      <div className="map-toolbar-title">
        <strong>{studyId}</strong>
        {runId && <span className="map-toolbar-run">archived run {runId}</span>}
      </div>

      <div className="map-toolbar-group">
        <label className="map-toolbar-label">Year</label>
        <button
          className="btn btn-xs btn-secondary"
          onClick={() => step(-1)}
          disabled={yearIndex <= 0}
          title="Previous year"
          aria-label="Previous year"
        >
          ‹
        </button>
        <select
          className="form-input map-toolbar-select"
          value={year}
          onChange={(e) => {
            const tps = years[e.target.value] ?? [];
            onChange({ year: e.target.value, timepoint: tps.includes(timepoint) ? timepoint : tps[0] });
          }}
        >
          {yearList.map((y) => (
            <option key={y} value={y}>
              {y}
            </option>
          ))}
        </select>
        <button
          className="btn btn-xs btn-secondary"
          onClick={() => step(1)}
          disabled={yearIndex < 0 || yearIndex >= yearList.length - 1}
          title="Next year"
          aria-label="Next year"
        >
          ›
        </button>
      </div>

      {timepoints.length > 1 && (
        <div className="map-toolbar-group">
          <label className="map-toolbar-label">Timepoint</label>
          <select
            className="form-input map-toolbar-select"
            value={timepoint}
            onChange={(e) => onChange({ timepoint: e.target.value })}
          >
            {timepoints.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>
      )}

      <div className="map-toolbar-group">
        <label className="map-toolbar-label">Level</label>
        <select
          className="form-input map-toolbar-select"
          value={level}
          onChange={(e) => onChange({ level: e.target.value })}
        >
          {levels.map((l) => (
            <option key={l} value={l}>
              {l}
            </option>
          ))}
        </select>
      </div>

      <div className="map-toolbar-group">
        <label className="map-toolbar-label">Colour by</label>
        <select
          className="form-input map-toolbar-select"
          value={metric}
          onChange={(e) => onChange({ metric: e.target.value })}
        >
          {/* Every level offers the same list so the choices do not shift under the user.
              What a level has no data for is disabled, and says which level does. */}
          {metrics.map((m) => (
            <option
              key={m.key}
              value={m.key}
              disabled={!m.available}
              title={
                m.available
                  ? undefined
                  : m.whereAvailable.length
                    ? `Not available at this level. Switch to ${m.whereAvailable.join(' or ')} to colour by ${m.label.toLowerCase()}.`
                    : `No ${m.label.toLowerCase()} in this study.`
              }
            >
              {m.label}
              {m.available
                ? ''
                : m.whereAvailable.length
                  ? ` (in ${m.whereAvailable.join(', ')})`
                  : ' (no data)'}
            </option>
          ))}
        </select>
      </div>

      <div className="map-toolbar-group">
        <label className="map-toolbar-label">Pixel layer</label>
        <select
          className="form-input map-toolbar-select"
          value={pixel}
          onChange={(e) => onChange({ pixel: e.target.value })}
        >
          <option value="">Off</option>
          {pixelOptions.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label}
            </option>
          ))}
        </select>
        {pixel && (
          <input
            type="range"
            min={0}
            max={100}
            value={Math.round(opacity * 100)}
            // Local state, applied straight to the live layer: routing this through the
            // URL re-rendered the page on every input event and dropped the drag.
            onChange={(e) => onOpacity(Number(e.target.value) / 100)}
            className="map-toolbar-opacity"
            title={`Layer opacity ${Math.round(opacity * 100)}%`}
            aria-label="Layer opacity"
          />
        )}
      </div>

      <div className="map-toolbar-group">
        <label className="map-toolbar-label">Basemap</label>
        <select
          className="form-input map-toolbar-select"
          value={basemap}
          onChange={(e) => onChange({ basemap: e.target.value })}
        >
          {BASEMAPS.filter((b) => b !== 'Sentinel-2 imagery' || hasImagery).map((b) => (
            <option key={b} value={b}>
              {b}
            </option>
          ))}
        </select>
      </div>

      <div className="map-toolbar-group">
        <input
          className="form-input map-toolbar-search"
          placeholder="Find region…"
          value={search}
          onChange={(e) => onSearch(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && onSearchSubmit()}
          aria-label="Find a region by name"
        />
      </div>

      <button
        className={`btn btn-sm ${statsOpen ? 'btn-primary' : 'btn-secondary'} btn-icon`}
        onClick={() => onChange({ stats: statsOpen ? '0' : '1' })}
        title={statsOpen ? 'Hide summary and charts' : 'Show summary and charts'}
        aria-label="Toggle summary panel"
      >
        <ChartIcon />
      </button>

    </div>
  );
}
