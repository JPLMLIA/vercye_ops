import { useEffect, useState } from 'react';
import { CROPLAND_COLOR, NO_DATA_FILL, divergingColor, formatNumber, isDiverging, rampColor } from './scale';

type Props = {
  title: string;
  min: number;
  max: number;
  digits?: number;
  /** Cropmask is 0/1, so a continuous ramp would imply values that do not exist. */
  binary?: boolean;
  /** Metric key; diverging metrics are drawn symmetrically about a neutral zero. */
  metric?: string;
  /** When given, the bounds become editable and this reports a new pair (or null to reset). */
  onRange?: (range: { min: number; max: number } | null) => void;
  /** True when the current bounds were set by hand rather than derived from the data. */
  pinned?: boolean;
};

const STOPS = 24;

/**
 * One legend per layer: the aggregated vector values and the pixel layer are different
 * quantities on different scales, and a single legend for both invited reading a region's
 * fill against the pixel numbers.
 *
 * The bounds are editable where a range is supplied. The default domain is shared across
 * every year and admin level so a colour keeps its meaning, but a study whose interest
 * sits in a narrow band still needs to be able to stretch the ramp over it.
 */
export default function Legend({ title, min, max, digits = 0, binary, metric, onRange, pinned }: Props) {
  const [editing, setEditing] = useState(false);
  const [lo, setLo] = useState(String(min));
  const [hi, setHi] = useState(String(max));

  useEffect(() => {
    if (!editing) {
      setLo(String(Math.round(min * 100) / 100));
      setHi(String(Math.round(max * 100) / 100));
    }
  }, [min, max, editing]);

  if (binary) {
    return (
      <div className="map-legend">
        <div className="map-legend-title">{title}</div>
        <div className="map-legend-nodata">
          <span className="map-legend-swatch" style={{ background: CROPLAND_COLOR }} /> cropland
        </div>
        <div className="map-legend-nodata">
          <span className="map-legend-swatch" style={{ background: NO_DATA_FILL }} /> not cropland
        </div>
      </div>
    );
  }

  const diverging = !!metric && isDiverging(metric);
  // A diverging scale is drawn symmetrically so the neutral midpoint always means zero.
  const bound = diverging ? Math.max(Math.abs(min), Math.abs(max)) || 1 : 0;
  const gradient = Array.from({ length: STOPS }, (_, i) => {
    const u = i / (STOPS - 1);
    return diverging ? divergingColor(u * 2 - 1) : rampColor(u);
  }).join(', ');

  const apply = () => {
    const a = Number(lo);
    const b = Number(hi);
    if (Number.isFinite(a) && Number.isFinite(b) && b > a) onRange?.({ min: a, max: b });
    setEditing(false);
  };

  return (
    <div className="map-legend">
      <div className="map-legend-title">
        <span>{title}</span>
        {onRange && (
          <button
            type="button"
            className="map-legend-edit"
            onClick={() => setEditing((e) => !e)}
            title={editing ? 'Cancel' : 'Set the colour scale by hand'}
          >
            {editing ? 'cancel' : 'set range'}
          </button>
        )}
      </div>
      <div className="map-legend-bar" style={{ background: `linear-gradient(to right, ${gradient})` }} />

      {editing ? (
        <div className="map-legend-range">
          <input
            className="form-input"
            value={lo}
            onChange={(e) => setLo(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && apply()}
            aria-label="Scale minimum"
          />
          <input
            className="form-input"
            value={hi}
            onChange={(e) => setHi(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && apply()}
            aria-label="Scale maximum"
          />
          <button type="button" className="btn btn-xs btn-primary" onClick={apply}>
            Apply
          </button>
          {pinned && (
            <button
              type="button"
              className="btn btn-xs btn-secondary"
              onClick={() => {
                onRange?.(null);
                setEditing(false);
              }}
              title="Go back to the study-wide range"
            >
              Reset
            </button>
          )}
        </div>
      ) : (
        <div className="map-legend-scale">
          <span>{formatNumber(diverging ? -bound : min, digits)}</span>
          <span>{formatNumber(diverging ? 0 : (min + max) / 2, digits)}</span>
          <span>{formatNumber(diverging ? bound : max, digits)}</span>
        </div>
      )}

      <div className="map-legend-nodata">
        <span className="map-legend-swatch" style={{ background: NO_DATA_FILL }} /> no data
        {pinned && !editing && <span className="map-legend-pinned">custom range</span>}
      </div>
    </div>
  );
}
