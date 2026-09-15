import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { CloseIcon } from '@/components/Icons';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip } from 'chart.js';
import { Line } from 'react-chartjs-2';
import { STAT_LABELS, formatNumber } from './scale';
import type { LaiSeries, RegionStats } from '@/api/results';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip);

type Props = {
  region: string;
  stats?: RegionStats;
  /** Observed-LAI curves for the level; this region's is picked out of it. */
  lai?: LaiSeries | null;
  /** Present only for the level whose region ids have report plots (the primary level). */
  plotUrl?: string;
  /** Name of the level that does carry per-region reports, for the "switch to" prompt. */
  plotsLevel?: string;
  onGoToPlotsLevel?: () => void;
  /** Levels that do have reference yields, for the "no reference data here" note. */
  referenceLevels?: string[];
  onGoToLevel?: (level: string) => void;
  /** Whether this region is currently in the comparison set. */
  inComparison: boolean;
  onToggleCompare: () => void;
  onClose: () => void;
};

/**
 * Detail pane for a clicked region.
 *
 * The per-region report plot is ~0.3 MiB and there is one per region, so it is fetched
 * here on demand rather than shipped up front - the legacy map extracted all of them
 * (189 MiB on Ukraine) before showing anything.
 */
export default function RegionPanel({
  region,
  stats,
  lai,
  plotUrl,
  plotsLevel,
  onGoToPlotsLevel,
  referenceLevels,
  onGoToLevel,
  inComparison,
  onToggleCompare,
  onClose,
}: Props) {
  const [plotState, setPlotState] = useState<'idle' | 'loading' | 'error'>('idle');
  // The report is a dense multi-panel figure; at panel width it is unreadable, so it
  // opens full-screen on click rather than making the reader zoom the browser.
  const [enlarged, setEnlarged] = useState(false);

  useEffect(() => {
    setPlotState(plotUrl ? 'loading' : 'idle');
  }, [plotUrl, region]);

  const rows = STAT_LABELS.filter(([key]) => stats && stats[key] !== undefined && stats[key] !== null);
  // An empty reference cell is ambiguous: it could mean this region was not surveyed, or
  // that the whole level has no ground truth. Say which, and where to find it.
  const hasReference = typeof stats?.reported_mean_yield_kg_ha === 'number';
  const elsewhere = (referenceLevels ?? []).filter((l) => l !== undefined);

  return (
    <aside className="region-panel">
      <div className="region-panel-head">
        <div className="region-panel-title" title={region}>
          {region}
        </div>
        <button className="btn btn-xs btn-secondary" onClick={onClose} title="Close" aria-label="Close region details">
          <CloseIcon size={14} />
        </button>
      </div>

      <div className="region-panel-actions">
        <button
          className={`btn btn-xs ${inComparison ? 'btn-primary' : 'btn-secondary'}`}
          onClick={onToggleCompare}
          title="Compare this region against others in the summary panel"
        >
          {inComparison ? 'In comparison - remove' : '+ Compare with others'}
        </button>
        <span className="region-panel-shortcut">or shift-click regions on the map</span>
      </div>

      {!stats && <div className="region-panel-empty">No statistics for this region in the selected year.</div>}

      {rows.length > 0 && (
        <table className="region-panel-table">
          <tbody>
            {rows.map(([key, label, digits]) => (
              <tr key={key}>
                <th>{label}</th>
                <td>{formatNumber(stats?.[key], digits)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {!hasReference && (
        <div className="region-panel-note">
          <strong>Reference yield:</strong> not available at this level.
          {elsewhere.length > 0 ? (
            <>
              {' '}
              Reported yields exist at{' '}
              {elsewhere.map((l, i) => (
                <span key={l}>
                  {i > 0 && (i === elsewhere.length - 1 ? ' or ' : ', ')}
                  {onGoToLevel ? (
                    <button type="button" className="stats-clear" onClick={() => onGoToLevel(l)}>
                      {l}
                    </button>
                  ) : (
                    <strong>{l}</strong>
                  )}
                </span>
              ))}
              , so accuracy metrics and the error colourings are only meaningful there.
            </>
          ) : (
            ' This study has no reported yields for any level, so nothing is evaluated against ground truth.'
          )}
        </div>
      )}

      {lai?.regions?.[region] && (
        <div className="region-panel-plot">
          <div className="stats-block-title">Observed LAI</div>
          <div className="stats-chart" style={{ height: 130 }}>
            <Line
              data={{
                labels: lai.dates,
                datasets: [
                  {
                    label: region,
                    data: lai.regions[region],
                    borderColor: 'rgb(50 98 71)',
                    borderWidth: 1.8,
                    pointRadius: 0,
                    spanGaps: true,
                    tension: 0.25,
                  },
                ],
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                  x: { grid: { color: 'rgb(237 236 235)' }, ticks: { color: 'rgb(118 115 111)', font: { size: 9 }, maxRotation: 0, autoSkipPadding: 16 } },
                  y: { grid: { color: 'rgb(237 236 235)' }, ticks: { color: 'rgb(118 115 111)', font: { size: 9 } } },
                },
              }}
            />
          </div>
        </div>
      )}

      {/* Reports exist per simulated region, so an aggregation level has none. Saying so
          beats an empty pane that reads as a load that never finished. */}
      {!plotUrl && plotsLevel && (
        <div className="region-panel-plot">
          <div className="stats-block-title">Simulation report</div>
          <div className="region-panel-empty">
            No simulation report at this level. Reports are produced per simulated region.
            {onGoToPlotsLevel && (
              <>
                {' '}
                <button type="button" className="stats-clear" onClick={onGoToPlotsLevel}>
                  Switch to the {plotsLevel} level
                </button>{' '}
                to see them.
              </>
            )}
          </div>
        </div>
      )}

      {plotUrl && (
        <div className="region-panel-plot">
          <div className="stats-block-title">
            Yield report
            {plotState !== 'error' && <span className="region-plot-hint">click to enlarge</span>}
          </div>
          {plotState === 'loading' && <div className="region-panel-empty">Loading plot…</div>}
          {plotState === 'error' && <div className="region-panel-empty">No plot available for this region.</div>}
          <button
            type="button"
            className="region-plot-button"
            onClick={() => setEnlarged(true)}
            title="Enlarge report"
            style={{ display: plotState === 'error' ? 'none' : 'block' }}
          >
            <img
              src={plotUrl}
              alt={`Yield report for ${region}`}
              onLoad={() => setPlotState('idle')}
              onError={() => setPlotState('error')}
            />
          </button>
        </div>
      )}

      {/* Rendered into document.body: this panel sets its own z-index and so creates a
          stacking context, inside which no z-index can rise above the panel itself - the
          overlay was landing *under* the map's zoom buttons and the toolbar. */}
      {enlarged &&
        plotUrl &&
        createPortal(
          <div
            className="plot-lightbox"
            role="dialog"
            aria-modal="true"
            aria-label={`Yield report for ${region}`}
            onClick={() => setEnlarged(false)}
            onKeyDown={(e) => e.key === 'Escape' && setEnlarged(false)}
            tabIndex={-1}
            ref={(el) => el?.focus()}
          >
            <div className="plot-lightbox-bar">
              <span>{region} - yield report</span>
              <button
                className="btn btn-xs btn-secondary"
                onClick={() => setEnlarged(false)}
                aria-label="Close report"
              >
                <CloseIcon size={14} />
              </button>
            </div>
            {/* Stop propagation so clicking the figure itself does not dismiss it. */}
            <img src={plotUrl} alt={`Yield report for ${region}`} onClick={(e) => e.stopPropagation()} />
          </div>,
          document.body,
        )}

    </aside>
  );
}
