import { useMemo } from 'react';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend } from 'chart.js';
import { Line, Scatter } from 'react-chartjs-2';
import { METRICS } from '@/api/results';
import type { EvaluationResponse, LaiSeries, MultiyearResponse, RegionStats } from '@/api/results';
import { formatNumber } from './scale';
import { CloseIcon } from '@/components/Icons';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend);

const BRAND = 'rgb(50 98 71)';
const NEUTRAL = 'rgb(162 159 155)';
const SERIES = ['rgb(50 98 71)', 'rgb(163 84 68)', 'rgb(82 118 170)', 'rgb(163 136 58)', 'rgb(90 153 125)'];

type Props = {
  level: string;
  metric: string;
  regions: Record<string, RegionStats>;
  lai: LaiSeries | null;
  /** Pipeline-computed accuracy metrics for this level and year. */
  evaluation: EvaluationResponse | null;
  /** Pipeline-computed metrics and yields for every year of this level. */
  multiyear: MultiyearResponse | null;
  /** 'year' | 'multiyear' */
  view: string;
  onView: (v: string) => void;
  /** Regions the user has picked for comparison. */
  selected: string[];
  onClearSelection: () => void;
  onClose: () => void;
};

/**
 * Summary panel: the aggregate numbers, predicted-vs-reference scatter, and LAI curves.
 *
 * Accuracy metrics are read from the pipeline's own evaluation output rather than
 * computed here. They were being recomputed in the browser, which happened to agree, but
 * a panel that derives its own statistics can drift from the reports without anyone
 * noticing - and the pipeline reports MAPE, relative RMSE and the error quantiles that
 * were not being derived at all.
 */
export default function StatsPanel({
  level,
  metric,
  regions,
  lai,
  evaluation,
  multiyear,
  view,
  onView,
  selected,
  onClearSelection,
  onClose,
}: Props) {
  const meta = METRICS.find((m) => m.key === metric) ?? METRICS[0];
  const isMultiyear = view === 'multiyear';

  const summary = useMemo(() => {
    const vals = Object.values(regions)
      .map((r) => r[metric])
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v));
    const pairs = Object.entries(regions)
      .map(([id, r]) => ({ id, pred: r.mean_yield_kg_ha, ref: r.reported_mean_yield_kg_ha }))
      .filter(
        (p): p is { id: string; pred: number; ref: number } =>
          typeof p.pred === 'number' && typeof p.ref === 'number' && Number.isFinite(p.pred) && Number.isFinite(p.ref),
      );
    return {
      n: Object.keys(regions).length,
      withValue: vals.length,
      mean: vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null,
      min: vals.length ? Math.min(...vals) : null,
      max: vals.length ? Math.max(...vals) : null,
      pairs,
    };
  }, [regions, metric]);

  /**
   * Comparison chart: only meaningful with two or more regions to compare.
   *
   * With one region the per-region panel already shows its curve, and with none a
   * level-mean line said little while implying this pane was about the level as a whole
   * rather than about comparing regions.
   */
  const laiChart = useMemo(() => {
    if (!lai || selected.length < 2) return null;
    const ids = selected.filter((id) => lai.regions[id]);
    if (ids.length < 2) return null;

    return {
      labels: lai.dates,
      datasets: ids.slice(0, 5).map((id, i) => ({
        label: id,
        data: lai.regions[id],
        borderColor: SERIES[i % SERIES.length],
        backgroundColor: SERIES[i % SERIES.length],
        borderWidth: 1.6,
        pointRadius: 0,
        spanGaps: true,
        tension: 0.25,
      })),
    };
  }, [lai, selected]);

  /** Predicted and reported yield per year, averaged over the level's regions. */
  const multiyearChart = useMemo(() => {
    if (!multiyear) return null;
    const byYear: Record<string, { pred: number[]; ref: number[] }> = {};
    for (const series of Object.values(multiyear.regions)) {
      for (const row of series) {
        const b = (byYear[row.year] ??= { pred: [], ref: [] });
        if (typeof row.predicted === 'number') b.pred.push(row.predicted);
        if (typeof row.reported === 'number') b.ref.push(row.reported);
      }
    }
    const years = multiyear.years.filter((y) => byYear[y]);
    if (!years.length) return null;
    const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
    return {
      labels: years,
      datasets: [
        {
          label: 'Predicted',
          data: years.map((y) => mean(byYear[y].pred)),
          borderColor: BRAND,
          backgroundColor: BRAND,
          borderWidth: 1.8,
          pointRadius: 2.5,
          spanGaps: true,
        },
        {
          label: 'Reference',
          data: years.map((y) => mean(byYear[y].ref)),
          borderColor: 'rgb(163 84 68)',
          backgroundColor: 'rgb(163 84 68)',
          borderWidth: 1.8,
          borderDash: [4, 3],
          pointRadius: 2.5,
          spanGaps: true,
        },
      ],
    };
  }, [multiyear]);

  const axis = {
    grid: { color: 'rgb(237 236 235)' },
    ticks: { color: 'rgb(118 115 111)', font: { size: 10 }, maxRotation: 0, autoSkipPadding: 18 },
  };
  const common = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: false } as const },
    scales: { x: axis, y: axis },
  };
  const withLegend = {
    ...common,
    plugins: { legend: { labels: { boxWidth: 10, font: { size: 10 }, color: 'rgb(77 75 73)' } } },
  };

  /** Metric rows, straight from the pipeline's evaluation output. */
  const evalRows = evaluation?.metrics
    ? evaluation.fields.filter((f) => typeof evaluation.metrics?.[f.key] === 'number')
    : [];

  return (
    <aside className="stats-panel">
      <div className="region-panel-head">
        <div className="region-panel-title">{level} summary</div>
        <button className="btn btn-xs btn-secondary" onClick={onClose} title="Close" aria-label="Close summary">
          <CloseIcon size={14} />
        </button>
      </div>

      <div className="stats-tabs" role="tablist">
        <button
          role="tab"
          aria-selected={!isMultiyear}
          className={`stats-tab ${!isMultiyear ? 'is-active' : ''}`}
          onClick={() => onView('year')}
        >
          This year
        </button>
        <button
          role="tab"
          aria-selected={isMultiyear}
          className={`stats-tab ${isMultiyear ? 'is-active' : ''}`}
          onClick={() => onView('multiyear')}
        >
          All years
        </button>
      </div>

      <div className="stats-panel-body">
        {!isMultiyear && (
          <>
            <table className="region-panel-table">
              <tbody>
                <tr>
                  <th>Regions</th>
                  <td>
                    {summary.withValue} / {summary.n}
                  </td>
                </tr>
                <tr>
                  <th>
                    {meta.label} mean{meta.unit ? ` (${meta.unit})` : ''}
                  </th>
                  <td>{formatNumber(summary.mean, meta.digits)}</td>
                </tr>
                <tr>
                  <th>Range</th>
                  <td>
                    {formatNumber(summary.min, meta.digits)} - {formatNumber(summary.max, meta.digits)}
                  </td>
                </tr>
              </tbody>
            </table>

            {evalRows.length > 0 && (
              <div className="stats-block">
                <div className="stats-block-title">Accuracy vs reference</div>
                <table className="region-panel-table">
                  <tbody>
                    {evalRows.map((f) => (
                      <tr key={f.key}>
                        <th>{f.label}</th>
                        <td>{formatNumber(evaluation?.metrics?.[f.key], f.digits)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="stats-hint">From the pipeline&rsquo;s evaluation step, not recomputed here.</div>
              </div>
            )}

            {summary.pairs.length > 1 && (
              <div className="stats-block">
                <div className="stats-block-title">Predicted vs reference (kg/ha)</div>
                <div className="stats-chart">
                  <Scatter
                    data={{
                      datasets: [
                        {
                          label: 'regions',
                          data: summary.pairs.map((p) => ({ x: p.ref, y: p.pred })),
                          backgroundColor: BRAND,
                          pointRadius: 3,
                        },
                        {
                          label: '1:1',
                          type: 'line' as unknown as undefined,
                          data: (() => {
                            const lo = Math.min(...summary.pairs.flatMap((p) => [p.ref, p.pred]));
                            const hi = Math.max(...summary.pairs.flatMap((p) => [p.ref, p.pred]));
                            return [
                              { x: lo, y: lo },
                              { x: hi, y: hi },
                            ];
                          })(),
                          borderColor: NEUTRAL,
                          borderDash: [4, 4],
                          borderWidth: 1,
                          pointRadius: 0,
                        },
                      ],
                    }}
                    options={common}
                  />
                </div>
              </div>
            )}
          </>
        )}

        {isMultiyear && (
          <>
            {!multiyear && <div className="region-panel-empty">Loading all years...</div>}

            {multiyear && multiyearChart && (
              <div className="stats-block">
                <div className="stats-block-title">Mean yield by year (kg/ha)</div>
                <div className="stats-chart">
                  <Line data={multiyearChart} options={withLegend} />
                </div>
                <div className="stats-hint">
                  Level means over the regions in{' '}
                  <code>{multiyear.source ?? 'the pipeline predictions file'}</code>.
                </div>
              </div>
            )}

            {multiyear && multiyear.metrics.length > 0 && (
              <div className="stats-block">
                <div className="stats-block-title">Accuracy by year</div>
                <div className="stats-scroll">
                  <table className="region-panel-table compare-table">
                    <thead>
                      <tr>
                        <th>Year</th>
                        <td>R&sup2;</td>
                        <td>RMSE</td>
                        <td>rRMSE</td>
                        <td>MAE</td>
                      </tr>
                    </thead>
                    <tbody>
                      {multiyear.metrics.map((m) => (
                        <tr key={String(m.year)}>
                          <th>{m.year}</th>
                          <td>{formatNumber(m.r2_scikit, 3)}</td>
                          <td>{formatNumber(m.rmse_kg_ha, 0)}</td>
                          <td>{formatNumber(m.rrmse, 1)}</td>
                          <td>{formatNumber(m.mean_abs_err_kg_ha, 0)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="stats-hint">
                  RMSE, rRMSE and MAE in kg/ha except rRMSE (%). All read from the pipeline&rsquo;s
                  per-year evaluation output.
                </div>
              </div>
            )}

            {multiyear && !multiyear.metrics.length && (
              <div className="stats-block">
                <div className="stats-block-title">Accuracy by year</div>
                <div className="stats-empty-hint">
                  This level has no reference yields, so the pipeline computes no accuracy metrics
                  for it.
                </div>
              </div>
            )}
          </>
        )}

        {selected.length === 0 && (
          <div className="stats-block">
            <div className="stats-block-title">Compare regions</div>
            <div className="stats-empty-hint">
              Select regions to compare their LAI curves, as well as their reported and predicted
              yields, to get a quick insight into how these might correlate.
              <br />
              <strong>Shift-click</strong> regions on the map for easy comparison.
            </div>
          </div>
        )}

        {selected.length > 0 && (
          <div className="stats-block">
            <div className="stats-block-title">
              Selected regions
              <button className="stats-clear" onClick={onClearSelection}>
                clear {selected.length}
              </button>
            </div>
            <div className="stats-scroll">
              <table className="region-panel-table compare-table">
                <thead>
                  <tr>
                    <th>Region</th>
                    <td>Pred.</td>
                    <td>Ref.</td>
                    <td>&Delta;</td>
                  </tr>
                </thead>
                <tbody>
                  {selected.map((id, i) => {
                    const r = regions[id];
                    const pred = r?.mean_yield_kg_ha;
                    const ref = r?.reported_mean_yield_kg_ha;
                    const diff = typeof pred === 'number' && typeof ref === 'number' ? pred - ref : null;
                    return (
                      <tr key={id}>
                        <th title={id}>
                          <span
                            className="compare-swatch"
                            style={{ background: SERIES[i % SERIES.length] }}
                            aria-hidden
                          />
                          <span className="compare-name">{id}</span>
                        </th>
                        <td>{formatNumber(pred)}</td>
                        <td>{formatNumber(ref)}</td>
                        <td className={diff !== null && diff < 0 ? 'compare-neg' : undefined}>
                          {diff === null ? '-' : `${diff > 0 ? '+' : ''}${formatNumber(diff)}`}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            <div className="stats-hint">Yields in kg/ha; &Delta; is predicted minus reference.</div>
          </div>
        )}

        {selected.length === 1 && lai && (
          <div className="stats-block">
            <div className="stats-block-title">Observed LAI</div>
            <div className="stats-empty-hint">
              Add a second region to overlay their LAI curves. This region&rsquo;s own curve is in the
              detail pane on the right.
            </div>
          </div>
        )}

        {laiChart && (
          <div className="stats-block">
            <div className="stats-block-title">Observed LAI</div>
            <div className="stats-chart">
              <Line data={laiChart} options={withLegend} />
            </div>
            <div className="stats-hint">Shift-click regions on the map to add or remove them.</div>
          </div>
        )}
      </div>
    </aside>
  );
}
