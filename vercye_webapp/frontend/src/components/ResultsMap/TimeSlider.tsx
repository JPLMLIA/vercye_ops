import { useRef } from 'react';
import type { Month } from '@/api/planetary';

/**
 * Month scrubber for the Sentinel-2 imagery layer.
 *
 * Modelled on stacnotator's TimeSlider (features/visualizers/viewer/TimeSlider.tsx):
 * the ticks say where you are in the record, the label above says what you are looking
 * at. Scrub by pointer, step with the arrow keys or the chevrons. Written in plain CSS
 * rather than Tailwind since this app has no Tailwind.
 */
export default function TimeSlider({
  months,
  index,
  onSelect,
  status,
}: {
  months: Month[];
  index: number;
  onSelect: (index: number) => void;
  status?: 'loading' | 'error' | 'zoom' | null;
}) {
  const track = useRef<HTMLDivElement>(null);
  if (months.length === 0) return null;

  const clamp = (i: number) => Math.min(months.length - 1, Math.max(0, i));
  const step = (delta: number) => onSelect(clamp(index + delta));

  const pickAt = (clientX: number) => {
    const rect = track.current?.getBoundingClientRect();
    if (!rect || rect.width === 0) return;
    onSelect(clamp(Math.floor(((clientX - rect.left) / rect.width) * months.length)));
  };

  const current = months[index];
  const pct = ((index + 0.5) / months.length) * 100;

  // One label per year the season spans, placed over that year's first month.
  const yearMarks: Array<{ year: string; ratio: number }> = [];
  months.forEach((m, i) => {
    const year = m.key.slice(0, 4);
    if (!yearMarks.some((y) => y.year === year)) {
      yearMarks.push({ year, ratio: (i + 0.5) / months.length });
    }
  });

  return (
    <div className="time-slider" data-testid="imagery-time-slider">
      <button
        type="button"
        className="time-slider-step"
        aria-label="Previous month"
        title="Previous month"
        disabled={index === 0}
        onClick={() => step(-1)}
      >
        ‹
      </button>

      <div className="time-slider-main">
        <div className="time-slider-head">
          <span className="time-slider-label">{current?.label}</span>
          <span className="time-slider-count">
            {status === 'loading' && <span className="time-slider-busy">loading… </span>}
            {status === 'error' && <span className="time-slider-warn">no imagery </span>}
            {status === 'zoom' && <span className="time-slider-warn">zoom in for imagery </span>}
            {index + 1} / {months.length}
          </span>
        </div>

        <div
          ref={track}
          role="slider"
          tabIndex={0}
          aria-label="Imagery month"
          aria-valuemin={1}
          aria-valuemax={months.length}
          aria-valuenow={index + 1}
          aria-valuetext={current?.label}
          className="time-slider-track"
          onPointerDown={(e) => {
            if (e.pointerType === 'mouse' && e.button !== 0) return;
            e.currentTarget.setPointerCapture(e.pointerId);
            pickAt(e.clientX);
          }}
          onPointerMove={(e) => {
            if (e.currentTarget.hasPointerCapture(e.pointerId)) pickAt(e.clientX);
          }}
          onKeyDown={(e) => {
            if (e.key === 'ArrowLeft') step(-1);
            else if (e.key === 'ArrowRight') step(1);
            else return;
            e.preventDefault();
          }}
        >
          <span className="time-slider-rail" />
          <span className="time-slider-fill" style={{ width: `${pct}%` }} />
          {months.map((m, i) => (
            <span
              key={m.key}
              title={m.label}
              className={`time-slider-tick ${i === index ? 'is-current' : ''}`}
              style={{ left: `${((i + 0.5) / months.length) * 100}%` }}
            />
          ))}
          <span className="time-slider-thumb" style={{ left: `${pct}%` }} />
        </div>

        {yearMarks.length > 1 && (
          <div className="time-slider-years">
            {yearMarks.map((y) => (
              <span key={y.year} style={{ left: `${y.ratio * 100}%` }}>
                {y.year}
              </span>
            ))}
          </div>
        )}
      </div>

      <button
        type="button"
        className="time-slider-step"
        aria-label="Next month"
        title="Next month"
        disabled={index === months.length - 1}
        onClick={() => step(1)}
      >
        ›
      </button>
    </div>
  );
}
