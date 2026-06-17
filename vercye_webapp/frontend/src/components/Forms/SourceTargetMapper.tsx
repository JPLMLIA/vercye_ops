import React, { useMemo, useState, useId } from "react";

export type MapperSource = { id: string; label: string; file?: File };
export type MappingState = Record<string, string | string[]>; // sourceId -> target(s)

interface SourceTargetMapperProps {
  title?: string;
  hint?: string;
  sources: MapperSource[];
  targets: string[];
  value: MappingState;
  onChange: (next: MappingState) => void;
  allowDuplicateTargets?: boolean; // default true
  /** If true, each source can map to multiple targets (checkboxes). Otherwise radios */
  multi?: boolean; // default false
  /** Grid template for the Source | Map-to columns. Default splits 50/50.
   *  Use e.g. "minmax(64px, 96px) 1fr" to give long target names more room. */
  gridTemplateColumns?: string; // default "2fr 2fr"
  /** Show a per-row text filter above the target list (useful for long target lists). */
  searchable?: boolean; // default false
  /** Minimum width of each target card; larger values fit fewer, wider cards per row. */
  targetMinWidth?: number; // default 180
  /** Allow long target labels to wrap onto multiple lines instead of truncating. */
  wrapLabels?: boolean; // default false
}

const SourceTargetMapper: React.FC<SourceTargetMapperProps> = ({
  title,
  hint,
  sources,
  targets,
  value,
  onChange,
  allowDuplicateTargets = true,
  multi = false,
  gridTemplateColumns = "2fr 2fr",
  searchable = false,
  targetMinWidth = 180,
  wrapLabels = false,
}) => {
  const uid = useId();
  const [queries, setQueries] = useState<Record<string, string>>({});

  // All selected values across all rows
  const assignedTargets: string[] = useMemo(() => {
    const vals = Object.values(value).filter(
      (v) => v !== "" && v !== undefined
    ) as (string | string[])[];
    return vals.flatMap((v) => (Array.isArray(v) ? v : [v]));
  }, [value]);

  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    assignedTargets.forEach((t) => (c[t] = (c[t] || 0) + 1));
    return c;
  }, [assignedTargets]);

  const setOne = (sourceId: string, targetValue: string | string[]) => {
    onChange({ ...value, [sourceId]: targetValue });
  };

  const toggleMulti = (sourceId: string, t: string) => {
    const currentRaw = value[sourceId] ?? [];
    const current = Array.isArray(currentRaw) ? currentRaw : [currentRaw].filter(Boolean);
    const exists = current.includes(t);

    let next = exists ? current.filter((x) => x !== t) : [...current, t];
    // Ensure stable ordering like targets list
    next = targets.filter((tt) => next.includes(tt));
    setOne(sourceId, next);
  };

  const selectSingle = (sourceId: string, t: string) => {
    setOne(sourceId, t);
  };

  return (
    <div>
      {title ? (
        <div className="form-label" style={{ marginTop: 0 }}>
          {title}
        </div>
      ) : null}
      {hint ? (
        <p className="subtitle" style={{ marginTop: 0 }}>
          {hint}
        </p>
      ) : null}

      <div
        style={{
          border: "1px solid var(--gray-200)",
          borderRadius: 8,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            display: "grid",
            gridTemplateColumns,
            background: "var(--gray-50)",
            padding: "8px 12px",
            fontWeight: 600,
            fontFamily: "Roboto, sans-serif",
          }}
        >
          <div>Source</div>
          <div>{multi ? "Map to (tick one or more)" : "Map to (pick one)"}</div>
        </div>

        {sources.length === 0 ? (
          <div style={{ padding: 12, fontStyle: "italic" }}>No sources yet.</div>
        ) : (
          sources.map((s) => {
            const selectedRaw = value[s.id];
            const selected = (selectedRaw ?? (multi ? [] : "")) as string | string[];
            const selectedList = Array.isArray(selected)
              ? selected
              : [selected].filter(Boolean);

            const query = (queries[s.id] ?? "").trim().toLowerCase();
            const visibleTargets = query
              ? targets.filter((t) => t.toLowerCase().includes(query))
              : targets;

            return (
              <div
                key={s.id}
                style={{
                  display: "grid",
                  gridTemplateColumns,
                  padding: "10px 12px",
                  borderTop: "1px solid var(--gray-100)",
                  alignItems: "start",
                  gap: 12,
                }}
              >
                <div
                  style={{ overflow: "hidden", textOverflow: "ellipsis" }}
                  title={s.label}
                >
                  {s.label}
                </div>

                <div>
                  {searchable && targets.length > 1 ? (
                    <input
                      type="text"
                      value={queries[s.id] ?? ""}
                      onChange={(e) =>
                        setQueries((prev) => ({ ...prev, [s.id]: e.target.value }))
                      }
                      placeholder="Filter…"
                      aria-label="Filter targets"
                      style={{
                        width: "100%",
                        marginBottom: 6,
                        padding: "6px 8px",
                        border: "1px solid var(--gray-200)",
                        borderRadius: 8,
                        fontSize: 13,
                        boxSizing: "border-box",
                      }}
                    />
                  ) : null}
                  <div
                    id={`stm-list-${uid}-${s.id}`}
                    style={{
                      border: "1px solid var(--gray-200)",
                      borderRadius: 8,
                      padding: 8,
                      maxHeight: 260,
                      overflow: "auto",
                      display: "grid",
                      gridTemplateColumns: `repeat(auto-fill, minmax(${targetMinWidth}px, 1fr))`,
                      gap: 6,
                      background: "white",
                    }}
                    aria-label={multi ? "Targets checklist" : "Targets"}
                  >
                    {visibleTargets.length === 0 ? (
                      <div style={{ padding: "4px 4px", fontStyle: "italic", color: "var(--gray-500)" }}>
                        No matches.
                      </div>
                    ) : null}
                    {visibleTargets.map((t, i) => {
                      const name = `stm-single-${uid}-${s.id}`;
                      const inputId = `stm-${uid}-${s.id}-${i}`;
                      const isSelectedHere = !multi && value[s.id] === t;

                      const isTakenElsewhere =
                        !allowDuplicateTargets &&
                        (multi
                          ? !(Array.isArray(value[s.id]) && (value[s.id] as string[]).includes(t))
                          : !isSelectedHere) &&
                        (counts[t] || 0) > 0;

                      return (
                        <div key={`${s.id}-${t}`} data-label={t}>
                          <label
                            htmlFor={inputId}
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: 8,
                              padding: "6px 8px",
                              border: "1px solid var(--gray-200)",
                              borderRadius: 8,
                              opacity: isTakenElsewhere ? 0.5 : 1,
                              cursor: isTakenElsewhere ? "not-allowed" : "pointer",
                              userSelect: "none",
                              width: "100%",
                              minWidth: 0,
                              whiteSpace: wrapLabels ? "normal" : "nowrap",
                              background: "white",
                            }}
                            title={isTakenElsewhere ? "Already assigned to another source" : t}
                          >
                            <input
                              id={inputId}
                              type={multi ? "checkbox" : "radio"}
                              name={multi ? `stm-${uid}-${s.id}` : name}
                              value={t}
                              checked={
                                multi
                                  ? Array.isArray(value[s.id]) &&
                                    (value[s.id] as string[]).includes(t)
                                  : isSelectedHere
                              }
                              disabled={isTakenElsewhere}
                              onChange={() => (multi ? toggleMulti(s.id, t) : selectSingle(s.id, t))}
                              style={{ flex: "0 0 auto", margin: 0 }}
                            />

                            <span
                              style={{
                                flex: 1,
                                minWidth: 0,
                                overflow: "hidden",
                                textOverflow: "ellipsis",
                                wordBreak: wrapLabels ? "break-word" : "normal",
                              }}
                            >
                              {t}
                            </span>

                            {!isTakenElsewhere ? null : (
                              <span
                                style={{
                                  flex: "0 0 auto",
                                  fontSize: 11,
                                  background: "#fee2e2",
                                  color: "#c53030",
                                  padding: "2px 6px",
                                  borderRadius: 6,
                                  border: "1px solid #fecaca",
                                }}
                              >
                                taken
                              </span>
                            )}
                          </label>
                        </div>
                      );
                    })}
                  </div>

                  {!allowDuplicateTargets && multi && selectedList.length > 0 && (
                    <p className="subtitle" style={{ marginTop: 6 }}>
                      Selected: {selectedList.join(", ")}
                    </p>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};

export default SourceTargetMapper;
