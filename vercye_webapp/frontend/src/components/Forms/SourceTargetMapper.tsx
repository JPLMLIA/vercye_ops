import React, { useMemo } from "react";

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
  multi?: boolean; // default false
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
}) => {
  // Flatten all assigned target values across rows
  const assignedTargets: string[] = useMemo(() => {
    const vals = Object.values(value).filter(
      (v) => v !== "" && v !== undefined
    ) as (string | string[])[];
    return vals.flatMap((v) => (Array.isArray(v) ? v : [v]));
  }, [value]);

  // Compute which targets are duplicated across rows (only if not allowed)
  const duplicates = useMemo(() => {
    if (allowDuplicateTargets) return new Set<string>();
    const counts = assignedTargets.reduce<Record<string, number>>((acc, t) => {
      acc[t] = (acc[t] || 0) + 1;
      return acc;
    }, {});
    const dupes = new Set<string>();
    Object.entries(counts).forEach(([t, c]) => {
      if (c > 1) dupes.add(t);
    });
    return dupes;
  }, [allowDuplicateTargets, assignedTargets]);

  const setOne = (sourceId: string, targetValue: string | string[]) => {
    onChange({ ...value, [sourceId]: targetValue });
  };

  return (
    <div>
      {title ? <h4 style={{ marginTop: 0 }}>{title}</h4> : null}
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
            gridTemplateColumns: "2fr 1.2fr",
            background: "var(--gray-50)",
            padding: "8px 12px",
            fontWeight: 600,
            fontFamily: "Roboto, sans-serif",
          }}
        >
          <div>Source</div>
          <div>{multi ? "Map to (one or more with CTRL + Click)" : "Map to"}</div>
        </div>

        {sources.length === 0 ? (
          <div style={{ padding: 12, fontStyle: "italic" }}>No sources yet.</div>
        ) : (
          sources.map((s) => {
            const selectedRaw = value[s.id];
            const selected = (selectedRaw ??
              (multi ? [] : "")) as string | string[];

            const selectedList = Array.isArray(selected) ? selected : [selected].filter(Boolean);
            const hasDupeConflict =
              !allowDuplicateTargets &&
              selectedList.some((t) => duplicates.has(t));

            return (
              <div
                key={s.id}
                style={{
                  display: "grid",
                  gridTemplateColumns: "2fr 1.2fr",
                  padding: "10px 12px",
                  borderTop: "1px solid var(--gray-100)",
                  alignItems: "center",
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
                  <select
                    className={`form-select${
                      hasDupeConflict ? " is-invalid" : ""
                    }`}
                    multiple={multi}
                    value={
                      multi
                        ? (selected as string[])
                        : ((selected as string) || "")
                    }
                    onChange={(e) => {
                      if (multi) {
                        const opts = Array.from(
                          e.currentTarget.selectedOptions
                        ).map((o) => o.value);
                        setOne(s.id, opts);
                      } else {
                        setOne(s.id, e.currentTarget.value);
                      }
                    }}
                  >
                    {!multi && <option value="">— Select —</option>}
                    {targets.map((t) => (
                      <option key={t} value={t}>
                        {t}
                      </option>
                    ))}
                  </select>
                  {hasDupeConflict && (
                    <div className="form-error" style={{ marginTop: 6 }}>
                      This target is already used elsewhere.
                    </div>
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
