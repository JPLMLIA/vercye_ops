import { useState } from "react";

const tokenise = (str: string): string[] =>
  (str || "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

interface MultiValueInputProps {
  label: string;
  placeholder?: string;
  values: string[];
  setValues: React.Dispatch<React.SetStateAction<string[]>>;
  onValueEdit?: (oldVal: string, newVal: string) => void;
}

const Chip: React.FC<{
  text: string;
  onRemove: () => void;
  onEdit?: (newText: string) => void;
}> = ({ text, onRemove, onEdit }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [draft, setDraft] = useState(text);

  const commitEdit = () => {
    if (draft.trim() && draft !== text && onEdit) {
      onEdit(draft.trim());
    }
    setIsEditing(false);
  };

  const enableEditing = () => {
    if (onEdit) {
      setIsEditing(true);
    }
  };

  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        padding: "6px 10px",
        background: "var(--gray-100)",
        color: "var(--gray-700)",
        border: "1px solid var(--gray-200)",
        borderRadius: 8,
        fontSize: 12,
        fontFamily: "Roboto, sans-serif",
        marginRight: 8,
        marginBottom: 8,
      }}
    >
      {isEditing ? (
        <input
          autoFocus
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commitEdit}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              commitEdit();
            } else if (e.key === "Escape") {
              setDraft(text);
              setIsEditing(false);
            }
          }}
          style={{ fontSize: 12, fontFamily: "inherit", padding: "2px 4px", width: "80px" }}
        />
      ) : (
        <span
          onClick={enableEditing}
          style={{ cursor: onEdit ? "pointer" : "default" }}
          title={onEdit ? "Click to edit" : undefined}
        >
          {text}
        </span>
      )}
      <button
        type="button"
        aria-label={`Remove ${text}`}
        onClick={onRemove}
        className="btn btn-danger btn-sm"
        style={{ padding: "4px 8px" }}
      >
        ✕
      </button>
    </span>
  );
};
const MultiValueInput: React.FC<MultiValueInputProps> = ({
  label,
  placeholder,
  values,
  setValues,
  onValueEdit
}) => {
  const [draft, setDraft] = useState("");

  const addDraft = () => {
    const toks = tokenise(draft);
    if (!toks.length) return;
    const next = Array.from(new Set([...(values || []), ...toks]));
    setValues(next);
    setDraft("");
  };

  return (
    <div className="form-group">
      <label className="form-label">{label}</label>
      <div style={{ display: "flex", gap: "0.5rem" }}>
        <input
          className="form-input"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={placeholder || "Type a value and press Add (or paste comma-separated)"}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              addDraft();
            }
          }}
        />
        <button type="button" className="btn btn-secondary" onClick={addDraft}>
          Add
        </button>
      </div>
      <div style={{ marginTop: 8 }}>
        {(values || []).map((v, idx) => (
         <Chip
          key={v}
          text={v}
          onRemove={() => setValues(values.filter((x) => x !== v))}
          onEdit={
            onValueEdit != undefined
              ? (newText) => {
                  const next = [...values];
                  const idx = next.indexOf(v);
                  if (idx !== -1) {
                    next[idx] = newText;
                    setValues(Array.from(new Set(next)));
                    onValueEdit?.(v, newText);
                  }
                }
              : undefined
          }
        />
        ))}
      </div>
    </div>
  );
};

export default MultiValueInput
