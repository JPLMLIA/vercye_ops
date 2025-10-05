import { ChangeEvent } from "react";

interface Props {
  id: string;
  accept?: string;
  multiple?: boolean;
  label: string;
  value: File[];
  onChange: (files: File[]) => void;
}

const FileUpload = ({ id, accept, multiple, label, value, onChange }: Props) => {
  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const newFiles = Array.from(e.target.files);
    onChange(multiple ? [...value, ...newFiles] : newFiles);
    e.target.value = "";
  };

  const removeFile = (index: number) => {
    onChange(value.filter((_, i) => i !== index));
  };

  return (
    <div className={`file-upload ${value.length > 0 ? "has-file" : ""}`} id={`${id}FileUpload`}>
      <div className="file-upload-input-wrapper">
        <input
          id={id}
          type="file"
          accept={accept}
          multiple={multiple}
          onChange={handleChange}
        />
        <div className="file-upload-label">
          <span>{label}</span>
        </div>
      </div>

      {value.length > 0 && (
        <div className="file-list" style={{ marginTop: "10px" }}>
          {value.map((file, idx) => (
            <div key={idx} className="file-chip">
              <span className="file-name">{file.name}</span>
              <button type="button" onClick={() => removeFile(idx)} className="file-remove">
                ✕
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default FileUpload;
