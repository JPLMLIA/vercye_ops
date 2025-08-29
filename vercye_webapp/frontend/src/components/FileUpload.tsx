
import { useState, ChangeEvent } from 'react';

interface Props {
  id: string;
  accept?: string;
  multiple?: boolean;
  label: string;
  onChange: (files: FileList | null) => void;
}

const FileUpload = ({ id, accept, multiple, label, onChange }: Props) => {
  const [hasFile, setHasFile] = useState(false);
  const [text, setText] = useState(label);

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const names = Array.from(files).map((f) => f.name);
      const shown = names.slice(0, 2).join(', ');
      const more = names.length > 2 ? ` +${names.length - 2} more` : '';
      setText(`${shown}${more}`);
      setHasFile(true);
    } else {
      setText(label);
      setHasFile(false);
    }
    onChange(files);
  }

  return (
    <div className={`file-upload ${hasFile ? 'has-file' : ''}`} id={`${id}FileUpload`}>
      <input id={id} type="file" accept={accept} multiple={multiple} onChange={handleChange} />
      <div className="file-upload-label">{text}</div>
    </div>
  );
}

export default FileUpload
