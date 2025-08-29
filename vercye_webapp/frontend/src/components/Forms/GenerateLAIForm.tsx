import { useState } from "react";
import Fieldset from "@/components/Forms/FieldSet";
import FileUpload from "../FileUpload";

export interface GenerateLAIPayload {
  shapefile: File;
  resolution: number;
  keep_imagery: boolean;
  name: string;
  date_ranges: { start_date: string; end_date: string }[];
  imagery_src: "MPC" | "ES_S2C1";
  chunk_days: number;
}

interface GenerateLAIFormProps {
  onSubmit: (payload: GenerateLAIPayload) => void;
}

const GenerateLAIForm: React.FC<GenerateLAIFormProps> = ({ onSubmit }) => {
  const [name, setName] = useState("");
  const [resolution, setResolution] = useState<number>(10);
  const [keepImagery, setKeepImagery] = useState(false);
  const [imagerySrc, setImagerySrc] = useState<"MPC" | "ES_S2C1">("MPC");
  const [chunkDays, setChunkDays] = useState<number>(30);
  const [shapefile, setShapefile] = useState<File | null>(null);
  const [dateRanges, setDateRanges] = useState<{ start_date: string; end_date: string }[]>([
    { start_date: "", end_date: "" }
  ]);

  const updateDateRange = (idx: number, field: "start_date" | "end_date", value: string) => {
    setDateRanges((prev) =>
      prev.map((dr, i) => (i === idx ? { ...dr, [field]: value } : dr))
    );
  };

  const addDateRange = () => {
    setDateRanges((prev) => [...prev, { start_date: "", end_date: "" }]);
  };

  const deleteDateRange = (idx: number) => {
    setDateRanges((prev) => prev.filter((_, i) => i !== idx));
  };

  const handleSubmit = () => {
    if (!shapefile) {
      alert("Please upload a shapefile before submitting.");
      return;
    }
    const payload: GenerateLAIPayload = {
      shapefile,
      resolution,
      keep_imagery: keepImagery,
      name,
      date_ranges: dateRanges,
      imagery_src: imagerySrc,
      chunk_days: chunkDays
    };
    onSubmit(payload);
  };

  return (
    <div className="container">
      <Fieldset legend="General Settings">
        <div className="form-group">
          <label className="form-label">Name</label>
          <input
            className="form-input"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Enter LAI generation name"
          />
          <p className="subtitle" style={{ marginTop: 4 }}>
            Descriptive and unique name for your region of interest.
          </p>
        </div>

        <div className="form-group">
          <label className="form-label">Resolution (m)</label>
          <input
            type="number"
            className="form-input"
            value={resolution}
            onChange={(e) => setResolution(Number(e.target.value))}
          />
          <p className="subtitle" style={{ marginTop: 4 }}>
            Resolution in meters (UTM). Typically 10 or 20m. 10m uses a dedicated model, all other LAI is generated with the 20m model.
          </p>
        </div>

        <div className="form-group">
          <label className="form-label">Zipped Shapefile (.zip)</label>
          <FileUpload
            id="shapefileFile"
            accept=".geojson"
            label="📁 Choose shapefile (.geojson)"
            onChange={(files) => setShapefile(files?.[0] ?? null)}
          />
          <p className="subtitle" style={{ marginTop: 4 }}>
            Shapefile specifying the Region of Interest. If it contains multiple geometries, will try to download all intersecting tiles. However, if too many small geometries create a too large request, might use the bounding box.
          </p>
        </div>
      </Fieldset>

      <Fieldset legend="Date Ranges">
        <p className="subtitle" style={{ marginTop: 4 }}>
          Date ranges for which to create LAI data. Date ranges are processed sequentially.
          Should be as small as possible to avoid downloading unnecessary data.
        </p>
        {dateRanges.map((dr, idx) => (
          <div
            key={idx}
            style={{
              display: "grid",
              gap: "0.75rem",
              gridTemplateColumns: "1fr 1fr auto",
              alignItems: "center",
              marginTop: "1rem"
            }}
          >
            <div className="form-group">
              <label className="form-label">Start Date</label>
              <input
                type="date"
                className="form-input"
                value={dr.start_date}
                onChange={(e) => updateDateRange(idx, "start_date", e.target.value)}
              />
            </div>
            <div className="form-group">
              <label className="form-label">End Date</label>
              <input
                type="date"
                className="form-input"
                value={dr.end_date}
                onChange={(e) => updateDateRange(idx, "end_date", e.target.value)}
              />
            </div>
            <button
              type="button"
              className="btn btn-danger"
              onClick={() => deleteDateRange(idx)}
            >
              ✕
            </button>
          </div>
        ))}
        <button className="btn btn-secondary" onClick={addDateRange} type="button">
          + Add Date Range
        </button>
      </Fieldset>

      <Fieldset legend="Advanced Options">
        <div className="form-group">
          <label className="form-label">Keep Original Imagery</label>
          <input
            type="checkbox"
            checked={keepImagery}
            onChange={(e) => setKeepImagery(e.target.checked)}
          />
          <p className="subtitle" style={{ marginTop: 4 }}>
            Specifies whether original Sentinel-2 scenes should be kept in storage after generating LAI. Should be turned off to save storage space.
          </p>
        </div>

        <div className="form-group">
          <label className="form-label">Imagery Source</label>
          <select
            className="form-input"
            value={imagerySrc}
            onChange={(e) => setImagerySrc(e.target.value as "MPC" | "ES_S2C1")}
          >
            <option value="MPC">Microsoft Planetary Computer Sentinel-2 L2A</option>
            <option value="ES_S2C1">Element84 EarthSearch Sentinel-2 L2A Collection-1</option>
          </select>
          <p className="subtitle" style={{ marginTop: 4 }}>
            Choose the imagery source.
          </p>
        </div>

        <div className="form-group">
          <label className="form-label">Chunk Days</label>
          <input
            type="number"
            className="form-input"
            value={chunkDays}
            onChange={(e) => setChunkDays(Number(e.target.value))}
          />
           <p className="subtitle" style={{ marginTop: 4 }}>
             Number of days to download per batch. Defines max storage requirements at any time.
            </p>
        </div>
      </Fieldset>

      <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.75rem" }}>
        <button
          className="btn btn-primary"
          onClick={handleSubmit}
        >
          Create LAI
        </button>
      </div>
    </div>
  );
};

export default GenerateLAIForm;
