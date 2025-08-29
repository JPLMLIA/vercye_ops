import React, { useMemo, useState, ReactNode, useEffect } from "react";
import FileUpload from "../FileUpload";
import Fieldset from "./FieldSet";
import MultiValueInput from "./MultiValueInput";
import SourceTargetMapper, { MapperSource, MappingState } from "./SourceTargetMapper";
import shp from 'shpjs';

interface WindowConfig {
  id: string;
  year: string;
  timepoint: string;
  sim_start_date: string;
  sim_end_date: string;
  met_start_date: string;
  met_end_date: string;
  lai_start_date: string;
  lai_end_date: string;
}

interface ShapefileData {
  type: string;
  features: Array<{
    geometry: any;
    properties: Record<string, any>;
  }>;
}

interface SetupStudyFormProps {
  onSubmit: (payload: SetupSubmissionsPayload) => void;
}

export interface SetupSubmissionsPayload {
  shapefile: File;
  regionExtraction: {
    adminNameColumn: string;
    targetProjection: string;
    filter?: {
      column: string;
      allow: string[];
    };
  };
  apsimFiles: File[];
  apsimColumn: string;
  apsimMapping: MappingState;
  referenceFiles: File[];
  referenceMapping: MappingState;
  referenceYearsMapping: MappingState;
  years: string[];
  timepoints: string[];
  simulationWindows: Omit<WindowConfig, "id">[];
}

const initWindow = (override: Partial<WindowConfig> = {}): WindowConfig => ({
  id: crypto?.randomUUID?.() || String(Math.random()).slice(2),
  year: "",
  timepoint: "",
  sim_start_date: "",
  sim_end_date: "",
  met_start_date: "",
  met_end_date: "",
  lai_start_date: "",
  lai_end_date: "",
  ...override,
});

const coerceFilesFromOnChange = (
  eOrFiles: any
): File[] => {
  if (Array.isArray(eOrFiles)) return eOrFiles as File[];
  if (eOrFiles?.target?.files) return Array.from(eOrFiles.target.files) as File[];
  if (eOrFiles instanceof FileList) return Array.from(eOrFiles);
  return [];
};

const pairKey = (y: string, t: string) => `${y}__${t}`;

const buildCartesianWindows = (
  years: string[],
  timepoints: string[],
  prev: WindowConfig[]
): WindowConfig[] => {
  const prevByKey = new Map<string, WindowConfig>();
  prev.forEach(w => prevByKey.set(pairKey(w.year, w.timepoint), w));

  const next: WindowConfig[] = [];
  for (const y of years) {
    for (const t of timepoints) {
      const k = pairKey(y, t);
      const existing = prevByKey.get(k);
      next.push(existing ?? initWindow({ year: y, timepoint: t }));
    }
  }
  return next;
};

function loadColValues(column: string, _shapefileFile?: File | null): string[] {
  if (!column) return ["all"];

  // Expect the parsed GeoJSON to be set during handleFileUpload:
  // (window as any).__lastShapefileGeoJSON = geojson;
  const geojson = (window as any).__lastShapefileGeoJSON as
    | { features?: Array<{ properties?: Record<string, unknown> }> }
    | undefined;

  const features = geojson?.features;
  if (!features || features.length === 0) return ["all"];

  const values = Array.from(
    new Set(
      features
        .map((f) => f.properties?.[column])
        .filter((v): v is string | number => v !== undefined && v !== null && v !== "")
        .map((v) => String(v))
    )
  ).sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" }));

  return ["all", ...values];
}

const SetupStudyForm: React.FC<SetupStudyFormProps> = ({ onSubmit }) => {
  const [shapefileUpload, setShapefileUpload] = useState<File | null>(null);
  const [apsimFiles, setApsimFiles] = useState<File[]>([]);
  const [apsimAllowedValues, setApsimAllowedValues] = useState<string[]>(['all']);
  const [apsimColumn, setApsimColumn] = useState("");

  const [referenceFiles, setReferenceFiles] = useState<File[]>([]);

  const [adminNameColumn, setAdminNameColumn] = useState("");
  const [targetProjection, setTargetProjection] = useState("");
  const [regionFilterColumn, setRegionFilterColumn] = useState("");
  const [regionAllowedValues, setRegionAllowedValues] = useState<string[]>([]);

  const [detectedColumns, setDetectedColumns] = useState<string[]>([]);

  const [years, setYears] = useState<string[]>([])
  const [timepoints, setTimepoints] = useState<string[]>([])
  const [windows, setWindows] = useState<WindowConfig[]>([])

  const updateWindow = (id: string, patch: Partial<WindowConfig>) =>
    setWindows((prev) => prev.map((w) => (w.id === id ? { ...w, ...patch } : w)));

  // APSIM mapping: each APSIM file -> one target from regionAllowedValues
  const apsimSources: MapperSource[] = useMemo(
    () => apsimFiles.map((f, i) => ({ id: f.name, label: f.name, file: f })),
    [apsimFiles]
  );
  const [apsimMapping, setApsimMapping] = useState<MappingState>({});

  // Reference CSV mapping: each CSV -> one target from aggregationLevels and year
  const [aggregationLevels, setAggregationLevels] = useState<string[]>(['primary']);
  const refSources: MapperSource[] = useMemo(
    () => referenceFiles.map((f, i) => ({ id: f.name, label: f.name, file: f })),
    [referenceFiles]
  );
  const [referenceMapping, setReferenceMapping] = useState<MappingState>({});
  const [referenceYearsMapping, setReferenceYearsMapping] = useState<MappingState>({})

  // Keep mapping keys in sync if files change (cleanup removed sources)
 useEffect(() => {
  setApsimMapping((prev) => {
    const next: MappingState = {};
    apsimSources.forEach((s) => (next[s.id] = (prev[s.id] ?? []) as string[]));
    return next;
  });
}, [apsimSources]);

  useEffect(() => {
    setReferenceMapping((prev) => {
      const next: MappingState = {};
      refSources.forEach((s) => (next[s.id] = prev[s.id] || ""));
      return next;
    });
  }, [refSources]);

  useEffect(() => {
    setWindows(prev => buildCartesianWindows(years, timepoints, prev));
  }, [years, timepoints]);

 useEffect(() => {
  setApsimAllowedValues(loadColValues(apsimColumn, shapefileUpload));
}, [apsimColumn, shapefileUpload]);

  // ===== Handlers =====
  const buildPayload = (): SetupSubmissionsPayload => {
    if (!shapefileUpload) throw new Error("Shapefile is required before building the payload.");

    const maybeFilter =
      regionFilterColumn && regionAllowedValues?.length
        ? { column: regionFilterColumn, allow: regionAllowedValues }
        : undefined;

    let finalApsimMapping = apsimMapping
    if (apsimFiles.length === 1) {
      finalApsimMapping = {
        [apsimFiles[0].name]: ["all"]
      };
    }

    return {
      shapefile: shapefileUpload,
      regionExtraction: {
        adminNameColumn,
        targetProjection,
        ...(maybeFilter ? { filter: maybeFilter } : {}),
      },
      apsimFiles,
      apsimColumn,
      apsimMapping: finalApsimMapping,
      referenceFiles,
      referenceMapping,
      referenceYearsMapping,
      years,
      timepoints,
      simulationWindows: windows.map(({ id, ...rest }) => rest),
    };
  };

  const handleFileUpload = async (files: FileList | null) => {
    const file = files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.zip')) {
        alert('Please upload a ZIP file containing shapefile components (.shp, .shx, .dbf, etc.)');
        return;
    }

    try {
        setShapefileUpload(file);
      const arrayBuffer = await file.arrayBuffer();
    const geojson = await shp(arrayBuffer) as ShapefileData;
    (window as any).__lastShapefileGeoJSON = geojson;

      // Get all unique column names from all features
      const allColumnNames = getAllColumnNames(geojson);

      if (allColumnNames.length > 0) {
        setDetectedColumns(allColumnNames);
        setApsimAllowedValues(loadColValues(apsimColumn, file));
      } else {
        alert('No attribute columns found in shapefile');
      }
    } catch (err) {
      alert(`Error parsing shapefile ZIP: ${err instanceof Error ? err.message : 'Unknown error'}`);
    }
  };

   const getAllColumnNames = (geojson: ShapefileData): string[] => {
    const allColumns = new Set<string>();

    if (!geojson.features || geojson.features.length === 0) {
      return [];
    }

    // Collect all unique property keys from all features
    // This is important because some features might have different attributes
    geojson.features.forEach(feature => {
      if (feature.properties) {
        Object.keys(feature.properties).forEach(key => {
          allColumns.add(key);
        });
      }
    });

    return Array.from(allColumns).sort(); // Sort alphabetically for better UX
  };


  return (
    <div className="container">
      <div className="alert alert-info" style={{ marginTop: "1rem" }}>
        <strong>Step 2:</strong> Setup your Yieldstudy
        <br />
        This will create your initial study directory structure by extracting individual geometries,
        preparing APSIM files etc.
      </div>

      {/* ========================= Region Extraction ========================= */}
      <Fieldset
        legend="Region Extraction"
        hint="Choose your shapefile source, projection, and optional filter to restrict extracted features."
      >
        <div className="form-group">
        <label className="form-label">Zipped Shapefile (.zip)</label>
        <FileUpload id="shapefileFile" accept=".zip" label="📁 Choose zipped shapefile (.zip)" onChange={handleFileUpload} />
    </div>

        <div
          style={{
            display: "grid",
            gap: "1rem",
            gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
          }}
        >
          <div className="form-group">
            <label className="form-label">Admin Name Column</label>
            <input
              list="detectedColumns"
              className="form-input"
              placeholder="e.g., admin_name or ADM2_EN"
              value={adminNameColumn}
              onChange={(e) => setAdminNameColumn(e.target.value)}
            />
            <datalist id="detectedColumns">
              {detectedColumns.map((c) => (
                <option key={c} value={c} />
              ))}
            </datalist>
            <p className="subtitle" style={{ marginTop: 4 }}>
              Column in the shapefile that contains the administrative name.
            </p>
          </div>

          <div className="form-group">
            <label className="form-label">Target Projection (EPSG or proj string)</label>
            <input
              className="form-input"
              placeholder="EPSG:4326 or +proj=longlat +datum=WGS84 +no_defs"
              value={targetProjection}
              onChange={(e) => setTargetProjection(e.target.value)}
            />
            <p className="subtitle" style={{ marginTop: 4 }}>
              Should be a local equal area projection (e.g Albers)
            </p>
          </div>
        </div>

        <div
          style={{
            display: "grid",
            gap: "1rem",
            gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
            marginTop: "0.5rem",
          }}
        >

          <div className="form-group">
            <label className="form-label">Optional: Filter Column</label>
            <input
              list="detectedColumns"
              className="form-input"
              placeholder="e.g., admin_name or ADM2_EN"
              value={regionFilterColumn}
              onChange={(e) => setRegionFilterColumn(e.target.value)}
            />
            <datalist id="detectedColumns">
              {detectedColumns.map((c) => (
                <option key={c} value={c} />
              ))}
            </datalist>
            <p className="subtitle" style={{ marginTop: 4 }}>
              Only features whose value is in the allow-list (next field) will be extracted. Keep empty to use all.
            </p>
          </div>

          <MultiValueInput
            label="Optional: Allowed Filter Values"
            placeholder="Type a value and press Add, or paste comma-separated"
            values={regionAllowedValues}
            setValues={setRegionAllowedValues}
          />
        </div>
      </Fieldset>

      {/* ========================= Simulation Windows ========================= */}
      <Fieldset
        legend="Simulation Windows (Years & Timepoints)"
        hint="Add years and timepoints and fill in the required dates. These will be injected into APSIM and data fetchers."
      >
        <MultiValueInput
            label="Years (must be numerical e.g 2019)"
            placeholder="2019, 2020"
            values={years}
            setValues={setYears}
        />
        <MultiValueInput
            label="Timepoints (e.g T-0)."
            placeholder="T-0, T-30"
            values={timepoints}
            setValues={setTimepoints}
        />

        {windows.map((w, idx) => (
          <section
            key={w.id}
            style={{
              border: "1px solid var(--gray-200)",
              borderRadius: 10,
              padding: "1rem",
              marginBottom: "1rem",
              background: "var(--gray-50)",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: "0.75rem",
              }}
            >
              <strong style={{ color: "var(--dark-primary)", fontFamily: "Roboto, sans-serif" }}>
                {w.year} {w.timepoint}
              </strong>
            </div>

            <div
              style={{
                display: "grid",
                gap: "1rem",
                gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
              }}
            >
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">sim_start_date</label>
                <input
                  className="form-input"
                  type="date"
                  value={w.sim_start_date}
                  onChange={(e) => updateWindow(w.id, { sim_start_date: e.target.value })}
                />
              </div>

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">sim_end_date</label>
                <input
                  className="form-input"
                  type="date"
                  value={w.sim_end_date}
                  onChange={(e) => updateWindow(w.id, { sim_end_date: e.target.value })}
                />
              </div>

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">met_start_date</label>
                <input
                  className="form-input"
                  type="date"
                  value={w.met_start_date}
                  onChange={(e) => updateWindow(w.id, { met_start_date: e.target.value })}
                />
              </div>

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">met_end_date</label>
                <input
                  className="form-input"
                  type="date"
                  value={w.met_end_date}
                  onChange={(e) => updateWindow(w.id, { met_end_date: e.target.value })}
                />
              </div>

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">lai_start_date</label>
                <input
                  className="form-input"
                  type="date"
                  value={w.lai_start_date}
                  onChange={(e) => updateWindow(w.id, { lai_start_date: e.target.value })}
                />
              </div>

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">lai_end_date</label>
                <input
                  className="form-input"
                  type="date"
                  value={w.lai_end_date}
                  onChange={(e) => updateWindow(w.id, { lai_end_date: e.target.value })}
                />
              </div>
            </div>
          </section>
        ))}
      </Fieldset>

      {/* ========================= APSIM Files ========================= */}
      <Fieldset
        legend="APSIM Files"
        hint="Upload APSIM files and map each one to a region value (from the Region Extraction allow-list)."
      >
        <div className="form-group">
          <label className="form-label">APSIM File(s) (.apsimx)</label>
          <FileUpload
            id="apsimFile"
            accept=".apsimx"
            multiple
            label="📁 Choose one or more APSIM files (.apsimx)"
            onChange={(arg: any) => {
              const files = coerceFilesFromOnChange(arg);
              setApsimFiles(files);
            }}
          />
        </div>

        {apsimFiles.length > 1 ? (<><div className="form-group">
            <label className="form-label">Column to use for assignment</label>
            <input
              list="detectedColumns"
              className="form-input"
              placeholder="e.g., admin_name or ADM2_EN"
              value={apsimColumn}
              onChange={(e) => setApsimColumn(e.target.value)}
            />
            <datalist id="detectedColumns">
              {detectedColumns.map((c) => (
                <option key={c} value={c} />
              ))}
            </datalist>
            <p className="subtitle" style={{ marginTop: 4 }}>
              Field in the shapefile that contains the administrative name.
            </p>
          </div>


        <SourceTargetMapper
          title="Map APSIM to Region Value"
          hint="For each APSIM file, choose a target value from the the specified column."
          sources={apsimSources}
          targets={apsimAllowedValues}
          value={apsimMapping}
          onChange={setApsimMapping}
          allowDuplicateTargets={true}
            multi={true}
        /></>) : ''}
      </Fieldset>

      {/* ========================= Reference Data ========================= */}
      <Fieldset
        legend="Reference Data"
        hint="Upload CSVs and map each file to an aggregation level."
      >
        <div className="form-group">
          <label className="form-label">Reference Data (.csv)</label>
          <FileUpload
            id="referenceFile"
            accept=".csv"
            multiple
            label="📁 Choose one or more Referencedata files (.csv)"
            onChange={(arg: any) => {
              const files = coerceFilesFromOnChange(arg);
              setReferenceFiles(files);
            }}
          />
        </div>

        <MultiValueInput
          label="Aggregation levels"
          placeholder="Specify levels at which to aggregate your data. Free choice of names. Primary is used for simulation level aggregation."
          values={aggregationLevels}
          setValues={setAggregationLevels}
        />

        <SourceTargetMapper
          title="Map Reference CSV to Aggregation Level"
          hint="For each CSV, choose an aggregation level specified above. If the aggregation level is the simulation level, choose primary."
          sources={refSources}
          targets={aggregationLevels}
          value={referenceMapping}
          onChange={setReferenceMapping}
          allowDuplicateTargets={true}
        />

        <SourceTargetMapper
          title="Map Reference CSV to Years"
          hint="For each CSV, choose a year it belongs to."
          sources={refSources}
          targets={years}
          value={referenceYearsMapping}
          onChange={setReferenceYearsMapping}
          allowDuplicateTargets={true}
        />
      </Fieldset>

      <div style={{ display: "flex", gap: "0.75rem", justifyContent: "space-between" }}>
        <div></div>
        <div style={{ display: "flex", gap: ".5rem" }}>
          <button className="btn btn-primary" onClick={() => {
              try {
                const payload = buildPayload();
                onSubmit(payload);
              } catch (err) {
                alert((err as Error).message);
              }
            }}>
            Upload Setup Config
          </button>
        </div>
      </div>
    </div>
  );
};

export default SetupStudyForm;
