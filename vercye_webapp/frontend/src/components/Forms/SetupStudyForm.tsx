import React, { useMemo, useState, ReactNode, useEffect } from "react";
import FileUpload from "../FileUpload";
import Fieldset from "./FieldSet";
import MultiValueInput from "./MultiValueInput";
import SourceTargetMapper, { MapperSource, MappingState } from "./SourceTargetMapper";
import shp from 'shpjs';
import useToast from '@/components/Toast';
import { SetupConfigTemplate } from "@/types";

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
  initialData: SetupConfigTemplate | null
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

interface TimepointPattern {
  sim_start_day: number;
  sim_start_month: number;
  sim_start_year_offset: number;
  sim_end_day: number;
  sim_end_month: number;
  sim_end_year_offset: number;
  lai_start_day: number;
  lai_start_month: number;
  lai_start_year_offset: number;
  lai_end_day: number;
  lai_end_month: number;
  lai_end_year_offset: number;
  met_start_day: number;
  met_start_month: number;
  met_start_year_offset: number;
  met_end_day: number;
  met_end_month: number;
  met_end_year_offset: number;
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

const defaultPattern: TimepointPattern = {
  sim_start_day: 20,
  sim_start_month: 8,
  sim_start_year_offset: -1,
  sim_end_day: 1,
  sim_end_month: 8,
  sim_end_year_offset: 0,
  lai_start_day: 1,
  lai_start_month: 2,
  lai_start_year_offset: 0,
  lai_end_day: 1,
  lai_end_month: 8,
  lai_end_year_offset: 0,
  met_start_day: 1,
  met_start_month: 8,
  met_start_year_offset: -20,
  met_end_day: 1,
  met_end_month: 8,
  met_end_year_offset: 0,
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
  if (!column) return [];

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

  return values;
}

const SetupStudyForm: React.FC<SetupStudyFormProps> = ({ onSubmit, initialData }) => {
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

  const [isInitialized, setIsInitialized] = useState(false)
  const [usePattern, setUsePattern] = useState(false)
  const [timepointPatterns, setTimepointPatterns] = useState<{[tp: string]: TimepointPattern}>({})

  const { show, Toast } = useToast();

  const updateWindow = (id: string, patch: Partial<WindowConfig>) =>
    setWindows((prev) => prev.map((w) => (w.id === id ? { ...w, ...patch } : w)));

  useEffect(() => {
    if (!initialData) {setIsInitialized(true); return}
    (window as any).__lastShapefileGeoJSON = initialData.shapefileData;
    setShapefileUpload(new File([], initialData.shapefileName))

    const detectedColumns = getAllColumnNames(initialData.shapefileData)
    setDetectedColumns(detectedColumns);
    setApsimAllowedValues(loadColValues(initialData.apsimColumn, null));

    setAdminNameColumn(initialData.regionExtraction.adminNameColumn);
    setTargetProjection(initialData.regionExtraction.targetProjection);
    setRegionFilterColumn(initialData.regionExtraction.filter?.column ?? "");
    setRegionAllowedValues(initialData.regionExtraction.filter?.allow ?? []);

    setApsimColumn(initialData.apsimColumn || "");
    setApsimMapping(initialData.apsimMapping || {});
    setApsimFiles(initialData.apsimFiles.map((name) => new File([], name)));

    setAggregationLevels([...new Set(Object.values(initialData.referenceMapping))]);
    setReferenceMapping(initialData.referenceMapping || {});
    setReferenceYearsMapping(initialData.referenceYearsMapping || {});
    setReferenceFiles(initialData.referenceFiles.map((name) => new File([], name)));

    setYears(initialData.years);
    setTimepoints(initialData.timepoints);

    const newWindows = initialData.simulationWindows.map((w) => initWindow({...w }));
    setWindows(newWindows);
    setIsInitialized(true)
  }, [initialData]);

  // Keep mapping keys in sync if files change (cleanup removed sources)
  useEffect(() => {
    if (!isInitialized) return
    setApsimMapping((prev) => {
      const next: MappingState = {};
      apsimSources.forEach((s) => (next[s.id] = (prev[s.id] ?? []) as string[]));
      return next;
    });
  }, [apsimSources, isInitialized]);

  // Keep reference mapping in sync when reference files change (cleanup removed sources)
  useEffect(() => {
    if (!isInitialized) return;
    setReferenceMapping((prev) => {
      const next: MappingState = {};
      refSources.forEach((s) => (next[s.id] = (prev[s.id] ?? "") as string));
      return next;
    });
  }, [refSources, isInitialized]);

  useEffect(() => {
    if (!isInitialized) return;
    setReferenceYearsMapping(prev => {
      const next: MappingState = {};
      refSources.forEach(s => {
        const v = prev[s.id];
        // keep only if still a valid year, otherwise reset
        next[s.id] = (typeof v === 'string' && years.includes(v)) ? v : "";
      });
      return next;
    });
  }, [refSources, years, isInitialized]);

  useEffect(() => {
    if (!isInitialized) return
    setWindows(prev => buildCartesianWindows(years, timepoints, prev));
  }, [years, timepoints, isInitialized]);

  useEffect(() => {
    if (!isInitialized) return
    setApsimAllowedValues(loadColValues(apsimColumn, shapefileUpload));
  }, [apsimColumn, shapefileUpload, isInitialized]);

  useEffect(() => {
    if (!isInitialized) return;
    if(apsimFiles.length <= 1) {
      setApsimColumn("")
    }
  }, [apsimFiles])

  useEffect(() => {
    setTimepointPatterns(prev => {
      const next = {...prev};
      timepoints.forEach(tp => {
        if (!next[tp]) {
          next[tp] = {...defaultPattern};
        }
      });
      return next;
    });
  }, [timepoints]);

useEffect(() => {
  if (!isInitialized) return;

  setWindows((prev) => {
    const next: WindowConfig[] = [];

    for (const y of years) {
      for (const t of timepoints) {
        // Either use the existing one (to preserve user edits), or create new
        const existing = prev.find((w) => w.year === y && w.timepoint === t);

        // If pattern usage is enabled and we have a pattern for this timepoint, build dates
        if (usePattern && timepointPatterns[t]) {
          const pattern = timepointPatterns[t];
          const yearNum = Number(y);

          const formatDate = (day: number, month: number, yearOffset: number) => {
            // pad to yyyy-mm-dd
            const yyyy = String(yearNum + yearOffset).padStart(4, "0");
            const mm = String(month).padStart(2, "0");
            const dd = String(day).padStart(2, "0");
            return `${yyyy}-${mm}-${dd}`;
          };

          next.push(
            existing
              ? {
                  ...existing,
                  sim_start_date: formatDate(
                    pattern.sim_start_day,
                    pattern.sim_start_month,
                    pattern.sim_start_year_offset
                  ),
                  sim_end_date: formatDate(
                    pattern.sim_end_day,
                    pattern.sim_end_month,
                    pattern.sim_end_year_offset
                  ),
                  met_start_date: formatDate(
                    pattern.met_start_day,
                    pattern.met_start_month,
                    pattern.met_start_year_offset
                  ),
                  met_end_date: formatDate(
                    pattern.met_end_day,
                    pattern.met_end_month,
                    pattern.met_end_year_offset
                  ),
                  lai_start_date: formatDate(
                    pattern.lai_start_day,
                    pattern.lai_start_month,
                    pattern.lai_start_year_offset
                  ),
                  lai_end_date: formatDate(
                    pattern.lai_end_day,
                    pattern.lai_end_month,
                    pattern.lai_end_year_offset
                  ),
                }
              : initWindow({
                  year: y,
                  timepoint: t,
                  sim_start_date: formatDate(
                    pattern.sim_start_day,
                    pattern.sim_start_month,
                    pattern.sim_start_year_offset
                  ),
                  sim_end_date: formatDate(
                    pattern.sim_end_day,
                    pattern.sim_end_month,
                    pattern.sim_end_year_offset
                  ),
                  met_start_date: formatDate(
                    pattern.met_start_day,
                    pattern.met_start_month,
                    pattern.met_start_year_offset
                  ),
                  met_end_date: formatDate(
                    pattern.met_end_day,
                    pattern.met_end_month,
                    pattern.met_end_year_offset
                  ),
                  lai_start_date: formatDate(
                    pattern.lai_start_day,
                    pattern.lai_start_month,
                    pattern.lai_start_year_offset
                  ),
                  lai_end_date: formatDate(
                    pattern.lai_end_day,
                    pattern.lai_end_month,
                    pattern.lai_end_year_offset
                  ),
                })
          );
        } else {
          // if pattern is not enabled, fall back to keeping existing or blank init
          next.push(existing ?? initWindow({ year: y, timepoint: t }));
        }
      }
    }

    return next;
  });
}, [years, timepoints, timepointPatterns, usePattern, isInitialized]);

  const buildPayload = (): SetupSubmissionsPayload => {
    if (!shapefileUpload) throw new Error("Shapefile is required!.");
    if (!targetProjection) throw new Error("Target projection is required!");
    if (!adminNameColumn) throw new Error("Admin Name Column is required!");
    if (apsimFiles.length < 1) throw new Error("Must upload at least one APSIM file");
    if (years.length < 1 || timepoints.length < 1)
      throw new Error("Must at least specify one year and timepoint. Ensure to press add!");

    // region filter consistency
    if (regionFilterColumn && (!regionAllowedValues || regionAllowedValues.length === 0)) {
      throw new Error("Region allowed values must be set when region filter column is specified!");
    }

    // apsim mapping
    let finalApsimMapping = apsimMapping;
    if (apsimFiles.length === 1) {
      finalApsimMapping = {
        [apsimFiles[0].name]: ["all"],
      };
    } else {
      // ensure every file has a mapping
      for (const file of apsimFiles) {
        if (!finalApsimMapping[file.name] || finalApsimMapping[file.name].length === 0) {
          throw new Error(`APSIM mapping missing for file: ${file.name}`);
        }
      }
      // ensure no duplicate values across mappings
      const seen = new Set<string>();
      for (const [file, values] of Object.entries(finalApsimMapping)) {
        for (const v of values) {
          if (seen.has(v)) {
            throw new Error(`Duplicate APSIM mapping value detected: "${v}" (in file ${file})`);
          }
          seen.add(v);
        }
      }
    }

    // validate simulation windows
    windows.forEach((win, idx) => {
      const requiredFields: (keyof Omit<WindowConfig, "id">)[] = [
        "year",
        "timepoint",
        "sim_start_date",
        "sim_end_date",
        "met_start_date",
        "met_end_date",
        "lai_start_date",
        "lai_end_date",
      ];
      requiredFields.forEach((field) => {
        if (!win[field]) {
          throw new Error(`Window ${idx + 1} is missing required field: ${field}`);
        }
      });
    });

    const maybeFilter =
      regionFilterColumn && regionAllowedValues?.length
        ? { column: regionFilterColumn, allow: regionAllowedValues }
        : undefined;

    if (referenceFiles.length > 0) {
      for (const f of referenceFiles) {
        const y = referenceYearsMapping[f.name];
        if (!y || typeof y !== 'string' || !years.includes(y)) {
          throw new Error(`Please select a valid year for reference file: ${f.name}`);
        }
      }
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

  const handleFileUpload = async (files: File[]) => {
    const file = files[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".zip")) {
      show(
        "Please upload a ZIP file containing shapefile components (.shp, .shx, .dbf, etc.)",
        "error"
      );
      return;
    }

    try {
      setShapefileUpload(file);
      const arrayBuffer = await file.arrayBuffer();
      const geojson = (await shp(arrayBuffer)) as ShapefileData;
      (window as any).__lastShapefileGeoJSON = geojson;

      const allColumnNames = getAllColumnNames(geojson);

      if (allColumnNames.length > 0) {
        setDetectedColumns(allColumnNames);
        setApsimAllowedValues(loadColValues(apsimColumn, file));
      } else {
        show("No attribute columns found in shapefile", "error");
      }
    } catch (err) {
      show(
        `Error parsing shapefile ZIP: ${
          err instanceof Error ? err.message : "Unknown error"
        }`,
        "error"
      );
    }
  };


  const getAllColumnNames = (geojson: ShapefileData): string[] => {
    const allColumns = new Set<string>();

    if (!geojson.features || geojson.features.length === 0) {
      return [];
    }

    geojson.features.forEach(feature => {
      if (feature.properties) {
        Object.keys(feature.properties).forEach(key => {
          allColumns.add(key);
        });
      }
    });

    return Array.from(allColumns).sort();
  };

  const migrateWindows = (
    oldValue: string,
    newValue: string,
  ) => {
    setWindows((prev) =>
      prev.map((w) => {
        if (w.year === oldValue) {
          return { ...w, year: newValue };
        }
        return w;
      })
    );
  };

  const updateTimepointPattern = (
    timepoint: string,
    key: keyof TimepointPattern,
    value: number
  ) => {
    setTimepointPatterns({
      ...timepointPatterns,
      [timepoint]: {
        ...timepointPatterns[timepoint],
        [key]: value
      }
    })
  }



  return (
    <div className="container">
      <div className="alert alert-info" style={{ marginTop: "1rem" }}>
        <strong>Step 2:</strong> Setup your Yieldstudy
        <br />
        This will create your initial study directory structure by extracting individual geometries,
        preparing APSIM files etc.
      </div>

      {initialData && (
        <div className="alert alert-info" style={{ marginTop: "1rem" }}>
          This form was pre-filled with your previously submitted configuration.
        </div>
      )}

      {/* Region Extraction */}
      <Fieldset
        legend="Region Extraction"
        hint="Choose your shapefile source, projection, and optional filter to restrict extracted features."
      >
        <div className="form-group">
        <label className="form-label">Zipped Shapefile (.zip)</label>
        <FileUpload
          id="shapefileFile"
          accept=".zip"
          label="📁 Choose zipped shapefile (.zip)"
          value={shapefileUpload ? [shapefileUpload] : []}
          onChange={(files) => {
            const newFile = files[0] ?? null;
            setShapefileUpload(newFile);

            if (newFile) {
              handleFileUpload(files);
            } else {
              setDetectedColumns([]);
              setAdminNameColumn("");
              setRegionFilterColumn("");
              setRegionAllowedValues([]);
              setApsimColumn("");
              setApsimAllowedValues(["all"]);
              (window as any).__lastShapefileGeoJSON = undefined;
            }
          }}
        />
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
            <select
                className="form-input"
                value={adminNameColumn}
                onChange={(e) => setAdminNameColumn(e.target.value)}
              >
                <option value="">-- Select column --</option>
                {detectedColumns.map((c) => (
                  <option key={c} value={c}>
                    {c}
                  </option>
                ))}
              </select>
            <p className="subtitle" style={{ marginTop: 4 }}>
              Column in the shapefile that contains the administrative name.
            </p>
          </div>

          <div className="form-group">
            <label className="form-label">Target Projection (EPSG or proj string)</label>
            <input
              className="form-input"
              placeholder="e.g EPSG:4326 or +proj=longlat +datum=WGS84 +no_defs"
              value={targetProjection}
              onChange={(e) => setTargetProjection(e.target.value)}
            />
            <p className="subtitle" style={{ marginTop: 4 }}>
              Should be a local equal area projection (e.g Albers)
            </p>
          </div>
        </div>

        {/* <div
          style={{
            display: "grid",
            gap: "1rem",
            gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
            marginTop: "0.5rem",
          }}
        >
          <Fieldset
            legend="Optional Filters"
            hint="Experimental Feature. Currently disabled \n Restrict which features are extracted. Leave blank to include all. Set column to filter on and then define the values that should be accepted for this column. \n ATTENTION: Currently only working for string columns."
          >
            <div className="form-group">
              <label className="form-label">Optional: Filter Column</label>
              <input
                list="detectedColumns"
                className="form-input"
                placeholder="e.g., admin_name or ADM2_EN"
                value={regionFilterColumn}
                onChange={(e) => setRegionFilterColumn(e.target.value)}
                disabled={true}
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
          </Fieldset>
        </div> */}
      </Fieldset>

      {/* Simulation Windows*/}
      <Fieldset
        legend="Simulation Windows (Years & Timepoints)"
        hint="Add years and timepoints and fill in the required dates. These will be injected into APSIM and data fetchers."
      >
        <MultiValueInput
            label="Years (must be numerical e.g 2019)"
            placeholder="e.g 2019, 2020"
            values={years}
            setValues={(next) => setYears(next)}
            onValueEdit={(oldVal, newVal) => migrateWindows(oldVal, newVal)}
        />
        <MultiValueInput
            label="Timepoints (e.g T-0)."
            placeholder="e.g T-0, T-30"
            values={timepoints}
            setValues={(next) => setTimepoints(next)}
        />

      <div className="form-group" style={{ marginBottom: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
          <input
            className="form-input"
            type="checkbox"
            checked={usePattern}
            onChange={(e) => setUsePattern(e.target.checked)}
            style={{ width: "auto", margin: 0 }}
          />
          <label className="form-label">Use Pattern for Timepoints</label>
          </div>
          <p className="subtitle" style={{ marginTop: "0.25rem", marginLeft: "1.5rem" }}>
            Define a pattern for each timepoint that will be applied to all windows with that timepoint.
          </p>
        </div>

       {usePattern && (
          <>
            {timepoints.map(tp => {
              return (
                <section
                  key={tp}
                  style={{
                    border: "1px solid var(--gray-200)",
                    borderRadius: 10,
                    padding: "1rem",
                    marginBottom: "1rem",
                    marginTop: "1rem",
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
                      {tp} Pattern
                    </strong>
                  </div>

                  <p className="subtitle" style={{ marginBottom: "0.75rem" }}>
                    The year offset is applied to each year of the timepoint and could be e.g -1 for the previous year or +1 for the subsequent year or 0 etc.
                  </p>

                  {/* APSIM Simulation Start */}
                  <div
                    style={{
                      display: "grid",
                      gap: "1rem",
                      gridTemplateColumns: "repeat(auto-fit, minmax(100px, 1fr))",
                      marginBottom: "0.75rem"
                    }}
                  >
                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">APSIM Simulation Start Date Day</label>
                      <input
                        className="form-input"
                        type="number"
                        value={timepointPatterns[tp].sim_start_day}
                        onChange={(e) => updateTimepointPattern(tp, 'sim_start_day', Number(e.target.value))}
                      />
                    </div>

                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">APSIM Simulation Start Date Month</label>
                      <input
                        className="form-input"
                        type="number"
                        value={timepointPatterns[tp].sim_start_month}
                        onChange={(e) => updateTimepointPattern(tp, 'sim_start_month', Number(e.target.value))}
                      />
                    </div>

                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">APSIM Simulation Start Date Year Offset</label>
                      <input
                        className="form-input"
                        type="number"
                        value={timepointPatterns[tp].sim_start_year_offset}
                        onChange={(e) => updateTimepointPattern(tp, 'sim_start_year_offset', Number(e.target.value))}
                      />
                    </div>
                  </div>

                  {/* APSIM Simulation End */}
                  <div
                    style={{
                      display: "grid",
                      gap: "1rem",
                      gridTemplateColumns: "repeat(auto-fit, minmax(100px, 1fr))",
                      marginBottom: "0.75rem"
                    }}
                  >
                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">APSIM Simulation End Date Day</label>
                      <input
                        className="form-input"
                        type="number"
                        value={timepointPatterns[tp].sim_end_day}
                        onChange={(e) => updateTimepointPattern(tp, 'sim_end_day', Number(e.target.value))}
                      />
                    </div>

                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">APSIM Simulation End Date Month</label>
                      <input
                        className="form-input"
                        type="number"
                        value={timepointPatterns[tp].sim_end_month}
                        onChange={(e) => updateTimepointPattern(tp, 'sim_end_month', Number(e.target.value))}
                      />
                    </div>

                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">APSIM Simulation End Date Year Offset</label>
                      <input
                        className="form-input"
                        type="number"
                        value={timepointPatterns[tp].sim_end_year_offset}
                        onChange={(e) => updateTimepointPattern(tp, 'sim_end_year_offset', Number(e.target.value))}
                      />
                    </div>
                  </div>

                  {/* Meteorological Data Start */}
                  <div
                    style={{
                      display: "grid",
                      gap: "1rem",
                      gridTemplateColumns: "repeat(auto-fit, minmax(100px, 1fr))",
                      marginBottom: "0.75rem"
                    }}
                  >
                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">Meteorological Data Start Date Day</label>
                      <input
                        className="form-input"
                        type="number"
                        value={timepointPatterns[tp].met_start_day}
                        onChange={(e) => updateTimepointPattern(tp, 'met_start_day', Number(e.target.value))}
                      />
                    </div>

                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">Meteorological Data Start Date Month</label>
                      <input
                        className="form-input"
                        type="number"
                        value={timepointPatterns[tp].met_start_month}
                        onChange={(e) => updateTimepointPattern(tp, 'met_start_month', Number(e.target.value))}
                      />
                    </div>

                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">Meteorological Data Start Date Year Offset</label>
                      <input
                        className="form-input"
                        type="number"
                        value={timepointPatterns[tp].met_start_year_offset}
                        onChange={(e) => updateTimepointPattern(tp, 'met_start_year_offset', Number(e.target.value))}
                      />
                    </div>
                  </div>

                  {/* Meteorological Data End */}
                  <div
                    style={{
                      display: "grid",
                      gap: "1rem",
                      gridTemplateColumns: "repeat(auto-fit, minmax(100px, 1fr))",
                      marginBottom: "0.75rem"
                    }}
                  >
                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">Meteorological Data End Date Day</label>
                      <input
                        className="form-input"
                        type="number"
                        value={timepointPatterns[tp].met_end_day}
                        onChange={(e) => updateTimepointPattern(tp, 'met_end_day', Number(e.target.value))}
                      />
                    </div>

                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">Meteorological Data End Date Month</label>
                      <input
                        className="form-input"
                        type="number"
                        value={timepointPatterns[tp].met_end_month}
                        onChange={(e) => updateTimepointPattern(tp, 'met_end_month', Number(e.target.value))}
                      />
                    </div>

                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">Meteorological Data End Date Year Offset</label>
                      <input
                        className="form-input"
                        type="number"
                        value={timepointPatterns[tp].met_end_year_offset}
                        onChange={(e) => updateTimepointPattern(tp, 'met_end_year_offset', Number(e.target.value))}
                      />
                    </div>
                  </div>

                  {/* LAI Data Start */}
                  <div
                    style={{
                      display: "grid",
                      gap: "1rem",
                      gridTemplateColumns: "repeat(auto-fit, minmax(100px, 1fr))",
                      marginBottom: "0.75rem"
                    }}
                  >
                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">LAI Data Start Date Day</label>
                      <input
                        className="form-input"
                        type="number"
                        value={timepointPatterns[tp].lai_start_day}
                        onChange={(e) => updateTimepointPattern(tp, 'lai_start_day', Number(e.target.value))}
                      />
                    </div>

                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">LAI Data Start Date Month</label>
                      <input
                        className="form-input"
                        type="number"
                        value={timepointPatterns[tp].lai_start_month}
                        onChange={(e) => updateTimepointPattern(tp, 'lai_start_month', Number(e.target.value))}
                      />
                    </div>

                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">LAI Data Start Date Year Offset</label>
                      <input
                        className="form-input"
                        type="number"
                        value={timepointPatterns[tp].lai_start_year_offset}
                        onChange={(e) => updateTimepointPattern(tp, 'lai_start_year_offset', Number(e.target.value))}
                      />
                    </div>
                  </div>

                  {/* LAI Data End */}
                  <div
                    style={{
                      display: "grid",
                      gap: "1rem",
                      gridTemplateColumns: "repeat(auto-fit, minmax(100px, 1fr))"
                    }}
                  >
                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">LAI Data End Date Day</label>
                      <input
                        className="form-input"
                        type="number"
                        value={timepointPatterns[tp].lai_end_day}
                        onChange={(e) => updateTimepointPattern(tp, 'lai_end_day', Number(e.target.value))}
                      />
                    </div>

                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">LAI Data End Date Month</label>
                      <input
                        className="form-input"
                        type="number"
                        value={timepointPatterns[tp].lai_end_month}
                        onChange={(e) => updateTimepointPattern(tp, 'lai_end_month', Number(e.target.value))}
                      />
                    </div>

                    <div className="form-group" style={{ marginBottom: 0 }}>
                      <label className="form-label">LAI Data End Date Year Offset</label>
                      <input
                        className="form-input"
                        type="number"
                        value={timepointPatterns[tp].lai_end_year_offset}
                        onChange={(e) => updateTimepointPattern(tp, 'lai_end_year_offset', Number(e.target.value))}
                      />
                    </div>
                  </div>
                </section>
              );
            })}
          </>
        )}

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
                <label className="form-label">APSIM Simulation Start Date</label>
                <input
                  className="form-input"
                  type="date"
                  value={w.sim_start_date}
                  onChange={(e) => updateWindow(w.id, { sim_start_date: e.target.value })}
                />
              </div>

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">APSIM Simulation End Date</label>
                <input
                  className="form-input"
                  type="date"
                  value={w.sim_end_date}
                  onChange={(e) => updateWindow(w.id, { sim_end_date: e.target.value })}
                />
              </div>

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">Meteorological Data Start Date</label>
                <input
                  className="form-input"
                  type="date"
                  value={w.met_start_date}
                  onChange={(e) => updateWindow(w.id, { met_start_date: e.target.value })}
                />
              </div>

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">Meteorological Data End Date</label>
                <input
                  className="form-input"
                  type="date"
                  value={w.met_end_date}
                  onChange={(e) => updateWindow(w.id, { met_end_date: e.target.value })}
                />
              </div>

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">LAI Data Start Date</label>
                <input
                  className="form-input"
                  type="date"
                  value={w.lai_start_date}
                  onChange={(e) => updateWindow(w.id, { lai_start_date: e.target.value })}
                />
              </div>

              <div className="form-group" style={{ marginBottom: 0 }}>
                <label className="form-label">LAI Data End Date</label>
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
              value={apsimFiles}
              onChange={setApsimFiles}
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


        {apsimColumn ? <SourceTargetMapper
          title="Map APSIM to Region Value"
          hint="For each APSIM file, choose a target value from the the specified column."
          sources={apsimSources}
          targets={apsimAllowedValues}
          value={apsimMapping}
          onChange={setApsimMapping}
          allowDuplicateTargets={false}
          multi={true}
        /> : <b>Select a column in your APSIM file first!</b>
        }
        </>) : ''}
      </Fieldset>

      {/* ========================= Reference Data ========================= */}
      <Fieldset
        legend="Optional: Reference Data"
        hint="Upload CSVs and map each file to an aggregation level. Must contain columns: 'region' containing the admin names as specified above and a column 'reported_mean_yield_kg_ha'"
      >
        <div className="form-group">
          <label className="form-label">Reference Data (.csv)</label>
          <FileUpload
            id="referenceFile"
            accept=".csv"
            multiple
            label="📁 Choose one or more Referencedata files (.csv)"
            value={referenceFiles}
            onChange={setReferenceFiles}
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
                show((err as Error).message, 'error');
              }
            }}>
            Upload Setup Config
          </button>
        </div>
      </div>
      <Toast />
    </div>
  );
};

export default SetupStudyForm;
