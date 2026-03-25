import { useEffect, useState } from "react";
import FileUpload from "../FileUpload";
import { LAIAPI } from "@/api/lai";
import { LAIEntry, RunConfigFormParams } from "@/types";
import SourceTargetMapper, { MapperSource, MappingState } from "./SourceTargetMapper";
import { CropmasksAPI } from "@/api/cropmasks";
import useToast from '@/components/Toast';
import { StudiesAPI } from "@/api/studies";

interface RunParamsFormProps {
    runConfigMessage: string;
    onSubmit: (payload: RunParamsSubmissionsPayload) => Promise<void>;
    onDownloadTemplate: () => void,
    currentStudyId: string | null,
    initialData: RunConfigFormParams | null
}

export interface RunParamsSubmissionsPayload {
    configFile: File,
    laiSourceResolution?: number
    laiSourceId?: string,
    cropmaskMapping: MappingState
}

interface MinimalLAIEntry {
    id: string;
    resolution: number
}

const RunParamsForm: React.FC<RunParamsFormProps> = ({ runConfigMessage, onSubmit, onDownloadTemplate, currentStudyId, initialData }) => {
    const [laiSource, setLaiSource] = useState<MinimalLAIEntry | null>(null);
    const [laiSources, setLaiSources] = useState<MinimalLAIEntry[]>([]);
    const [laiLoading, setLaiLoading] = useState(true);
    const [submitting, setSubmitting] = useState(false);
    const [cropmasks, setCropmasks] = useState<string[]>([]);
    const [requiredCropmaskYears, setRequiredCropmaskYears] = useState<MapperSource[]>([]);
    const [cropmaskMapping, setCropmaskMapping] = useState<MappingState>({})
    const [configFiles, setConfigFiles] = useState<File[]>([]);

    const { show, Toast } = useToast();

    useEffect(() => {
        if (!initialData || laiLoading || laiSources.length === 0 || cropmasks.length === 0) return;

        const matchingLAI = laiSources.find(
            (lai) => lai.id === initialData.laiId && lai.resolution === initialData.laiResolution
        );
        if (matchingLAI) setLaiSource(matchingLAI);

        if (initialData.cropmasks) setCropmaskMapping(initialData.cropmasks);
    }, [initialData, laiSources, cropmasks, laiLoading]);

    useEffect(() => {
        (async () => {
            setLaiLoading(true);
            try {
                const res = await LAIAPI.list();
                setLaiSources(res);
            } catch {
                setLaiSources([]);
            } finally {
                setLaiLoading(false);
            }
        })();
    }, []);

    useEffect(() => {
        (async () => {
            try {
                const res = await CropmasksAPI.list();
                setCropmasks(res.map((el) => el.id));
            } catch {
                show('Failed to load cropmasks', 'error')
                setCropmasks([]);
            }
        })()
    }, [])

    useEffect(() => {
        if(!currentStudyId) return
        (async () => {
            try {
                const res = await StudiesAPI.studyYears(currentStudyId);
                setRequiredCropmaskYears(res.map((el) => {return {id: String(el), label: String(el)}}));
            } catch {
                show('Failed to load required cropmask years', 'error')
                setRequiredCropmaskYears([]);
            }
        })()
    }, [currentStudyId])

    const handleLAISourceChanged = (value: string) => {
        const [id, resolutionWithM] = value.split(" ");
        const resolution = Number(resolutionWithM?.slice(0, -1));
        const newLAISource = laiSources.find((laiSrc) => laiSrc.id == id && laiSrc.resolution == resolution) || null;
        setLaiSource(newLAISource);
    };

    const buildPayload = (): RunParamsSubmissionsPayload => {
          if (configFiles.length === 0) throw new Error("Set config file first!");
          if (!laiSource) throw new Error("Please select an LAI source!");
        const payload: RunParamsSubmissionsPayload = {configFile: configFiles[0], cropmaskMapping };
        if (laiSource?.id && laiSource?.resolution) {
            payload.laiSourceId = laiSource.id;
            payload.laiSourceResolution = laiSource.resolution;
        }
        return payload;
    };

    const handleSubmit = async () => {
        try {
            setSubmitting(true);
            const payload = buildPayload();
            await onSubmit(payload);
        } catch (err: any) {
            setConfigFiles([]);
            show((err as Error).message, 'error');
        } finally {
            setSubmitting(false);
        }
    };

    return (
        <div>
            <div className="alert alert-info">
                <strong>Step 3:</strong> Set your run parameters.<br /><br />
                Upload your configured config.yaml file and specify the LAI source to use.
            </div>

            {/**TODO refactor this weird runmessage handling currently**/}
            {runConfigMessage && (runConfigMessage !== 'Template not yet filled in' && runConfigMessage !== 'OK!') && (
                <div className="alert alert-error">
                    <strong>Current config not valid</strong>: {runConfigMessage}
                </div>
            )}

            <div className="form-group">
                <label className="form-label">LAI Source</label>
                {laiLoading ? (
                    <p>⏳ Loading available LAI sources...</p>
                ) : laiSources.length === 0 ? (
                    <p style={{ color: "gray" }}>No LAI sources available</p>
                ) : (
                    <>
                        <select
                            className="form-input"
                            value={laiSource ? `${laiSource.id} ${laiSource.resolution}m` : ""}
                            onChange={(e) => handleLAISourceChanged(e.target.value)}
                            disabled={laiLoading || submitting}
                        >
                            <option value="">-- Select entry --</option>
                            {laiSources.map((c) => (
                                <option key={`${c.id}_${c.resolution}`} value={`${c.id} ${c.resolution}m`}>{`${c.id} ${c.resolution}m`}</option>
                            ))}
                        </select>
                        <p className="subtitle" style={{ marginTop: 4 }}>
                            If you haven't create the LAI data yet, visit the LAI Tab in the top navigation.
                        </p>
                    </>
                )}
            </div>

            <SourceTargetMapper
                title="Cropmasks"
                hint="For each year choose an already uploaded cropmask. You can upload cropmasks in the 'Cropmasks' tab in the top navigation."
                sources={requiredCropmaskYears}
                targets={cropmasks}
                value={cropmaskMapping}
                onChange={setCropmaskMapping}
                allowDuplicateTargets={true}
            />

           <div className="form-group" style={{ marginTop: "10px" }}>
                <label className="form-label">Run Configuration File</label>
                <p className="subtitle" style={{ marginTop: 4 }}>
                    Download the config and then edit the parameters in your file editor.
                    Finally, reupload the filled-in file.
                </p>

                <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
                    <button
                        className="btn btn-secondary"
                        onClick={() => onDownloadTemplate()}
                        disabled={submitting}
                    >
                        ⬇️ Download Config
                    </button>
                    <FileUpload
                        id="runFile"
                        accept=".yaml,.yml"
                        label="📁 Upload completed config.yaml"
                        value={configFiles}
                        onChange={setConfigFiles}
                        />
                </div>
            </div>

            <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
                <div></div>
                <button
                    className="btn btn-primary"
                    onClick={handleSubmit}
                    disabled={submitting}
                >
                    {submitting ? "Uploading..." : "Upload Run Config"}
                </button>
            </div>
            <Toast />
        </div>
    );
};

export default RunParamsForm;
