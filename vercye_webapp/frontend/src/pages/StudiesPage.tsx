import { useEffect, useMemo, useState } from 'react';
import Header from '@/components/Header';
import Modal from '@/components/Modal';
import StatusBadge from '@/components/StatusBadge';
import Stepper from '@/components/Stepper';
import useToast from '@/components/Toast';
import { StudiesAPI } from '@/api/studies';
import type { SetupConfigTemplate, StudyId, StudyStatus, RunConfigFormParams } from '@/types';
import SetupStudyForm, { SetupSubmissionsPayload } from '@/components/Forms/SetupStudyForm';
import RunParamsForm, { RunParamsSubmissionsPayload } from '@/components/Forms/RunParamsForm';
import { ApiError } from '@/api/client';
import { ansiToHtml } from '@/utils/utils';

function toArray(input: unknown): string[] {
  if (Array.isArray(input)) return input as string[];
  if (input && typeof input === 'object') {
    const anyInput = input as any;
    if (Array.isArray(anyInput.studies)) return anyInput.studies as string[];
    if (Array.isArray(anyInput.study_ids)) return anyInput.study_ids as string[];
    const keys = Object.keys(anyInput);
    if (keys.length && keys.every(k => typeof k === 'string')) return keys;
  }
  return [];
}

enum Result {
  Map = "map",
  Report = "Report"
}

const StudiesPage = () => {
  const [studies, setStudies] = useState<StudyId[] | null>(null);
  const [createOpen, setCreateOpen] = useState(false);
  const [createName, setCreateName] = useState('');
  const [detailOpen, setDetailOpen] = useState(false);
  const [detailStudy, setDetailStudy] = useState<StudyId | null>(null);
  const [step, setStep] = useState<1|2|3|4>(1);
  const [runConfigMessage, setRunConfigMessage] = useState('');
  const [logsOpen, setLogsOpen] = useState(false);
  const [logs, setLogs] = useState<string>('Loading logs...');
  const [statuses, setStatuses] = useState<Record<string, StudyStatus>>({});
  const [duplicateOpen, setDuplicateOpen] = useState(false);
  const [initialSetupData, setInitialSetupData] = useState<SetupConfigTemplate | null>(null)
  const [initialRunConfigData, setInitialRunConfigData] = useState<RunConfigFormParams | null>(null)

  // Results selector modal state
  const [resultYears, setResultYears] = useState<Record<string, string[]>>({});
  const [yearOptions, setYearOptions] = useState<string[]>([]);
  const [timepointOptions, setTimepointOptions] = useState<string[]>([]);
  const [resultsSelectorOpen, setResultsSelectorOpen] = useState(false);
  const [selectedYear, setSelectedYear] = useState('');
  const [selectedTimepoint, setSelectedTimepoint] = useState('');
  const [selectedResultType, setSelectedResultType] = useState<Result | null>(null)

  const { show, Toast } = useToast();

  const [loadingCount, setLoadingCount] = useState(0);
  const [loadingText, setLoadingText] = useState<string | null>(null);
  const isLoading = loadingCount > 0;

  const withLoading = async <T,>(task: () => Promise<T>, message?: string): Promise<T> => {
    if (message) setLoadingText(message);
    setLoadingCount(c => c + 1);
    try {
      return await task();
    } finally {
      setLoadingCount(c => Math.max(0, c - 1));
      setLoadingText(null);
    }
  };

  useEffect(() => {
    load();
    const iv = setInterval(async () => {
      if (!studies) return;
      for (const id of studies) {
        try {
          const s = await StudiesAPI.status(id);
          setStatuses((prev) => ({ ...prev, [id]: s.status }));
        } catch {}
      }
    }, 10000);
    return () => clearInterval(iv);
  }, [studies?.length]);

  const load = async () => {
    try {
      const data = await StudiesAPI.list() as unknown;
      const studies = toArray(data);
      setStudies(studies);
      for (const id of studies) {
        try {
          const s = await StudiesAPI.status(id);
          setStatuses((prev) => ({ ...prev, [id]: s.status }));
        } catch {}
      }
    } catch {
      setStudies([]);
      show('Failed to load studies', 'error');
    }
  }

  const fetchSetInitialSetupData = async (studyID: StudyId) => {
    try {
      const setupConfigData = await StudiesAPI.getSetupConfig(studyID)
      setInitialSetupData(setupConfigData)
    } catch (err) {
      if (err instanceof ApiError) {
        show(err.message, 'error');
      } else {
        show('Failed to load setup config template!', 'error');
      }
    }
  }

  const fetchSetInitialRunConfigData = async (studyID: StudyId) => {
    try {
      const runConfigData = await StudiesAPI.runConfigFormParams(studyID)
      setInitialRunConfigData(runConfigData)
    } catch (err) {
      if (err instanceof ApiError) {
        show(err.message, 'error');

      } else {
        show('Failed to load initial runconfig data!', 'error');
      }
    }
  }

  const openDetail = (id: StudyId) => {
    setDetailStudy(id);
    withLoading((async () => {
      try {
        await fetchSetInitialSetupData(id);
        await fetchSetInitialRunConfigData(id);
        let currentStep: 1|2|3|4 = 2;
        try {
          await StudiesAPI.runConfig(id);
          const st = await StudiesAPI.runConfigStatus(id);
          currentStep = st.status === 'valid' ? 4 : 3;
          setRunConfigMessage(st.details || '');
        } catch {
          currentStep = 2;
        }
        setStep(currentStep);
        setDetailOpen(true);
      } catch {
        setStep(2);
        setDetailOpen(true);
      }
    }), 'Loading configuration…');
  }

  const createStudy = async () => {
    if (!createName.trim()) return;
    await withLoading(async () => {
      try {
        await StudiesAPI.create(createName.trim());
        show(`Study "${createName}" created successfully!`, 'success');
        setCreateOpen(false);
        setCreateName('');
        await load();
        setTimeout(() => openDetail(createName.trim()), 400);
      } catch (err) {
        if (err instanceof ApiError) {
          show(err.message, 'error');
        } else {
          show('Failed to create study', 'error');
        }
      }
    }, 'Creating study…');
  }

  const runStudy = async (id: StudyId, forceRerun: boolean) => {
    await withLoading(async () => {
      try {
        await StudiesAPI.run(id, forceRerun);
        show('Run started', 'success');
        setStatuses((s) => ({ ...s, [id]: 'running' } as any));
        setDetailOpen(false);
      } catch (err) {
        if (err instanceof ApiError) {
          show(err.message, 'error');
        } else {
          show('Failed to start run', 'error');
        }
      }
    }, 'Starting run…');
  }

  const cancelRun = async (id: StudyId) => {
    await withLoading(async () => {
      try {
        await StudiesAPI.cancel(id);
        show('Run cancelled', 'success');
        setStatuses((s) => ({ ...s, [id]: 'cancelled' } as any));
      } catch (err) {
        if (err instanceof ApiError) {
          show(err.message, 'error');
        } else {
          show('Failed to cancel run', 'error');
        }
      }
    }, 'Cancelling…');
  }

  const openLogs = async (id: StudyId) => {
    setLogs('Loading logs...');
    setLogsOpen(true);
    try {
      const l = await StudiesAPI.logs(id);
      setLogs(l);
    } catch {
      setLogs('Failed to load logs');
    }
  }

  const openMapResultsSelector = async (id: StudyId) => {
    setSelectedResultType(Result.Map)
    await openResultsSelector(id)
  }

  const openReportResultsSelector =  async (id: StudyId) => {
    setSelectedResultType(Result.Report)
    await openResultsSelector(id)
  }

  const openResultsSelector = async (id: StudyId) => {
    setDetailStudy(id);
    try {
      const data = await StudiesAPI.resultTimepoints(id); // expects { timepoints: { [year]: string[] } }
      const yearsObj: Record<string, string[]> = data?.timepoints || {};
      const years = Object.keys(yearsObj).sort();
      setResultYears(yearsObj);
      setYearOptions(years);
      if (years.length > 0) {
        const y = years[0];
        const tps = yearsObj[y] || [];
        setSelectedYear(y);
        setTimepointOptions(tps);
        setSelectedTimepoint(tps[0] || '');
      } else {
        setSelectedYear('');
        setTimepointOptions([]);
        setSelectedTimepoint('');
      }
      setResultsSelectorOpen(true);
    } catch {
      show('Failed to load result timepoints', 'error');
    }
  }

  const onYearChange = (nextYear: string) => {
    setSelectedYear(nextYear);
    const tps = resultYears[nextYear] || [];
    setTimepointOptions(tps);
    setSelectedTimepoint(tps[0] || '');
  }

  const openSelectedResult = async () => {
    if (!detailStudy || !selectedYear || !selectedTimepoint) return;
    if(selectedResultType == Result.Map) {
      const url = StudiesAPI.mapResultUrl(detailStudy, selectedYear, selectedTimepoint)
      window.open(url, '_blank', 'noopener,noreferrer');
    } else if(selectedResultType == Result.Report) {
      try {
          const blob = await StudiesAPI.downloadReport(detailStudy, selectedYear, selectedTimepoint);
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url; a.download = `final_report_${detailStudy}_${selectedYear}_${selectedTimepoint}.pdf`; a.click(); URL.revokeObjectURL(url);
      } catch { show('Download failed', 'error'); }
    }
    setResultsSelectorOpen(false);
  }


  const showMultiyearResultsReport = (id: StudyId) => {
    const url = StudiesAPI.multiyearSummaryReportUrl(id)
    window.open(url, '_blank', 'noopener,noreferrer');
  }

  const handleCopyLogs = async () => {
    try {
      await navigator.clipboard.writeText(logs || '');
      show('Logs copied to clipboard', 'success');
    } catch {
      show('Failed to copy logs', 'error');
    }
  };

  const duplicateStudy = async () => {
    if (!detailStudy || !createName) return;
    await withLoading(async () => {
      try {
        await StudiesAPI.duplicate(detailStudy, createName);
        show('Study duplicated successfully', 'success');
        setCreateName('');
      } catch (err) {
        if (err instanceof ApiError) {
          show(err.message, 'error');
        } else {
          show('Failed to duplicate study!', 'error');
        }
      }
      await load()
    }, 'Duplicating study…');
  };

  const table = useMemo(() => {
    if (!studies) return (
      <div className="empty-state">
        <div className="loading"></div>
        <p style={{marginTop: '1rem'}}>Loading studies...</p>
      </div>
    );
    if (studies.length === 0) return (
      <div className="empty-state">
        <h3>No Studies Found</h3>
        <p>Create your first study to get started</p>
        <button className="btn btn-primary" style={{marginTop: 15}} onClick={() => setCreateOpen(true)}>✚ Create Your First Study</button>
      </div>
    );
    return (
      <div className="table-container">
        <table className="table">
          <thead>
            <tr><th>Study Name</th><th>Status</th><th>Actions</th></tr>
          </thead>
          <tbody>
            {studies.map((id) => {
              const s = statuses[id] ?? 'pending';
              const canResults = s === 'completed';
              const canCancel = s === 'running' || s === 'validating';
              const canRun = s === 'ready' ||  s === 'failed' || s === 'completed';
              const canConfigure = s !== 'running'
              return (
                <tr key={id}>
                  <td><div style={{fontWeight: 500}}>{id}</div></td>
                  <td><StatusBadge status={s as any} /></td>
                  <td>
                    <div className="actions-cell">
                      <button className="btn btn-sm btn-primary" onClick={() => openDetail(id)} disabled={!canConfigure}>Configure</button>
                        <button
                          className="btn btn-sm btn-success"
                          onClick={() => runStudy(id, false)}
                          disabled={!canRun}
                        >
                          Run
                        </button>
                      <button
                        className="btn btn-sm btn-success"
                        onClick={() => runStudy(id, true)}
                         disabled={!canRun}
                      >
                        Force Rerun
                      </button>
                      {canCancel && <button className="btn btn-sm btn-danger" onClick={() => cancelRun(id)}>Cancel</button>}
                      <button className="btn btn-sm btn-secondary" onClick={() => openMapResultsSelector(id)} style={{display: canResults ? 'inline-flex' : 'none'}}>Results map</button>
                      <button className="btn btn-sm btn-secondary" onClick={() => showMultiyearResultsReport(id)} style={{display: canResults ? 'inline-flex' : 'none'}}>Multiyer report</button>
                      <button className="btn btn-sm btn-secondary" onClick={() => openReportResultsSelector(id)} style={{display: canResults ? 'inline-flex' : 'none'}}>Yearly reports</button>
                      <button className="btn btn-sm btn-secondary" onClick={() => openLogs(id)}>Logs</button>
                      <button className="btn btn-sm btn-secondary" onClick={() => {setDetailStudy(id); setDuplicateOpen(true)}}>Use as template</button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    );
  }, [studies, statuses]);

  const handleSetupSubmission = async (payload: SetupSubmissionsPayload) => {
    if (!detailStudy) return;
    await withLoading(async () => {
      try {
        await StudiesAPI.uploadSetup(detailStudy, payload);
        show('Setup uploaded', 'success');
        await fetchSetInitialSetupData(detailStudy);
        setStep(3);
      } catch (err) {
        if (err instanceof ApiError) {
          show(err.message, 'error');
        } else {
          show('Failed to upload setup', 'error');
        }
      }
    }, 'Uploading setup…');
  };

  const handleRunParamsSubmission = async (payload: RunParamsSubmissionsPayload) => {
    if (!detailStudy) return;
    await withLoading(async () => {
      try {
        await StudiesAPI.uploadRunConfig(detailStudy, payload);
        show('Run config uploaded', 'success');
        await fetchSetInitialRunConfigData(detailStudy);
        setStep(4);
      } catch (err) {
        if (err instanceof ApiError) {
          show(err.message, 'error');
        } else {
          show('Failed to upload run config', 'error');
        }
        throw new Error('failed to upload run config');
      }
    }, 'Uploading run config…');
  };

  const handleDownloadTemplate = async () => {
    if (!detailStudy) return;
    try {
      const blob = await StudiesAPI.downloadRunConfigTemplate(detailStudy);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `run_config_${detailStudy}.yaml`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      if (err instanceof ApiError) {
        show(err.message, 'error');
      } else {
        show('Download failed', 'error');
      }
    }
  };

  return (
    <div className="container">
      <Header />
      <div className="actions">
        <button className="btn btn-primary" onClick={() => setCreateOpen(true)}>✚ Create New Study</button>
        <button className="btn btn-secondary" onClick={() => load()}>↻ Refresh</button>
      </div>
      <div id="studiesContainer">{table}</div>

      {/* Create Study */}
      <Modal open={createOpen} onClose={() => setCreateOpen(false)} title="Create New Study">
        <div className="form-group">
          <label htmlFor="studyName" className="form-label">Study Name</label>
          <input id="studyName" className="form-input" placeholder="Enter study name" value={createName} onChange={e=>setCreateName(e.target.value)}/>
        </div>
        <div style={{ display:'flex', gap: '0.75rem', justifyContent:'flex-end' }}>
          <button className="btn btn-secondary" onClick={()=>setCreateOpen(false)}>Cancel</button>
          <button className="btn btn-primary" onClick={createStudy}>Create Study</button>
        </div>
      </Modal>

      {/* Detail / Setup */}
      <Modal open={detailOpen} onClose={()=>setDetailOpen(false)} title={`${detailStudy ?? ''} - Configuration`} width={800}>
        <Stepper step={step} onStepChange={setStep}/>
          {step === 2 &&  <SetupStudyForm onSubmit={handleSetupSubmission} initialData={initialSetupData}/>}

          {step === 3 && <RunParamsForm
              onSubmit={handleRunParamsSubmission}
              onDownloadTemplate={handleDownloadTemplate}
              runConfigMessage={runConfigMessage}
              currentStudyId={detailStudy}
              initialData={initialRunConfigData}
            />}

          {step === 4 && (
            <div>
              <div className="alert alert-success">
                <strong>Setup Complete!</strong> Your study is ready to run.
              </div>
              <div style={{display:'flex', gap:'0.75rem', justifyContent:'space-between'}}>
                <button className="btn btn-secondary" onClick={async () => {
                  if (!detailStudy) return;
                  try {
                    const blob = await StudiesAPI.downloadRunConfigTemplate(detailStudy);
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url; a.download = 'config.yaml'; a.click(); URL.revokeObjectURL(url);
                  } catch { show('Download failed', 'error'); }
                }}>⬇️ Download Config</button>
                <div style={{display:'flex', gap:'0.75rem'}}>
                  <button className="btn btn-secondary" onClick={() => setStep(3)}>📝 Update Config</button>
                  <button className="btn btn-success" onClick={() => detailStudy && runStudy(detailStudy, false)}>▶️ Run Study</button>
                </div>
              </div>
            </div>
          )}
      </Modal>

      {/* Results year/timepoint selector */}
      <Modal open={resultsSelectorOpen} onClose={() => setResultsSelectorOpen(false)} title="Select Result Timepoint">
        <div className="form-group">
          <label className="form-label">Year</label>
          <select value={selectedYear} className="form-input" onChange={(e) => onYearChange(e.target.value)}>
            {yearOptions.map(y => <option key={y} value={y}>{y}</option>)}
          </select>
        </div>
        <div className="form-group">
          <label className="form-label">Timepoint</label>
          <select value={selectedTimepoint} className="form-input" onChange={(e) => setSelectedTimepoint(e.target.value)}>
            {timepointOptions.map(tp => <option key={tp} value={tp}>{tp}</option>)}
          </select>
        </div>
        <div style={{display:'flex', gap:'0.75rem', justifyContent:'flex-end'}}>
          <button className="btn btn-secondary" onClick={() => setResultsSelectorOpen(false)}>Cancel</button>
          <button className="btn btn-primary" onClick={openSelectedResult} disabled={!selectedYear || !selectedTimepoint}>Open</button>
        </div>
      </Modal>

      {/* Logs */}
      <Modal open={logsOpen} onClose={() => setLogsOpen(false)} title="Study Logs" width={800}>
        <button
          onClick={handleCopyLogs}
          className="btn btn-sm btn-secondary"
        >
          📋 Copy logs
        </button>
        <div className="logs-container" style={{ marginTop: '2rem' }}>
          <pre dangerouslySetInnerHTML={{ __html: ansiToHtml(logs || 'No logs available') }} />
        </div>
      </Modal>

      {/* Duplicate Study */}
      <Modal open={duplicateOpen} onClose={() => setDuplicateOpen(false)} title="Create New Study from Existing">
        <p className="subtitle" style={{ marginTop: 4 }}>
          Setup a new study based on the configuration of an existing one. Will use the same regions, APSIM files and reference data.
        </p>
        <div className="form-group">
          <label htmlFor="studyName" className="form-label">Study Name</label>
          <input id="studyName" className="form-input" placeholder="Enter study name" value={createName} onChange={e=>setCreateName(e.target.value)}/>
        </div>
        <div style={{ display:'flex', gap: '0.75rem', justifyContent:'flex-end' }}>
          <button className="btn btn-secondary" onClick={()=>setDuplicateOpen(false)}>Cancel</button>
          <button className="btn btn-primary" onClick={() => {duplicateStudy(); setDuplicateOpen(false)}}>Create Study</button>
        </div>
      </Modal>

      <Toast />
      {isLoading && (
        <div
          role="alert"
          aria-live="polite"
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.35)',
            zIndex: 9999,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            padding: 16
          }}
        >
          <div
            style={{
              background: '#fff',
              borderRadius: 16,
              padding: '18px 22px',
              boxShadow: '0 10px 30px rgba(0,0,0,0.2)',
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              minWidth: 220,
              justifyContent: 'center'
            }}
          >
            <div
              aria-hidden
              style={{
                width: 28,
                height: 28,
                borderRadius: '50%',
                border: '3px solid #E5E7EB',
                borderTopColor: '#3B82F6',
                animation: 'spin 1s linear infinite'
              }}
            />
            <span style={{ fontSize: 14, color: '#374151' }}>{loadingText || 'Working…'}</span>
          </div>
        </div>
      )}
      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

export default StudiesPage
