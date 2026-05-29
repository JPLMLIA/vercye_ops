import { Fragment, useEffect, useMemo, useState } from 'react';
import Header from '@/components/Header';
import Modal from '@/components/Modal';
import StatusBadge from '@/components/StatusBadge';
import Stepper from '@/components/Stepper';
import useToast from '@/components/Toast';
import { StudiesAPI } from '@/api/studies';
import type { SetupConfigTemplate, StudyId, StudyStatus, RunConfigFormParams, StudyRun, RunID } from '@/types';
import SetupStudyForm, { SetupSubmissionsPayload } from '@/components/Forms/SetupStudyForm';
import RunParamsForm, { RunParamsSubmissionsPayload } from '@/components/Forms/RunParamsForm';
import YamlTree from '@/components/YamlTree';
import { ApiError } from '@/api/client';
import { ansiToHtml } from '@/utils/utils';

enum Result {
  Map = 'map',
  Report = 'Report',
}

const SplitButton: React.FC<{
  label: string;
  onPrimary: () => void;
  disabled?: boolean;
  variant?: 'primary' | 'success' | 'secondary' | 'danger';
  size?: 'sm' | 'xs';
  menu: { label: string; onClick: () => void; tooltip?: string }[];
}> = ({ label, onPrimary, disabled, variant = 'primary', size = 'sm', menu }) => {
  const [open, setOpen] = useState(false);
  const sizeClass = `btn-${size}`;
  return (
    <div className="btn-split" onMouseLeave={() => setOpen(false)}>
      <button className={`btn ${sizeClass} btn-${variant} btn-part1 ${open ? 'open' : ''}`} onClick={onPrimary} disabled={disabled}>{label}</button>
      <button
        className={`btn ${sizeClass} btn-${variant} btn-part2 btn-caret ${open ? 'open' : ''}`}
        aria-haspopup="menu" aria-expanded={open} aria-label="More options"
        onClick={() => setOpen(o => !o)} disabled={disabled}
      > ▾</button>
      {open && (
        <div role="menu" className="dropdown-menu">
          {menu.map((m, i) => (
            <button key={i} className="dropdown-item" onClick={() => { setOpen(false); m.onClick(); }} title={m.tooltip || ''}>
              {m.label}{m.tooltip ? <span aria-hidden style={{ marginLeft: 6, fontWeight: 600 }}> ?</span> : null}
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

const formatDate = (iso: string | null | undefined) => {
  if (!iso) return '—';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return iso;
  return d.toLocaleString();
};

const formatSize = (bytes: number | null | undefined) => {
  if (!bytes || bytes <= 0) return '—';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  let v = bytes;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${units[i]}`;
};

const StudiesPage = () => {
  const [createOpen, setCreateOpen] = useState(false);
  const [createName, setCreateName] = useState('');
  const [detailOpen, setDetailOpen] = useState(false);
  const [detailStudy, setDetailStudy] = useState<StudyId | null>(null);
  const [step, setStep] = useState<1 | 2 | 3 | 4>(1);
  const [runConfigMessage, setRunConfigMessage] = useState('');
  const [logsOpen, setLogsOpen] = useState(false);
  const [logs, setLogs] = useState<string>('Loading logs...');
  const [duplicateOpen, setDuplicateOpen] = useState(false);
  const [initialSetupData, setInitialSetupData] = useState<SetupConfigTemplate | null>(null);
  const [initialRunConfigData, setInitialRunConfigData] = useState<RunConfigFormParams | null>(null);
  const [deleteOpen, setDeleteOpen] = useState(false);

  // Studies / pagination state
  const [studies, setStudies] = useState<StudyId[] | null>(null);
  const [statuses, setStatuses] = useState<Record<string, StudyStatus>>({});
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [total, setTotal] = useState(0);

  // Expansion / per-study runs state
  const [expanded, setExpanded] = useState<Record<StudyId, boolean>>({});
  const [runsByStudy, setRunsByStudy] = useState<Record<StudyId, StudyRun[] | 'loading' | 'error'>>({});

  // Per-run config viewer state (key: `${studyId}|${runId}`)
  const [configOpen, setConfigOpen] = useState<Record<string, boolean>>({});
  const [configByRun, setConfigByRun] = useState<Record<string, string | 'loading' | 'error'>>({});

  type ApsimManifest = {
    manifest: {
      filter_column: string | null;
      region_to_template: Record<string, string>;
      files: { name: string; size: number | null; regions: string[] }[];
    };
    files: { name: string; size: number }[];
  };
  const [apsimByRun, setApsimByRun] = useState<Record<string, ApsimManifest | 'loading' | 'error'>>({});

  // Result selector modal state (year/timepoint picker, scoped to a specific run)
  const [resultsSelectorOpen, setResultsSelectorOpen] = useState(false);
  const [selectorContext, setSelectorContext] = useState<{
    studyId: StudyId;
    runId: RunID;
    timepoints: Record<string, string[]>;
    type: Result;
  } | null>(null);
  const [selectedYear, setSelectedYear] = useState('');
  const [selectedTimepoint, setSelectedTimepoint] = useState('');

  // Delete-run confirmation state
  const [deleteRunCtx, setDeleteRunCtx] = useState<{ studyId: StudyId; runId: RunID } | null>(null);

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

  useEffect(() => { withLoading(load, 'Loading studies…'); }, [page, pageSize]);

  useEffect(() => {
    if (!studies?.length) return;
    const iv = setInterval(async () => {
      try {
        const map = await StudiesAPI.statusMany(studies);
        setStatuses(prev => ({ ...prev, ...map as any }));
      } catch {}
    }, 10000);
    return () => clearInterval(iv);
  }, [studies?.join(',')]);

  // When a study transitions to "completed" while expanded, refresh its runs so
  // the newly archived snapshot appears without requiring a manual reload.
  useEffect(() => {
    Object.entries(statuses).forEach(([id, s]) => {
      if (s === 'completed' && expanded[id]) {
        loadRuns(id);
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [statuses]);

  const load = async () => {
    try {
      const { items, total } = await StudiesAPI.listPaged(page, pageSize);
      setStudies(items);
      setTotal(total);
      if (items.length) {
        try {
          const map = await StudiesAPI.statusMany(items);
          setStatuses(prev => ({ ...prev, ...map as any }));
        } catch {}
      }
    } catch {
      setStudies([]);
      show('Failed to load studies', 'error');
    }
  };

  const loadRuns = async (id: StudyId) => {
    setRunsByStudy(prev => ({ ...prev, [id]: prev[id] && prev[id] !== 'error' ? prev[id] : 'loading' }));
    try {
      const { items } = await StudiesAPI.listRuns(id);
      setRunsByStudy(prev => ({ ...prev, [id]: items }));
    } catch {
      setRunsByStudy(prev => ({ ...prev, [id]: 'error' }));
    }
  };

  const toggleExpand = (id: StudyId) => {
    const willExpand = !expanded[id];
    setExpanded(prev => ({ ...prev, [id]: willExpand }));
    if (willExpand && !Array.isArray(runsByStudy[id])) {
      loadRuns(id);
    }
  };

  const fetchSetInitialSetupData = async (studyID: StudyId) => {
    try {
      const setupConfigData = await StudiesAPI.getSetupConfig(studyID);
      setInitialSetupData(setupConfigData);
    } catch (err) {
      if (err instanceof ApiError) show(err.message, 'error');
      else show('Failed to load setup config template!', 'error');
    }
  };

  const fetchSetInitialRunConfigData = async (studyID: StudyId) => {
    try {
      const runConfigData = await StudiesAPI.runConfigFormParams(studyID);
      setInitialRunConfigData(runConfigData);
    } catch (err) {
      if (err instanceof ApiError) show(err.message, 'error');
      else show('Failed to load initial runconfig data!', 'error');
    }
  };

  const openDetail = (id: StudyId) => {
    setDetailStudy(id);
    withLoading((async () => {
      try {
        await fetchSetInitialSetupData(id);
        await fetchSetInitialRunConfigData(id);
        let currentStep: 1 | 2 | 3 | 4 = 2;
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
  };

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
        if (err instanceof ApiError) show(err.message, 'error');
        else show('Failed to create study', 'error');
      }
    }, 'Creating study…');
  };

  const runStudy = async (id: StudyId, forceRerun: boolean) => {
    await withLoading(async () => {
      try {
        await StudiesAPI.run(id, forceRerun);
        show('Run started', 'success');
        setStatuses((s) => ({ ...s, [id]: 'running' } as any));
        setDetailOpen(false);
      } catch (err) {
        if (err instanceof ApiError) show(err.message, 'error');
        else show('Failed to start run', 'error');
      }
    }, 'Starting run…');
  };

  const cancelRun = async (id: StudyId, force: boolean = false) => {
    await withLoading(async () => {
      try {
        if (force) {
          await StudiesAPI.forceCancel(id);
          show('Run force-killed', 'success');
        } else {
          await StudiesAPI.cancel(id);
          show('Cancelling run (graceful)...', 'success');
        }
        setStatuses((s) => ({ ...s, [id]: force ? 'cancelled' : 'cancelling' } as any));
      } catch (err) {
        if (err instanceof ApiError) show(err.message, 'error');
        else show('Failed to cancel run', 'error');
      }
    }, force ? 'Force killing…' : 'Cancelling…');
  };

  const requestDelete = (id: StudyId) => { setDetailStudy(id); setDeleteOpen(true); };
  const confirmDelete = async () => {
    if (!detailStudy) return;
    await withLoading(async () => {
      try {
        await StudiesAPI.delete(detailStudy);
        show(`Deleted "${detailStudy}"`, 'success');
        setDeleteOpen(false);
        setDetailStudy(null);
        await load();
      } catch (err) {
        if (err instanceof ApiError) show(err.message, 'error');
        else show('Failed to delete study', 'error');
      }
    }, 'Deleting…');
  };

  const openLogs = async (id: StudyId) => {
    setLogs('Loading logs...');
    setDetailStudy(id);
    setLogsOpen(true);
    try {
      const l = await StudiesAPI.logs(id);
      setLogs(l);
    } catch {
      setLogs('Failed to load logs');
    }
  };

  const downloadFullLog = async (id: StudyId) => {
    try {
      const blob = await StudiesAPI.fullLog(id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = `${id}_log.txt`; a.click(); URL.revokeObjectURL(url);
    } catch {
      setLogs('Failed to download full log');
    }
  };

  const openRunResultSelector = (studyId: StudyId, run: StudyRun, type: Result) => {
    const years = Object.keys(run.timepoints || {}).sort();
    if (!years.length) {
      show('This run has no timepoints to open.', 'error');
      return;
    }
    const firstYear = years[0];
    const tps = run.timepoints[firstYear] || [];
    setSelectorContext({ studyId, runId: run.run_id, timepoints: run.timepoints, type });
    setSelectedYear(firstYear);
    setSelectedTimepoint(tps[0] || '');
    setResultsSelectorOpen(true);
  };

  const onYearChange = (nextYear: string) => {
    setSelectedYear(nextYear);
    const tps = selectorContext?.timepoints[nextYear] || [];
    setSelectedTimepoint(tps[0] || '');
  };

  const openSelectedResult = async () => {
    if (!selectorContext || !selectedYear || !selectedTimepoint) return;
    const { studyId, runId, type } = selectorContext;
    if (type === Result.Map) {
      const url = StudiesAPI.runMapUrl(studyId, runId, selectedYear, selectedTimepoint);
      window.open(url, '_blank', 'noopener,noreferrer');
    } else if (type === Result.Report) {
      try {
        const blob = await StudiesAPI.runReport(studyId, runId, selectedYear, selectedTimepoint);
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `final_report_${studyId}_${runId}_${selectedYear}_${selectedTimepoint}.pdf`;
        a.click();
        URL.revokeObjectURL(url);
      } catch { show('Download failed', 'error'); }
    }
    setResultsSelectorOpen(false);
  };

  const openRunMultiyear = (studyId: StudyId, runId: RunID) => {
    const url = StudiesAPI.runMultiyearUrl(studyId, runId);
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  const downloadRunArchive = async (studyId: StudyId, runId: RunID) => {
    await withLoading(async () => {
      try {
        const blob = await StudiesAPI.runArchiveDownload(studyId, runId);
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = `${studyId}_${runId}.zip`; a.click(); URL.revokeObjectURL(url);
      } catch (err) {
        if (err instanceof ApiError) show(err.message, 'error');
        else show('Download failed', 'error');
      }
    }, 'Preparing archive…');
  };

  const downloadRunConfig = async (studyId: StudyId, runId: RunID) => {
    try {
      const blob = await StudiesAPI.runConfigDownload(studyId, runId);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = `${studyId}_${runId}_config.yaml`; a.click(); URL.revokeObjectURL(url);
    } catch (err) {
      if (err instanceof ApiError) show(err.message, 'error');
      else show('Download failed', 'error');
    }
  };

  const runKey = (studyId: StudyId, runId: RunID) => `${studyId}|${runId}`;

  const toggleRunConfig = async (studyId: StudyId, runId: RunID) => {
    const key = runKey(studyId, runId);
    const willOpen = !configOpen[key];
    setConfigOpen(prev => ({ ...prev, [key]: willOpen }));
    if (!willOpen) return;
    // Fire config + apsim fetches in parallel on first open.
    if (typeof configByRun[key] !== 'string') {
      setConfigByRun(prev => ({ ...prev, [key]: 'loading' }));
      StudiesAPI.runConfigText(studyId, runId)
        .then(text => setConfigByRun(prev => ({ ...prev, [key]: text })))
        .catch(() => setConfigByRun(prev => ({ ...prev, [key]: 'error' })));
    }
    if (!apsimByRun[key] || apsimByRun[key] === 'error') {
      setApsimByRun(prev => ({ ...prev, [key]: 'loading' }));
      StudiesAPI.runApsim(studyId, runId)
        .then(data => setApsimByRun(prev => ({ ...prev, [key]: data })))
        .catch(() => setApsimByRun(prev => ({ ...prev, [key]: 'error' })));
    }
  };

  // Top-level config keys to auto-collapse in the YAML viewer because they
  // contain hundreds of entries that drown out the more interesting params.
  const collapsedConfigKeys = [
    'region',
    'regions',
    'scripts',
    'packaging_params',
    'eval_params',
    'apsim_execution',
    'years',
    'timepoints',
  ];

  const confirmDeleteRun = async () => {
    if (!deleteRunCtx) return;
    const { studyId, runId } = deleteRunCtx;
    await withLoading(async () => {
      try {
        await StudiesAPI.deleteRun(studyId, runId);
        show(`Deleted run "${runId}"`, 'success');
        await loadRuns(studyId);
      } catch (err) {
        if (err instanceof ApiError) show(err.message, 'error');
        else show('Failed to delete run', 'error');
      } finally {
        setDeleteRunCtx(null);
      }
    }, 'Deleting run…');
  };

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
        if (err instanceof ApiError) show(err.message, 'error');
        else show('Failed to duplicate study!', 'error');
      }
      await load();
    }, 'Duplicating study…');
  };

  const renderRunsPanel = (id: StudyId) => {
    const entry = runsByStudy[id];
    if (entry === 'loading' || !entry) {
      return (
        <div className="runs-panel-empty">
          <div className="loading" />
          <span style={{ marginLeft: 10 }}>Loading runs…</span>
        </div>
      );
    }
    if (entry === 'error') {
      return (
        <div className="runs-panel-empty">
          <span>Could not load runs.</span>
          <button className="btn btn-sm btn-secondary" style={{ marginLeft: 10 }} onClick={() => loadRuns(id)}>Retry</button>
        </div>
      );
    }
    if (!entry.length) {
      return (
        <div className="runs-panel-empty">
          <strong>No archived runs yet.</strong>
          <span style={{ marginLeft: 8 }}>
            A snapshot is created at the end of every successful pipeline run. Run the study to create one.
          </span>
        </div>
      );
    }
    return (
      <div className="runs-panel">
        <h4 className="runs-panel-title">Archived Runs</h4>
        <div className="runs-grid">
          <div className="runs-grid-head">
            <div></div>
            <div>Created</div>
            <div>Run ID</div>
            <div>Files</div>
            <div>Results</div>
          </div>
          {entry.map(run => {
            const key = runKey(id, run.run_id);
            const cfgOpen = !!configOpen[key];
            const cfg = configByRun[key];
            // Once the config viewer has ever been opened we keep its block
            // mounted and only flip visibility via CSS so toggling is instant
            // (no remount of the YAML tree's hundreds of nested nodes).
            const cfgMounted = cfgOpen || cfg !== undefined;
            return (
              <Fragment key={run.run_id}>
                <div className="runs-grid-row">
                  <div>
                    <button
                      className="row-toggle"
                      aria-label={cfgOpen ? 'Hide config' : 'Show config'}
                      aria-expanded={cfgOpen}
                      onClick={() => toggleRunConfig(id, run.run_id)}
                    >
                      <span className={`chevron ${cfgOpen ? 'open' : ''}`}>▶</span>
                    </button>
                  </div>
                  <div>{formatDate(run.created_at)}</div>
                  <div><code style={{ fontSize: 12 }}>{run.run_id}</code></div>
                  <div>
                    {run.file_count ?? '—'}
                    <span style={{ color: '#718096', marginLeft: 6 }}>({formatSize(run.size_bytes)})</span>
                  </div>
                  <div className="runs-grid-actions">
                    <button
                      className="btn btn-xs btn-primary"
                      onClick={() => openRunResultSelector(id, run, Result.Map)}
                      disabled={!Object.keys(run.timepoints || {}).length}
                    >Map</button>
                    <button
                      className="btn btn-xs btn-secondary"
                      onClick={() => openRunResultSelector(id, run, Result.Report)}
                      disabled={!Object.keys(run.timepoints || {}).length}
                    >Report</button>
                    <button
                      className="btn btn-xs btn-secondary"
                      onClick={() => openRunMultiyear(id, run.run_id)}
                      disabled={!run.has_multiyear_report}
                    >Multiyear</button>
                    <SplitButton
                      label="Download"
                      variant="secondary"
                      size="xs"
                      onPrimary={() => downloadRunArchive(id, run.run_id)}
                      menu={[
                        { label: 'Run archive (.zip)', onClick: () => downloadRunArchive(id, run.run_id) },
                        { label: 'config.yaml', onClick: () => downloadRunConfig(id, run.run_id) },
                      ]}
                    />
                    <button
                      className="btn btn-xs btn-secondary btn-delete"
                      title="Delete this archived run"
                      onClick={() => setDeleteRunCtx({ studyId: id, runId: run.run_id })}
                    >🗑</button>
                  </div>
                </div>
                {cfgMounted && (
                  <div className="runs-grid-config" style={cfgOpen ? undefined : { display: 'none' }}>
                    <div className="run-config-viewer">
                      <div className="run-config-header">
                        <strong>config.yaml</strong>
                        <button
                          className="btn btn-xs btn-secondary"
                          onClick={() => downloadRunConfig(id, run.run_id)}
                        >Download</button>
                      </div>
                      {cfg === 'loading' || cfg === undefined ? (
                        <div className="run-config-empty">
                          <div className="loading" />
                          <span style={{ marginLeft: 10 }}>Loading config…</span>
                        </div>
                      ) : cfg === 'error' ? (
                        <div className="run-config-empty">
                          Could not load config.
                          <button className="btn btn-xs btn-secondary" style={{ marginLeft: 10 }} onClick={() => toggleRunConfig(id, run.run_id)}>Retry</button>
                        </div>
                      ) : (
                        <div className="run-config-tree">
                          <YamlTree text={cfg} collapsedKeys={collapsedConfigKeys} />
                        </div>
                      )}

                      <div className="run-config-header" style={{ marginTop: 16 }}>
                        <strong>APSIM templates used</strong>
                        {apsimByRun[key] && apsimByRun[key] !== 'loading' && apsimByRun[key] !== 'error' && (apsimByRun[key] as ApsimManifest).manifest.filter_column && (
                          <span style={{ color: '#718096', fontSize: 11 }}>
                            mapped via column <code>{(apsimByRun[key] as ApsimManifest).manifest.filter_column}</code>
                          </span>
                        )}
                      </div>
                      {apsimByRun[key] === 'loading' || apsimByRun[key] === undefined ? (
                        <div className="run-config-empty">
                          <div className="loading" />
                          <span style={{ marginLeft: 10 }}>Loading APSIM mapping…</span>
                        </div>
                      ) : apsimByRun[key] === 'error' ? (
                        <div className="run-config-empty">Could not load APSIM mapping.</div>
                      ) : (apsimByRun[key] as ApsimManifest).files.length === 0 ? (
                        <div className="run-config-empty" style={{ minHeight: 60 }}>
                          No APSIM templates were captured for this run.
                        </div>
                      ) : (
                        <div className="apsim-files">
                          {(apsimByRun[key] as ApsimManifest).files.map(f => {
                            const fileManifest = (apsimByRun[key] as ApsimManifest).manifest.files.find(x => x.name === f.name);
                            const regions = fileManifest?.regions ?? [];
                            return (
                              <div className="apsim-file" key={f.name}>
                                <div className="apsim-file-info">
                                  <code className="apsim-file-name">{f.name}</code>
                                  <span className="apsim-file-meta">
                                    {formatSize(f.size)} · {regions.length} region{regions.length === 1 ? '' : 's'}
                                  </span>
                                </div>
                                <details className="apsim-file-regions">
                                  <summary>Show regions</summary>
                                  <div className="apsim-region-list">
                                    {regions.length === 0 ? <em>— no regions in mapping —</em> : regions.join(', ')}
                                  </div>
                                </details>
                                <a
                                  className="btn btn-xs btn-secondary"
                                  href={StudiesAPI.runApsimFileUrl(id, run.run_id, f.name)}
                                  download={f.name}
                                >Download</a>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </Fragment>
            );
          })}
        </div>
      </div>
    );
  };

  const table = useMemo(() => {
    if (!studies) return (
      <div className="empty-state">
        <div className="loading"></div>
        <p style={{ marginTop: '1rem' }}>Loading studies...</p>
      </div>
    );
    if (studies.length === 0) return (
      <div className="empty-state">
        <h3>No Studies Found</h3>
        <p>Create your first study to get started</p>
        <button className="btn btn-primary" style={{ marginTop: 15 }} onClick={() => setCreateOpen(true)}>Create Your First Study</button>
      </div>
    );
    return (
      <>
        <div className="table-container">
          <table className="table studies-table">
            <thead>
              <tr>
                <th style={{ width: 36 }}></th>
                <th>Study Name</th>
                <th style={{ width: 160 }}>Status</th>
                <th style={{ width: 200 }}>Latest Run</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {studies.map((id) => {
                const s = statuses[id] ?? 'pending';
                const canCancel = s === 'running' || s === 'validating' || s === 'cancelling';
                const canRun = s === 'ready' || s === 'failed' || s === 'completed' || s === 'cancelled';
                const canConfigure = s !== 'running';
                const isOpen = !!expanded[id];
                const runs = Array.isArray(runsByStudy[id]) ? (runsByStudy[id] as StudyRun[]) : null;
                const latest = runs?.[0];

                return (
                  <Fragment key={id}>
                    <tr className={isOpen ? 'study-row open' : 'study-row'}>
                      <td>
                        <button
                          className="row-toggle"
                          aria-label={isOpen ? 'Collapse' : 'Expand'}
                          aria-expanded={isOpen}
                          onClick={() => toggleExpand(id)}
                        >
                          <span className={`chevron ${isOpen ? 'open' : ''}`}>▶</span>
                        </button>
                      </td>
                      <td>
                        <div style={{ fontWeight: 500 }}>{id}</div>
                        {runs && runs.length > 0 && (
                          <div style={{ fontSize: 12, color: '#718096', marginTop: 2 }}>
                            {runs.length} archived run{runs.length === 1 ? '' : 's'}
                          </div>
                        )}
                      </td>
                      <td><StatusBadge status={s} /></td>
                      <td style={{ fontSize: 13, color: '#4a5568' }}>
                        {latest ? formatDate(latest.created_at) : (isOpen ? '—' : <span style={{ color: '#a0aec0' }}>expand to load</span>)}
                      </td>
                      <td>
                        <div className="actions-cell" style={{ flexWrap: 'wrap' }}>
                          <button className="btn btn-sm btn-primary" onClick={() => openDetail(id)} disabled={!canConfigure}>Configure</button>
                          <SplitButton
                            label="Run"
                            variant="success"
                            onPrimary={() => runStudy(id, false)}
                            disabled={!canRun}
                            menu={[
                              { label: 'Force Rerun', onClick: () => runStudy(id, true), tooltip: 'Rerun every step of the pipeline from scratch, ignoring already completed results.' }
                            ]}
                          />
                          {canCancel && (
                            <SplitButton
                              label="Cancel"
                              variant="danger"
                              onPrimary={() => cancelRun(id)}
                              menu={[
                                { label: 'Force Kill', onClick: () => cancelRun(id, true), tooltip: 'Immediately kills all processes without cleanup. May leave incomplete output files.' }
                              ]}
                            />
                          )}
                          <button className="btn btn-sm btn-secondary" onClick={() => openLogs(id)}>Logs</button>
                          <button className="btn btn-sm btn-secondary" onClick={() => { setDetailStudy(id); setDuplicateOpen(true); }}>Use as template</button>
                          <button className="btn btn-sm btn-secondary btn-delete" title="Delete study" onClick={() => requestDelete(id)}>🗑</button>
                        </div>
                      </td>
                    </tr>
                    {isOpen && (
                      <tr className="study-row-expanded">
                        <td colSpan={5}>{renderRunsPanel(id)}</td>
                      </tr>
                    )}
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 12 }}>
          <div style={{ fontSize: 13, color: '#4a5568' }}>
            Showing <strong>{studies.length ? (page - 1) * pageSize + 1 : 0}</strong>–
            <strong>{Math.min(page * pageSize, total)}</strong> of <strong>{total}</strong>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <label className="form-label" style={{ margin: 0 }}>Per page</label>
            <select className="form-input" value={pageSize} onChange={e => { setPage(1); setPageSize(Number(e.target.value)); }} style={{ width: 90, padding: '6px 8px', height: 34 }}>
              {[10, 20, 50, 100].map(n => <option key={n} value={n}>{n}</option>)}
            </select>
            <button className="btn btn-secondary btn-sm" onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}>← Prev</button>
            <button className="btn btn-secondary btn-sm" onClick={() => setPage(p => (p * pageSize < total ? p + 1 : p))} disabled={page * pageSize >= total}>Next →</button>
          </div>
        </div>
      </>
    );
  }, [studies, statuses, expanded, runsByStudy, configOpen, configByRun]);

  const handleSetupSubmission = async (payload: SetupSubmissionsPayload) => {
    if (!detailStudy) return;
    await withLoading(async () => {
      try {
        await StudiesAPI.uploadSetup(detailStudy, payload);
        show('Setup uploaded', 'success');
        await fetchSetInitialSetupData(detailStudy);
        await fetchSetInitialRunConfigData(detailStudy);
        setStep(3);
      } catch (err) {
        if (err instanceof ApiError) show(err.message, 'error');
        else show('Failed to upload setup', 'error');
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
        if (err instanceof ApiError) show(err.message, 'error');
        else show('Failed to upload run config', 'error');
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
      if (err instanceof ApiError) show(err.message, 'error');
      else show('Download failed', 'error');
    }
  };

  const yearOptions = selectorContext ? Object.keys(selectorContext.timepoints).sort() : [];
  const timepointOptions = selectorContext ? (selectorContext.timepoints[selectedYear] || []) : [];

  return (
    <div className="container">
      <Header />
      <div className="actions">
        <button className="btn btn-primary" onClick={() => setCreateOpen(true)}>Create New Study</button>
        <button className="btn btn-secondary" onClick={() => load()}>Refresh</button>
      </div>
      <div id="studiesContainer">{table}</div>

      {/* Create Study */}
      <Modal open={createOpen} onClose={() => setCreateOpen(false)} title="Create New Study">
        <div className="form-group">
          <label htmlFor="studyName" className="form-label">Study Name</label>
          <input id="studyName" className="form-input" placeholder="Enter study name" value={createName} onChange={e => setCreateName(e.target.value)} />
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
          <button className="btn btn-secondary" onClick={() => setCreateOpen(false)}>Cancel</button>
          <button className="btn btn-primary" onClick={createStudy}>Create Study</button>
        </div>
      </Modal>

      {/* Detail / Setup */}
      <Modal open={detailOpen} onClose={() => setDetailOpen(false)} title={`${detailStudy ?? ''} - Configuration`} width={800}>
        <Stepper step={step} onStepChange={setStep} />
        {step === 2 && <SetupStudyForm studyId={detailStudy ?? ''} onSubmit={handleSetupSubmission} initialData={initialSetupData} />}

        {step === 3 && <RunParamsForm
          key={detailStudy ?? 'no-study'}
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
            <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'space-between' }}>
              <button className="btn btn-secondary" onClick={async () => {
                if (!detailStudy) return;
                try {
                  const blob = await StudiesAPI.downloadRunConfigTemplate(detailStudy);
                  const url = URL.createObjectURL(blob);
                  const a = document.createElement('a');
                  a.href = url; a.download = 'config.yaml'; a.click(); URL.revokeObjectURL(url);
                } catch { show('Download failed', 'error'); }
              }}>Download Config</button>
              <div style={{ display: 'flex', gap: '0.75rem' }}>
                <button className="btn btn-secondary" onClick={() => setStep(3)}>Update Config</button>
                <button className="btn btn-success" onClick={() => detailStudy && runStudy(detailStudy, false)}>Run Study</button>
              </div>
            </div>
          </div>
        )}
      </Modal>

      {/* Results year/timepoint selector (per-run) */}
      <Modal open={resultsSelectorOpen} onClose={() => setResultsSelectorOpen(false)} title="Select Result Timepoint">
        {selectorContext && (
          <div style={{ fontSize: 13, color: '#4a5568', marginBottom: 12 }}>
            Run <code>{selectorContext.runId}</code> · study <strong>{selectorContext.studyId}</strong>
          </div>
        )}
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
        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
          <button className="btn btn-secondary" onClick={() => setResultsSelectorOpen(false)}>Cancel</button>
          <button className="btn btn-primary" onClick={openSelectedResult} disabled={!selectedYear || !selectedTimepoint}>Open</button>
        </div>
      </Modal>

      {/* Logs */}
      <Modal open={logsOpen} onClose={() => setLogsOpen(false)} title="Study Logs" width={1300}>
        <button onClick={handleCopyLogs} className="btn btn-sm btn-secondary">Copy logs</button>
        <button
          onClick={() => { if (detailStudy) downloadFullLog(detailStudy); }}
          className="btn btn-sm btn-secondary"
          style={{ marginLeft: '10px' }}
        >Download Full Log File</button>
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
          <input id="studyName" className="form-input" placeholder="Enter study name" value={createName} onChange={e => setCreateName(e.target.value)} />
        </div>
        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end' }}>
          <button className="btn btn-secondary" onClick={() => setDuplicateOpen(false)}>Cancel</button>
          <button className="btn btn-primary" onClick={() => { duplicateStudy(); setDuplicateOpen(false); }}>Create Study</button>
        </div>
      </Modal>

      {/* Delete Study */}
      <Modal open={deleteOpen} onClose={() => setDeleteOpen(false)} title="Delete Study">
        <p className="subtitle" style={{ marginTop: 4 }}>
          This will permanently remove the study folder and all of its archived runs. This cannot be undone.
        </p>
        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end', marginTop: 12 }}>
          <button className="btn btn-secondary" onClick={() => setDeleteOpen(false)}>Cancel</button>
          <button className="btn btn-danger" onClick={confirmDelete}>Confirm Delete</button>
        </div>
      </Modal>

      {/* Delete Run */}
      <Modal open={!!deleteRunCtx} onClose={() => setDeleteRunCtx(null)} title="Delete Archived Run">
        <p className="subtitle" style={{ marginTop: 4 }}>
          {deleteRunCtx && <>This will permanently remove the run <code>{deleteRunCtx.runId}</code> and all of its archived results.</>}
        </p>
        <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'flex-end', marginTop: 12 }}>
          <button className="btn btn-secondary" onClick={() => setDeleteRunCtx(null)}>Cancel</button>
          <button className="btn btn-danger" onClick={confirmDeleteRun}>Confirm Delete</button>
        </div>
      </Modal>

      <Toast />
      {isLoading && (
        <div
          role="alert"
          aria-live="polite"
          style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.35)', zIndex: 9999, display: 'flex', alignItems: 'center', justifyContent: 'center', padding: 16 }}
        >
          <div style={{ background: '#fff', borderRadius: 16, padding: '18px 22px', boxShadow: '0 10px 30px rgba(0,0,0,0.2)', display: 'flex', alignItems: 'center', gap: 12, minWidth: 220, justifyContent: 'center' }}>
            <div aria-hidden style={{ width: 28, height: 28, borderRadius: '50%', border: '3px solid #E5E7EB', borderTopColor: '#3B82F6', animation: 'spin 1s linear infinite' }} />
            <span style={{ fontSize: 14, color: '#374151' }}>{loadingText || 'Working…'}</span>
          </div>
        </div>
      )}
      <style>{`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}</style>
    </div>
  );
};

export default StudiesPage;
