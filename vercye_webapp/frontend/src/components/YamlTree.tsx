import { memo, useMemo, useState } from 'react';
import yaml from 'js-yaml';

type Json = unknown;

const isObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === 'object' && v !== null && !Array.isArray(v);

const formatLeaf = (v: unknown): { text: string; cls: string } => {
  if (v === null || v === undefined) return { text: 'null', cls: 'yt-null' };
  if (typeof v === 'string') return { text: v, cls: 'yt-string' };
  if (typeof v === 'number') return { text: String(v), cls: 'yt-number' };
  if (typeof v === 'boolean') return { text: String(v), cls: 'yt-bool' };
  return { text: JSON.stringify(v), cls: 'yt-string' };
};

const sizeOf = (v: unknown): number => {
  if (Array.isArray(v)) return v.length;
  if (isObject(v)) return Object.keys(v).length;
  return 0;
};

interface NodeProps {
  k: string;
  v: Json;
  depth: number;
  /** Depth strictly below which a node mounts already open. */
  defaultOpenDepth: number;
  /** Top-level keys that should mount collapsed regardless of defaultOpenDepth. */
  collapsedTopKeys?: ReadonlySet<string>;
}

const Node = memo<NodeProps>(({ k, v, depth, defaultOpenDepth, collapsedTopKeys }) => {
  const expandable = Array.isArray(v) || isObject(v);
  const forcedClosed = depth === 0 && collapsedTopKeys?.has(k);
  const [open, setOpen] = useState(!forcedClosed && depth < defaultOpenDepth);

  if (!expandable) {
    const leaf = formatLeaf(v);
    return (
      <div className="yt-row" style={{ paddingLeft: depth * 14 }}>
        <span className="yt-key">{k}:</span>
        <span className={`yt-value ${leaf.cls}`} title={leaf.text}>{leaf.text}</span>
      </div>
    );
  }

  const n = sizeOf(v);
  const previewLabel = Array.isArray(v) ? `[${n} item${n === 1 ? '' : 's'}]` : `{${n} key${n === 1 ? '' : 's'}}`;

  return (
    <div>
      <div
        className="yt-row yt-row-toggle"
        style={{ paddingLeft: depth * 14 }}
        onClick={() => setOpen(o => !o)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setOpen(o => !o); } }}
      >
        <span className={`yt-chevron ${open ? 'open' : ''}`}>▶</span>
        <span className="yt-key">{k}{Array.isArray(v) ? '' : ':'}</span>
        {!open && <span className="yt-preview">{previewLabel}</span>}
      </div>
      {open && (
        <div>
          {Array.isArray(v)
            ? (v as Json[]).map((item, i) => (
                <Node key={i} k={String(i)} v={item} depth={depth + 1} defaultOpenDepth={defaultOpenDepth} collapsedTopKeys={collapsedTopKeys} />
              ))
            : Object.entries(v as Record<string, Json>).map(([ck, cv]) => (
                <Node key={ck} k={ck} v={cv} depth={depth + 1} defaultOpenDepth={defaultOpenDepth} collapsedTopKeys={collapsedTopKeys} />
              ))}
        </div>
      )}
    </div>
  );
});
Node.displayName = 'YamlNode';

interface YamlTreeProps {
  text: string;
  /** Depth strictly below which nodes are open at first mount. Default = 1
   * (only top-level Object keys are expanded; nested children are collapsed
   * so we do not mount thousands of nodes for a large config). */
  defaultOpenDepth?: number;
  /** Top-level keys that should mount collapsed regardless of defaultOpenDepth. */
  collapsedKeys?: ReadonlyArray<string>;
}

const YamlTree: React.FC<YamlTreeProps> = memo(({ text, defaultOpenDepth = 1, collapsedKeys }) => {
  const collapsedSet = useMemo(
    () => (collapsedKeys && collapsedKeys.length ? new Set(collapsedKeys) : undefined),
    [collapsedKeys],
  );
  const parsed = useMemo(() => {
    try {
      return { ok: true as const, data: yaml.load(text) };
    } catch (err: any) {
      return { ok: false as const, error: String(err?.message || err) };
    }
  }, [text]);

  if (!parsed.ok) {
    return (
      <div className="yt-fallback">
        <div className="yt-error">Could not parse YAML — showing raw text. ({parsed.error})</div>
        <pre className="yt-raw">{text}</pre>
      </div>
    );
  }

  const data = parsed.data;
  if (data === undefined || data === null) {
    return <div className="yt-fallback yt-empty">Empty config.</div>;
  }

  if (!isObject(data) && !Array.isArray(data)) {
    const leaf = formatLeaf(data);
    return <div className={`yt-row yt-value ${leaf.cls}`}>{leaf.text}</div>;
  }

  return (
    <div className="yaml-tree">
      {Array.isArray(data)
        ? (data as Json[]).map((item, i) => (
            <Node key={i} k={String(i)} v={item} depth={0} defaultOpenDepth={defaultOpenDepth} collapsedTopKeys={collapsedSet} />
          ))
        : Object.entries(data as Record<string, Json>).map(([k, v]) => (
            <Node key={k} k={k} v={v} depth={0} defaultOpenDepth={defaultOpenDepth} collapsedTopKeys={collapsedSet} />
          ))}
    </div>
  );
});
YamlTree.displayName = 'YamlTree';

export default YamlTree;
