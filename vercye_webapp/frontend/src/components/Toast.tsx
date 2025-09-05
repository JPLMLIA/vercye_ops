import { useEffect, useState, useCallback, useMemo } from 'react';

const useToast = () => {
  const [msg, setMsg] = useState<string | null>(null);
  const [type, setType] = useState<'success' | 'error' | 'info'>('info');

  useEffect(() => {
    if (!msg) return;
    const t = setTimeout(() => setMsg(null), 3000);
    return () => clearTimeout(t);
  }, [msg]);

  const show = useCallback(
    (m: string, t: 'success' | 'error' | 'info' = 'info') => {
      setMsg(m);
      setType(t);
    },
    []
  );

  const Toast = useMemo(
    () =>
      function Toast() {
        if (!msg) return null;
        return (
          <div className={`toast alert-${type}`} role="status">
            {msg}
          </div>
        );
      },
    [msg, type]
  );

  return { show, Toast };
};

export default useToast;
