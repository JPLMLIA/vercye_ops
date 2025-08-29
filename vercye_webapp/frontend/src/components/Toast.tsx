
import { useEffect, useState } from 'react';

const useToast = () => {
  const [msg, setMsg] = useState<string | null>(null);
  const [type, setType] = useState<'success' | 'error' | 'info'>('info');
  useEffect(() => {
    if (!msg) return;
    const t = setTimeout(() => setMsg(null), 3000);
    return () => clearTimeout(t);
  }, [msg]);
  return {
    show: (m: string, t: 'success' | 'error' | 'info' = 'info') => { setMsg(m); setType(t); },
    Toast: () => msg ? <div className="toast alert-info" role="status">{msg}</div> : null
  };
}

export default useToast
