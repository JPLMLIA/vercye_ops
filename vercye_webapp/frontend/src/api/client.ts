
const API_BASE = '/api';

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let message = `Request failed with ${res.status}`;
    try {
      const data = await res.json();
      if (data?.detail) {
        message = data.detail;
      } else {
        message = JSON.stringify(data);
      }
    } catch {
      // fallback if response wasn't JSON
      const text = await res.text().catch(() => '');
      if (text) message = text;
    }
    throw new ApiError(message, res.status);
  }

  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) {
    return (await res.json()) as T;
  }
  if (ct.startsWith('text/')) {
    return (await res.text()) as unknown as T;
  }
  return (await res.blob()) as unknown as T;
}

export const http = {
  get: <T>(path: string) => {
    const url = `${API_BASE}${path}`;
    return fetch(url, {
      cache: 'no-store',
      headers: {
        'Cache-Control': 'no-cache',
        Pragma: 'no-cache',
      },
    }).then(handle<T>);
  },
  post: <T, B = unknown>(path: string, body?: B) =>
    fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers: body instanceof FormData ? undefined : { 'Content-Type': 'application/json' },
      body: body instanceof FormData ? body : body ? JSON.stringify(body) : undefined,
    }).then(handle<T>),
  put: <T, B = unknown>(path: string, body?: B) =>
    fetch(`${API_BASE}${path}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: body ? JSON.stringify(body) : undefined,
    }).then(handle<T>),
  del: <T>(path: string) => fetch(`${API_BASE}${path}`, { method: 'DELETE' }).then(handle<T>),
  download: (path: string) =>
    fetch(`${API_BASE}${path}`).then(async (res) => {
      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new ApiError(text || `Request failed with ${res.status}`, res.status);
      }
      return await res.blob();
    }),
};
