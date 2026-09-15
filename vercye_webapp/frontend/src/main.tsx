
import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import './styles/dashboard.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {/* The v7_* future flags are gone: they were v6 opt-ins for what v7 now does by
        default, which is the behaviour this app was already running with. */}
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
