import { Routes, Route } from 'react-router-dom';
import StudiesPage from '@/pages/StudiesPage';
import LAIPage from '@/pages/LAIPage';
import ResultsMapPage from '@/pages/ResultsMapPage';
import CropmasksPage from './pages/CropmasksPage';

const App = () => {
  return (
    <Routes>
      <Route path="/" element={<StudiesPage />} />
      <Route path="/lai" element={<LAIPage />} />
      <Route path="/cropmasks" element={<CropmasksPage />} />
      {/* Year and timepoint live in the path so a particular season is linkable;
          level, raster and selected region ride along as query params. */}
      <Route path="/studies/:studyId/results/:year/:timepoint" element={<ResultsMapPage />} />
      <Route path="/studies/:studyId/results" element={<ResultsMapPage />} />
      <Route path="/studies/:studyId/runs/:runId/results/:year/:timepoint" element={<ResultsMapPage />} />
      <Route path="/studies/:studyId/runs/:runId/results" element={<ResultsMapPage />} />
    </Routes>
  );
}

export default App
