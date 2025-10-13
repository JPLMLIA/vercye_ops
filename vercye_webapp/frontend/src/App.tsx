
import { Routes, Route } from 'react-router-dom';
import StudiesPage from '@/pages/StudiesPage';
import LAIPage from '@/pages/LAIPage';
import CropmasksPage from './pages/CropmasksPage';

const App = () => {
  return (
    <Routes>
      <Route path="/" element={<StudiesPage />} />
      <Route path="/lai" element={<LAIPage />} />
      <Route path="/cropmasks" element={<CropmasksPage />} />
    </Routes>
  );
}

export default App
