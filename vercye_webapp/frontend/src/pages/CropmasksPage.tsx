import { CropmasksAPI } from "@/api/cropmasks";
import FileUpload from "@/components/FileUpload"
import Header from "@/components/Header"
import Modal from "@/components/Modal"
import useToast from "@/components/Toast";
import { useEffect, useState } from "react";

const CropmasksPage = () => {
  const [createOpen, setCreateOpen] = useState<boolean>(false)
  const [cropmaskCreateName, setCropmaskCreateName] = useState<string | null>(null)
  const [cropmaskFiles, setCropmaskFiles] = useState<File[]>([])
  const [cropmasks, setCropmasks] = useState<{id: string}[] | null>(null)

  const { show, Toast } = useToast();

   const loadCropmasks = async () => {
      try {
          const res = await CropmasksAPI.list();
          setCropmasks(res);
      } catch {
          setCropmasks([]);
          show('Failed to load cropmasks', 'error');
      }
    };

  useEffect(() => {
      loadCropmasks();
      const iv = setInterval(async () => {
        loadCropmasks()
      }, 10000);
      return () => clearInterval(iv);
    }, []);

  const handleCropmaskSubmit = async () => {
    if (cropmaskFiles.length === 0) {
      show('Must select a cropmask file first.', 'error')
      return
    }

    if (!cropmaskCreateName) {
      show('Must set a cropmask name first.', 'error')
      return
    }

    try {
      await CropmasksAPI.create(cropmaskFiles[0], cropmaskCreateName);
      show('Cropmask uploaded', 'success');
      setCropmaskFiles([]);
      setCropmaskCreateName("");
      setCreateOpen(false);
    } catch {
      alert("Failed to upload cropmask!");
    }
  }

  const cropMasksTable = () => {
    if (!cropmasks)
      return (
        <div className="empty-state">
          <div className="loading"></div>
          <p style={{ marginTop: '1rem' }}>Loading available cropmasks data...</p>
        </div>
      );
    if (!cropmasks.length)
      return (
        <div className="empty-state">
          <h3>No Cropmasks available</h3>
          <p>Check back later.</p>
        </div>
      );
    return (
      <div className="table-container">
        <table className="table">
          <thead>
            <tr>
              <th>Cropmask Name</th>
            </tr>
          </thead>
          <tbody>
            {cropmasks.map((e) => (
              <tr key={e.id}>
                <td>{e.id}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }

  return (
      <div className="container">
        <Header />
        <div className="actions">
          <button className="btn btn-primary" onClick={() => setCreateOpen(true)}>
            ✚ Upload Cropmask
          </button>
        </div>


        {cropMasksTable()}


      <Modal open={createOpen} onClose={() => setCreateOpen(false)} title="Upload a Cropmask" width={800}>
        <div className="form-group">
          <label htmlFor="studyName" className="form-label">Cropmask Name</label>
          <input id="studyName" className="form-input" placeholder="Enter cropmask name" value={cropmaskCreateName? cropmaskCreateName : ''} onChange={e=>setCropmaskCreateName(e.target.value)}/>
           <p className="subtitle" style={{ marginTop: 4 }}>
              The Cropmask must be a binary (0/1) geotiff. Use 0 for the non-crop class and 1 for the pixels containing the crop of interest. It is reccomended to upload a compressed file.
            </p>
        </div>

         <div className="form-group">
            <label className="form-label">Cropmask File (.tif)</label>
             <FileUpload
                id="cropmaskFile"
                accept=".tif"
                label="📁 Choose cropmask file (.tif)"
                value={cropmaskFiles}
                onChange={setCropmaskFiles}
              />
              <p className="subtitle" style={{ marginTop: 4 }}>
              The Cropmask must be a binary (0/1) geotiff. Use 0 for the non-crop class and 1 for the pixels containing the crop of interest. It is reccomended to upload a compressed file.
            </p>
        </div>

        <div style={{ display:'flex', gap: '0.75rem', justifyContent:'flex-end' }}>
          <button className="btn btn-secondary" onClick={()=>setCreateOpen(false)}>Cancel</button>
          <button className="btn btn-primary" onClick={handleCropmaskSubmit}>Upload</button>
        </div>
      </Modal>
      <Toast />
    </div>
    )
}

export default CropmasksPage
