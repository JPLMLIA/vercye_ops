
import { ReactNode } from 'react';

interface ModalProps {
  open: boolean;
  onClose: () => void;
  title: string;
  width?: number | string;
  children: ReactNode;
}

const Modal = ({ open, onClose, title, width = 600, children }: ModalProps) => {
  return (
    <div className={`modal ${open ? 'show' : ''}`} role="dialog" aria-hidden={!open}>
      <div className="modal-content" style={{ maxWidth: width }}>
        <button className="modal-close" onClick={onClose} aria-label="Close">&times;</button>
        <h2 className="modal-title">{title}</h2>
        {children}
      </div>
    </div>
  );
}

export default Modal
