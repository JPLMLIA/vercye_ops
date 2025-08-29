
interface StepperProps {
  step: 1 | 2 | 3 | 4;
}

const Stepper = ({ step }: StepperProps) => {
  const width = `${(step / 4) * 100}%`;
  return (
    <div className="step-indicator">
      <div className={`step ${step >= 1 ? 'completed' : ''}`}><div className="step-circle">1</div><div className="step-label">Create</div></div>
      <div className={`step ${step >= 2 ? (step>2 ? 'completed' : 'active') : ''}`}><div className="step-circle">2</div><div className="step-label">Setup</div></div>
      <div className={`step ${step >= 3 ? (step>3 ? 'completed' : 'active') : ''}`}><div className="step-circle">3</div><div className="step-label">Run Parameters</div></div>
      <div className={`step ${step >= 4 ? 'completed active' : ''}`}><div className="step-circle">4</div><div className="step-label">Ready</div></div>
      <div className="progress-line" style={{ width }}></div>
    </div>
  );
}
export default Stepper
