import React, { useState } from "react";

interface StepperProps {
  step: 1 | 2 | 3 | 4;
  onStepChange?: (step: 1 | 2 | 3 | 4) => void;
}

const Stepper = ({ step, onStepChange }: StepperProps) => {
  const width = `${(step / 4) * 100}%`;

  const [backsteps, setBacksteps] = useState(0)

  const handleBack = () => {
    if (step > 1 && onStepChange) {
      onStepChange((step - 1) as 1 | 2 | 3 | 4);
      setBacksteps(backsteps+1)
    }
  };

  const handleForward = () => {
    if (step < 4 && onStepChange && backsteps != 0) {
      onStepChange((step + 1) as 1 | 2 | 3 | 4)
      setBacksteps(backsteps-1)
    }
  };

  return (
    <div className="stepper-wrapper">
      <div className="stepper-header">
          <button
            className="btn btn-secondary btn-sm step-back-btn"
            onClick={handleBack}
            style={{marginBottom: "10px"}}
            disabled={step <= 2}
          >
            ←  Previous
          </button>
          <button
            className="btn btn-secondary btn-sm step-back-btn"
            onClick={handleForward}
            style={{marginBottom: "10px", marginLeft: "10px"}}
            disabled={(step >= 4 || backsteps == 0)}
          >
            Next →
          </button>
        <div className="step-indicator">
          <div className={`step ${step >= 1 ? "completed" : ""}`}>
            <div className="step-circle">1</div>
            <div className="step-label">Create</div>
          </div>
          <div
            className={`step ${
              step >= 2 ? (step > 2 ? "completed" : "active") : ""
            }`}
          >
            <div className="step-circle">2</div>
            <div className="step-label">Setup</div>
          </div>
          <div
            className={`step ${
              step >= 3 ? (step > 3 ? "completed" : "active") : ""
            }`}
          >
            <div className="step-circle">3</div>
            <div className="step-label">Run Parameters</div>
          </div>
          <div className={`step ${step >= 4 ? "completed active" : ""}`}>
            <div className="step-circle">4</div>
            <div className="step-label">Ready</div>
          </div>
          <div className="progress-line" style={{ width }}></div>
        </div>
      </div>
    </div>
  );
};

export default Stepper;
