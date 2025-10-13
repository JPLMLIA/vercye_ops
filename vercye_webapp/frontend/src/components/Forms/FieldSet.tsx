interface FieldsetProps {
  legend: string;
  children: React.ReactNode;
  hint?: string;
}

const Fieldset: React.FC<FieldsetProps> = ({ legend, children, hint }) => (
  <section
    style={{
      background: "var(--white)",
      border: "1px solid var(--gray-200)",
      borderRadius: 12,
      boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
      padding: "1.25rem",
      marginBottom: "1.25rem",
    }}
  >
    <div style={{ marginBottom: ".75rem" }}>
      <h3
        style={{
          margin: 0,
          fontFamily: "Roboto, sans-serif",
          color: "var(--dark-primary)",
          fontSize: "1.1rem",
          fontWeight: 600,
        }}
      >
        {legend}
      </h3>
      {hint ? (
        <p className="subtitle" style={{ marginTop: 4 }}>
          {hint}
        </p>
      ) : null}
    </div>
    {children}
  </section>
);

export default Fieldset
