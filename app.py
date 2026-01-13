import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import streamlit as st

# ---------------------
# Funzioni di calcolo
# ---------------------

def curva_caratteristica(Q, B, C):
    """
    Curva caratteristica del pozzo:
        s(Q) = B * Q + C * Q^2
    dove:
        s = abbassamento (m)
        Q = portata (unità coerente: l/s, m³/h, ecc.)
    """
    return B * Q + C * Q**2


def portata_critica(B, C):
    """
    Per s(Q) = B*Q + C*Q^2, la portata alla quale B*Q = C*Q^2 è Qcrit = B/C.
    Restituisce np.nan se C <= 0.
    """
    return B / C if C > 0 else np.nan


def run_analysis(Q_vals, s_vals):
    """
    Qui mettiamo tutta la logica di calcolo che prima stava in calcola()
    (ma senza Tkinter).
    Q_vals e s_vals sono liste o array di float.
    """
    Q = np.array(Q_vals, dtype=float)
    s = np.array(s_vals, dtype=float)

    # Controlli base
    mask_valid = (Q > 0) & (s > 0)
    Q = Q[mask_valid]
    s = s[mask_valid]

    if len(Q) < 3:
        raise ValueError("Servono almeno 3 gradini con Q > 0 e s > 0.")

    # Abbassamento specifico e portata specifica
    s_spec = s / Q      # s/Q
    Q_spec = Q / s      # Q/s

    # Fit della curva caratteristica: s = B*Q + C*Q^2
    popt, _ = curve_fit(curva_caratteristica, Q, s)
    B, C = popt

    # Delta H totale (tra primo e ultimo gradino usato)
    delta_H = s[-1] - s[0]

    # Portata critica
    Qcrit = portata_critica(B, C)
    s_crit = curva_caratteristica(Qcrit, B, C) if np.isfinite(Qcrit) else np.nan

    # DataFrame risultati (per Excel e tabella)
    df = pd.DataFrame({
        "Gradino": np.arange(1, len(Q) + 1),
        "Q": Q,
        "s": s,
        "s/Q": s_spec,
        "Q/s": Q_spec,
        "B": [B] * len(Q),
        "C": [C] * len(Q)
    })

    # ----------------- Grafici matplotlib -----------------

    # 1) Curva caratteristica
    Q_range = np.linspace(Q.min(), Q.max(), 200)
    s_fit = curva_caratteristica(Q_range, B, C)

    fig1, ax1 = plt.subplots()
    ax1.scatter(Q, s, label="Dati sperimentali")
    ax1.plot(Q_range, s_fit, label=f"Fit: s = {B:.2f}Q + {C:.2f}Q²")
    if np.isfinite(Qcrit):
        ax1.axvline(Qcrit, linestyle="--", label=f"Qcrit = {Qcrit:.2f}")
    ax1.set_title("Curva caratteristica s(Q)")
    ax1.set_xlabel("Q")
    ax1.set_ylabel("s (m)")
    ax1.grid(True)
    ax1.legend()

    # 2) s/Q vs Q
    s_spec = s / Q
    coeff_sQ = np.polyfit(Q, s_spec, 1)
    sQ_fit = np.polyval(coeff_sQ, Q_range)

    fig2, ax2 = plt.subplots()
    ax2.scatter(Q, s_spec, label="Dati sperimentali")
    ax2.plot(Q_range, sQ_fit, label=f"Fit: s/Q = {coeff_sQ[0]:.2f}Q + {coeff_sQ[1]:.2f}")
    ax2.set_title("Abbassamento specifico s/Q vs Q")
    ax2.set_xlabel("Q")
    ax2.set_ylabel("s/Q")
    ax2.grid(True)
    ax2.legend()

    # 3) Q/s vs s
    Q_spec = Q / s
    coeff_Qs = np.polyfit(Q_spec, s, 1)
    Qs_range = np.linspace(Q_spec.min(), Q_spec.max(), 200)
    s_Qs_fit = np.polyval(coeff_Qs, Qs_range)

    fig3, ax3 = plt.subplots()
    ax3.scatter(Q_spec, s, label="Dati sperimentali")
    ax3.plot(Qs_range, s_Qs_fit, label=f"Fit: s = {coeff_Qs[0]:.2f}(Q/s) + {coeff_Qs[1]:.2f}")
    ax3.set_title("Portata specifica Q/s vs s")
    ax3.set_xlabel("Q/s")
    ax3.set_ylabel("s (m)")
    ax3.grid(True)
    ax3.legend()

    results = {
        "B": B,
        "C": C,
        "delta_H": delta_H,
        "Qcrit": Qcrit,
        "s_crit": s_crit,
        "df": df,
        "fig1": fig1,
        "fig2": fig2,
        "fig3": fig3,
    }
    return results


# ---------------------
# Interfaccia Streamlit
# ---------------------

st.set_page_config(page_title="Analisi Prova di Portata", layout="centered")

st.title("Analisi Prova di Portata Pozzo")

st.markdown(
    "Inserisci i gradini della prova di portata.\n\n"
    "**Attenzione**: usa sempre la **stessa unità** per Q (ad es. tutti in l/s, oppure tutti in m³/h)."
)

# Numero di gradini
num_gradini = st.number_input("Numero gradini di portata", min_value=3, max_value=10, value=5, step=1)

st.subheader("Dati di ingresso")

Q_vals = []
s_vals = []

# Intestazioni tabellina
c0, c1, c2 = st.columns(3)
c0.write("Gradino")
c1.write("Q")
c2.write("s (m)")

for i in range(num_gradini):
    col0, col1, col2 = st.columns(3)
    col0.write(f"{i+1}")
    q_val = col1.number_input(f"Q_{i+1}", key=f"Q_{i+1}", value=0.0, step=0.1, format="%.4f")
    s_val = col2.number_input(f"s_{i+1}", key=f"s_{i+1}", value=0.0, step=0.1, format="%.4f")
    Q_vals.append(q_val)
    s_vals.append(s_val)

if st.button("Calcola"):
    try:
        results = run_analysis(Q_vals, s_vals)

        B = results["B"]
        C = results["C"]
        delta_H = results["delta_H"]
        Qcrit = results["Qcrit"]
        s_crit = results["s_crit"]
        df = results["df"]

        st.subheader("Risultati principali")

        st.write(f"**B** = {B:.6f} (unità: s/Q)")
        st.write(f"**C** = {C:.6f} (unità: s/Q²)")
        if np.isfinite(Qcrit):
            st.write(f"**Portata critica Qcrit** = {Qcrit:.6f}")
            st.write(f"**Abbassamento a Qcrit** = {s_crit:.4f} m")
        else:
            st.write("**Portata critica non definita** (C ≤ 0).")
        st.write(f"**ΔH (ultimo - primo gradino)** = {delta_H:.4f} m")

        st.subheader("Tabella dei dati e parametri")
        st.dataframe(df, use_container_width=True)

        st.subheader("Grafici")

        st.pyplot(results["fig1"])
        st.pyplot(results["fig2"])
        st.pyplot(results["fig3"])

        # Download Excel
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Risultati")
        buffer.seek(0)

        st.download_button(
            label="📥 Scarica risultati in Excel",
            data=buffer,
            file_name="prova_portata_pozzo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.error(f"Errore nel calcolo: {e}")
