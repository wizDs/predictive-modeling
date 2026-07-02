"""Dansk skatteberegner 2026 — leg med tallene."""

import pandas as pd
import streamlit as st

AM_BIDRAG_SATS = 0.08
BUNDSKAT_SATS = 0.1267
KOMMUNESKAT_SATS_DEFAULT = 0.253
MELLEMSKAT_SATS = 0.075
TOPSKAT_SATS = 0.075
TOPTOPSKAT_SATS = 0.05

PERSONFRADRAG = 49_700
MELLEMSKAT_GRAENSE = 641_200
TOPSKAT_GRAENSE = 777_900
TOPTOPSKAT_GRAENSE = 2_592_700

RENTEFRADRAG_TRIN = 50_000
RENTEFRADRAG_REDUKTION = 0.08

# https://skat.dk/borger/fradrag/arbejdsrelaterede-fradrag/beskaeftigelses-og-jobfradrag
MAX_BESKAEFTIGELSESFRADRAG = 63_300

DAGPENGE_MAX_MD_2026 = 22_000
DAGPENGE_FORHOEJET_MD_2026 = 26_500
FORHOEJET_PERIODE_MD = 3


def beregn_skat(
    bruttoløn: float,
    pension_pct: float,
    kommuneskat: float,
    fradrag: float = 0.0,
    rentefradrag: float = 0.0,
    am_pligtig: bool = True,
) -> dict:
    pension = bruttoløn * pension_pct
    am_bidrag = bruttoløn * AM_BIDRAG_SATS if am_pligtig else 0.0
    personlig_indkomst = bruttoløn - am_bidrag - pension
    skattepligtig = max(personlig_indkomst - PERSONFRADRAG - fradrag, 0)

    samlet_bundsats = BUNDSKAT_SATS + kommuneskat
    bundskat = samlet_bundsats * skattepligtig

    # Rentefradrag: skatteværdi op til 50k = fuld bundsats; over 50k = reduceret med 8 pct.point
    rente_lav = min(rentefradrag, RENTEFRADRAG_TRIN)
    rente_hoj = max(rentefradrag - RENTEFRADRAG_TRIN, 0)
    rente_skattevaerdi = (
        rente_lav * samlet_bundsats
        + rente_hoj * max(samlet_bundsats - RENTEFRADRAG_REDUKTION, 0)
    )

    mellemskat = MELLEMSKAT_SATS * max(personlig_indkomst - MELLEMSKAT_GRAENSE, 0)
    topskat = TOPSKAT_SATS * max(personlig_indkomst - TOPSKAT_GRAENSE, 0)
    toptopskat = TOPTOPSKAT_SATS * max(personlig_indkomst - TOPTOPSKAT_GRAENSE, 0)
    samlet_skat = max(bundskat + mellemskat + topskat + toptopskat - rente_skattevaerdi, 0)
    disponibel = bruttoløn - pension - am_bidrag - samlet_skat
    effektiv_sats = (am_bidrag + samlet_skat) / bruttoløn if bruttoløn > 0 else 0.0

    return {
        "bruttoløn": bruttoløn,
        "pension": pension,
        "am_bidrag": am_bidrag,
        "personlig_indkomst": personlig_indkomst,
        "skattepligtig": skattepligtig,
        "bundskat": bundskat,
        "mellemskat": mellemskat,
        "topskat": topskat,
        "toptopskat": toptopskat,
        "rente_skattevaerdi": rente_skattevaerdi,
        "samlet_skat": samlet_skat,
        "disponibel": disponibel,
        "effektiv_skattesats": effektiv_sats,
    }


st.title("🇩🇰 Skatteberegner 2026")
st.caption("Beregn disponibel indkomst ud fra bruttoløn — leg med tallene")

col1, col2, col3 = st.columns(3)
with col1:
    bruttoløn = st.number_input(
        "Bruttoløn (kr./år)",
        min_value=0,
        max_value=10_000_000,
        value=600_000,
        step=10_000,
    )
with col2:
    pension_pct = st.slider(
        "Pension (%)",
        min_value=0.0,
        max_value=25.0,
        value=12.0,
        step=0.5,
    ) / 100
with col3:
    kommuneskat = st.slider(
        "Kommuneskat (%)",
        min_value=22.0,
        max_value=30.0,
        value=KOMMUNESKAT_SATS_DEFAULT * 100,
        step=0.1,
    ) / 100

col4, col5 = st.columns(2)
with col4:
    fradrag = st.number_input(
        "Fradrag (kr./år)",
        min_value=0,
        max_value=1_000_000,
        value=MAX_BESKAEFTIGELSESFRADRAG,
        step=1_000,
        help="Ligningsmæssige fradrag — trækkes fra skattepligtig indkomst",
    )
with col5:
    rentefradrag = st.number_input(
        "Rentefradrag (kr./år)",
        min_value=0,
        max_value=1_000_000,
        value=0,
        step=1_000,
        help=(
            f"Renteudgifter. Skatteværdi: fuld bundsats op til {RENTEFRADRAG_TRIN:,} kr., "
            f"derefter reduceret med {RENTEFRADRAG_REDUKTION:.0%}-point."
        ),
    )

r = beregn_skat(bruttoløn, pension_pct, kommuneskat, fradrag, rentefradrag)

st.divider()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Disponibel (år)", f"{r['disponibel']:,.0f} kr.")
m2.metric("Disponibel (md)", f"{r['disponibel']/12:,.0f} kr.")
m3.metric("Samlet skat + AM", f"{r['samlet_skat'] + r['am_bidrag']:,.0f} kr.")
m4.metric("Effektiv skattesats", f"{r['effektiv_skattesats']:.1%}")

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Opgørelse")
    rows = [
        ("Bruttoløn", r["bruttoløn"]),
        ("− Pension", -r["pension"]),
        ("− AM-bidrag (8%)", -r["am_bidrag"]),
        ("Personlig indkomst", r["personlig_indkomst"]),
        ("Skattepligtig (efter personfradrag)", r["skattepligtig"]),
        ("− Bundskat + kommuneskat", -r["bundskat"]),
        ("− Mellemskat (7,5%)", -r["mellemskat"]),
        ("− Topskat (7,5%)", -r["topskat"]),
        ("− Top-topskat (5,0%)", -r["toptopskat"]),
        ("+ Rentefradrag (skatteværdi)", r["rente_skattevaerdi"]),
        ("Samlet skat", r["samlet_skat"]),
        ("Disponibel indkomst", r["disponibel"]),
    ]
    df = pd.DataFrame(rows, columns=["Post", "Beløb (kr.)"])
    st.dataframe(
        df.style.format({"Beløb (kr.)": "{:,.0f}"}),
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.subheader("Fordeling af bruttoløn")
    fordeling = pd.DataFrame(
        {
            "Kategori": ["Disponibel", "Pension", "AM-bidrag", "Skat"],
            "Beløb": [
                r["disponibel"],
                r["pension"],
                r["am_bidrag"],
                r["samlet_skat"],
            ],
        }
    )
    st.bar_chart(fordeling, x="Kategori", y="Beløb", horizontal=True)

st.divider()

st.subheader("Disponibel indkomst vs. bruttoløn")
kurve_rows = []
for løn in range(200_000, 3_100_000, 50_000):
    res = beregn_skat(løn, pension_pct, kommuneskat, fradrag, rentefradrag)
    kurve_rows.append(
        {
            "Bruttoløn": løn,
            "Disponibel": res["disponibel"],
            "Effektiv sats": res["effektiv_skattesats"],
        }
    )
kurve = pd.DataFrame(kurve_rows)

c1, c2 = st.columns(2)
with c1:
    st.line_chart(kurve, x="Bruttoløn", y="Disponibel")
with c2:
    st.line_chart(kurve, x="Bruttoløn", y="Effektiv sats")

with st.expander("💼 Hvad hvis jeg bliver arbejdsløs?"):
    st.caption(
        "Beregner i tre faser: fratrædelsesløn → forhøjet dagpenge → standard dagpenge. "
        "Dagpenge er skattepligtig A-indkomst men IKKE AM-bidragspligtig, og du indbetaler typisk ikke pension."
    )

    a, b = st.columns(2)
    with a:
        ledige_måneder = st.number_input(
            "Måneder uden job",
            min_value=1,
            max_value=24,
            value=6,
            step=1,
        )
    with b:
        fratraedelse_md = st.number_input(
            "Fratrædelse (mdr.)",
            min_value=0,
            max_value=12,
            value=3,
            step=1,
            help="Måneder med løn fra arbejdsgiver efter opsigelse",
        )

    d, e = st.columns(2)
    with d:
        dagpenge_md = st.number_input(
            "Dagpenge standard (kr./md)",
            min_value=0,
            max_value=40_000,
            value=DAGPENGE_MAX_MD_2026,
            step=500,
            help=f"Max dagpengesats 2026 ≈ {DAGPENGE_MAX_MD_2026:,} kr./md (fuld forsikring)",
        )
    with e:
        dagpenge_forhoejet_md = st.number_input(
            "Dagpenge forhøjet (kr./md)",
            min_value=0,
            max_value=40_000,
            value=DAGPENGE_FORHOEJET_MD_2026,
            step=500,
            help=f"Forhøjet sats første {FORHOEJET_PERIODE_MD} mdr. — kilde: aka.dk",
        )

    dp_std = beregn_skat(
        dagpenge_md * 12, 0.0, kommuneskat, fradrag, rentefradrag, am_pligtig=False
    )
    dp_for = beregn_skat(
        dagpenge_forhoejet_md * 12, 0.0, kommuneskat, fradrag, rentefradrag, am_pligtig=False
    )

    disp_job_md = r["disponibel"] / 12
    disp_std_md = dp_std["disponibel"] / 12
    disp_for_md = dp_for["disponibel"] / 12

    sev_used = min(fratraedelse_md, ledige_måneder)
    forhoejet_used = min(max(ledige_måneder - sev_used, 0), FORHOEJET_PERIODE_MD)
    standard_used = max(ledige_måneder - sev_used - forhoejet_used, 0)

    tab_for = forhoejet_used * (disp_job_md - disp_for_md)
    tab_std = standard_used * (disp_job_md - disp_std_md)
    buffer = tab_for + tab_std

    m1, m2, m3 = st.columns(3)
    m1.metric("Disp. forhøjet (md)", f"{disp_for_md:,.0f} kr.",
              delta=f"-{disp_job_md - disp_for_md:,.0f} kr.", delta_color="inverse")
    m2.metric("Disp. standard (md)", f"{disp_std_md:,.0f} kr.",
              delta=f"-{disp_job_md - disp_std_md:,.0f} kr.", delta_color="inverse")
    m3.metric(
        f"Buffer for {ledige_måneder} mdr.",
        f"{buffer:,.0f} kr.",
        help="Beløbet du skal have stående for at opretholde nuværende levestandard",
    )

    faser = pd.DataFrame(
        [
            {
                "Fase": f"Fratrædelse ({sev_used} mdr.)",
                "Disponibel/md": disp_job_md,
                "Tab/md": 0,
                "Tab i alt": 0,
            },
            {
                "Fase": f"Forhøjet dagpenge ({forhoejet_used} mdr.)",
                "Disponibel/md": disp_for_md,
                "Tab/md": disp_job_md - disp_for_md,
                "Tab i alt": tab_for,
            },
            {
                "Fase": f"Standard dagpenge ({standard_used} mdr.)",
                "Disponibel/md": disp_std_md,
                "Tab/md": disp_job_md - disp_std_md,
                "Tab i alt": tab_std,
            },
        ]
    )
    st.dataframe(
        faser.style.format(
            {"Disponibel/md": "{:,.0f}", "Tab/md": "{:,.0f}", "Tab i alt": "{:,.0f}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        f"Med **{sev_used} mdr.** fratrædelsesløn + **{forhoejet_used} mdr.** forhøjet dagpenge "
        f"+ **{standard_used} mdr.** standard dagpenge skal du have en buffer på **{buffer:,.0f} kr.** "
        f"for at opretholde nuværende levestandard."
    )

with st.expander("Skattesatser og grænser 2026"):
    st.markdown(
        f"""
        - **AM-bidrag:** {AM_BIDRAG_SATS:.0%}
        - **Bundskat:** {BUNDSKAT_SATS:.2%}
        - **Mellemskat:** {MELLEMSKAT_SATS:.1%} over {MELLEMSKAT_GRAENSE:,} kr.
        - **Topskat:** {TOPSKAT_SATS:.1%} over {TOPSKAT_GRAENSE:,} kr.
        - **Top-topskat:** {TOPTOPSKAT_SATS:.0%} over {TOPTOPSKAT_GRAENSE:,} kr.
        - **Personfradrag:** {PERSONFRADRAG:,} kr.
        """
    )
