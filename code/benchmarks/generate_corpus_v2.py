#!/usr/bin/env python3
"""Generate the v2 benchmark corpus: 200+ realistic scientific documents.

Run from the code/ directory:
    python3 benchmarks/generate_corpus_v2.py
"""

import os
import random
import textwrap
from pathlib import Path

random.seed(42)

CORPUS_DIR = Path(__file__).resolve().parent / "corpus_v2"


def write_doc(domain: str, filename: str, text: str) -> None:
    p = CORPUS_DIR / domain / filename
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(text).strip() + "\n")


# =========================================================================
# MATERIALS SCIENCE (40 docs) -- tensile strength, thickness, strain
# =========================================================================

_mat_templates = [
    # 0 - steel tensile test
    lambda i: (f"mat_{i:03d}.txt", f"""
    Tensile Test Report: Sample MS-{1000+i}

    A standard tensile test was performed on a low-carbon steel specimen (ASTM A36)
    with initial gauge length of {random.uniform(45,55):.1f} mm and cross-sectional
    area of {random.uniform(12,14):.1f} mm x {random.uniform(3,5):.2f} mm. The specimen
    was loaded at a constant crosshead speed of {random.uniform(1,5):.1f} mm/min,
    corresponding to a nominal strain rate of {random.uniform(0.0005,0.002):.4f} s^{{-1}}.

    Results: The yield strength was {random.uniform(240,280):.1f} MPa (0.2% offset method)
    and the ultimate tensile strength reached {random.uniform(420,520):.1f} MPa at
    an engineering strain of {random.uniform(18,28):.1f}%. The specimen fractured after
    {random.uniform(120,300):.0f} s of loading. Reduction in area was {random.uniform(55,70):.1f}%.

    The elastic modulus was calculated from the linear portion of the stress-strain
    curve as {random.uniform(195,210):.0f} GPa, consistent with published values for
    mild steel. The fracture surface exhibited a cup-and-cone morphology typical
    of ductile failure. All measurements comply with ASTM E8/E8M-21 standards.
    Sample thickness was {random.uniform(3.0,5.0):.2f} mm measured with a micrometer
    having resolution of 0.01 mm. The applied force was computed as F = m * a
    from Newton's second law, where the crosshead acceleration was used to
    verify the load cell calibration (F = ma).
    """),

    # 1 - aluminum alloy
    lambda i: (f"mat_{i:03d}.txt", f"""
    Material Characterization: Al-6061-T6 Sheet, Batch {2000+i}

    This report documents the mechanical properties of aluminum alloy 6061-T6 sheet
    stock received from supplier. Sheet thickness: {random.uniform(1.5,3.2):.2f} mm.
    Three dog-bone specimens were laser-cut per ASTM E8 with gauge width {random.uniform(6,8):.1f} mm
    and gauge length {random.uniform(25,30):.0f} mm.

    Average tensile properties (n=3):
    - Yield strength (Rp0.2): {random.uniform(270,310):.1f} MPa
    - Ultimate tensile strength: {random.uniform(310,360):.1f} MPa
    - Elongation at break: {random.uniform(12,17):.1f}%
    - Young's modulus: {random.uniform(68,72):.1f} GPa

    The stress at {random.uniform(270,310):.0f} MPa was recorded as the onset of plastic
    deformation. Hardness measurements using Vickers (HV10) gave {random.uniform(105,115):.0f} HV,
    consistent with the T6 temper condition. Tests were conducted at {random.uniform(22,25):.1f} degC
    ambient temperature. The strain was monitored using a clip-on extensometer with
    {random.uniform(25,50):.0f} mm gauge length, sampling at {random.uniform(50,200):.0f} Hz.

    Duration per test: approximately {random.uniform(3,8):.1f} min. The load cell capacity
    was 100 kN, and the machine compliance was corrected in post-processing.
    The force on the specimen follows F = m * a (Newton's second law) during
    dynamic loading phases; static equilibrium was confirmed (F = ma check).
    """),

    # 2 - polymer testing
    lambda i: (f"mat_{i:03d}.txt", f"""
    Polymer Tensile Properties: Nylon 6,6 Injection-Molded Specimen

    Test ID: PLY-{3000+i}
    Specimen dimensions: {random.uniform(3.0,4.5):.1f} mm thick, {random.uniform(10,13):.1f} mm wide,
    overall length {random.uniform(160,170):.0f} mm (ISO 527-2, Type 1A).

    The specimen was conditioned at 23 degC and 50% RH for 48 hours before testing.
    Testing was performed on a universal testing machine with a {random.uniform(5,10):.0f} kN
    load cell at a crosshead speed of {random.uniform(5,50):.0f} mm/min.

    Measured properties:
    - Tensile strength: {random.uniform(55,80):.1f} MPa
    - Yield stress: {random.uniform(45,65):.1f} MPa
    - Strain at break: {random.uniform(25,100):.0f}%
    - Elastic modulus: {random.uniform(2.5,3.5):.2f} GPa

    The stress-strain curve exhibited distinct yield point followed by cold drawing.
    Neck formation was observed at approximately {random.uniform(8,15):.0f}% strain.
    The equivalent stress at yield, {random.uniform(45000,65000):.0f} kPa, falls within
    manufacturer specifications. Five replicate tests were conducted; the coefficient
    of variation in UTS was {random.uniform(2,6):.1f}%.

    Duration of test: {random.uniform(30,180):.0f} s. Specimen mass: {random.uniform(8,15):.1f} g.
    The inertial correction for the crosshead was negligible; F = m * a gives
    a dynamic contribution less than 0.1% of the applied force (F = ma).
    """),

    # 3 - composite laminate
    lambda i: (f"mat_{i:03d}.txt", f"""
    Composite Laminate Tensile Test: CFRP Layup [{random.choice(['0/90/0','0/45/-45/90','0/0/90/90'])}]s

    Sample ID: CMP-{4000+i}
    Laminate thickness: {random.uniform(1.8,3.0):.2f} mm ({random.randint(8,16)} plies)
    Specimen width: {random.uniform(15,25):.1f} mm
    Tab material: Glass/epoxy, bonded with cyanoacrylate adhesive

    The test was conducted per ASTM D3039 at a rate of {random.uniform(1,2):.1f} mm/min.
    Strain was measured by biaxial strain gauges with {random.uniform(5,10):.0f} mm
    gauge length, logged at {random.uniform(1000,5000):.0f} Hz.

    Key results:
    - Longitudinal tensile strength: {random.uniform(600,1200):.0f} MPa
    - Longitudinal modulus: {random.uniform(60,140):.0f} GPa
    - Transverse tensile strength: {random.uniform(30,80):.0f} MPa
    - Poisson ratio: {random.uniform(0.28,0.35):.3f}
    - Failure strain: {random.uniform(0.8,2.0):.2f}%

    Failure mode: lateral/gauge/middle (LGM) per ASTM classification. The laminate
    exhibited {random.choice(['fiber breakage','delamination','matrix cracking'])} as
    the dominant damage mechanism. The applied stress of {random.uniform(600000,1200000):.0f} kPa
    at failure is consistent with rule-of-mixtures predictions. Test duration was
    approximately {random.uniform(100,400):.0f} s.
    """),

    # 4 - high-strength steel
    lambda i: (f"mat_{i:03d}.txt", f"""
    High-Strength Steel Evaluation: DP800 Dual-Phase

    Specimen ID: HSS-{5000+i}
    Sheet thickness: {random.uniform(1.0,2.0):.2f} mm
    Rolling direction: longitudinal

    This dual-phase steel was tested at multiple strain rates to characterize its
    rate sensitivity. A servo-hydraulic machine was used for dynamic tests, while
    quasi-static tests used a screw-driven frame.

    Quasi-static results (strain rate {random.uniform(0.001,0.01):.4f} s^{{-1}}):
    - Yield strength: {random.uniform(450,550):.0f} MPa
    - UTS: {random.uniform(780,850):.0f} MPa
    - Uniform elongation: {random.uniform(12,18):.1f}%

    Dynamic results (strain rate {random.uniform(100,500):.0f} s^{{-1}}):
    - Yield strength: {random.uniform(550,700):.0f} MPa
    - UTS: {random.uniform(850,980):.0f} MPa

    The stress increase from quasi-static to dynamic loading confirms positive strain
    rate sensitivity typical of ferritic-martensitic microstructures. Sample preparation
    involved EDM cutting with {random.uniform(0.1,0.3):.2f} mm wire, followed by
    surface grinding to {random.uniform(0.8,1.6):.1f} um Ra finish. All pressures in
    the hydraulic system were maintained at {random.uniform(18,25):.1f} MPa during
    gripping. The test fixture was calibrated using a {random.uniform(50,100):.0f} kN
    reference load cell. Total test time: {random.uniform(0.5,5):.1f} min.
    """),
]

# 5 - thin film
_mat_templates.append(
    lambda i: (f"mat_{i:03d}.txt", f"""
    Thin Film Mechanical Properties: SiN_x on Silicon

    Sample: TF-{6000+i}
    Film thickness: {random.uniform(100,500):.0f} nm ({random.uniform(0.1,0.5):.3f} mm equivalent is NOT correct; actual is sub-micron)
    Substrate: Si(100), {random.uniform(0.3,0.7):.1f} mm thick

    Nanoindentation was performed using a Berkovich tip with maximum load
    {random.uniform(1,10):.1f} mN. The Oliver-Pharr method yielded:
    - Hardness: {random.uniform(15,25):.1f} GPa
    - Reduced modulus: {random.uniform(150,250):.0f} GPa

    The residual stress in the film was measured by wafer curvature (Stoney equation)
    at {random.uniform(100,600):.0f} MPa compressive. The biaxial modulus was
    {random.uniform(200,300):.0f} GPa. Deposition was carried out via PECVD at a
    chamber pressure of {random.uniform(50,200):.0f} Pa with substrate temperature
    {random.uniform(250,350):.0f} degC.

    Process gas: SiH4/NH3/N2 mixture at total flow {random.uniform(100,300):.0f} sccm.
    RF power: {random.uniform(50,200):.0f} W at 13.56 MHz. Deposition duration was
    {random.uniform(20,60):.0f} min, yielding a deposition rate of approximately
    {random.uniform(5,15):.1f} nm/min. Film adhesion was verified by scratch testing;
    critical load was {random.uniform(10,50):.0f} mN.
    """),
)

# 6 - fatigue test
_mat_templates.append(
    lambda i: (f"mat_{i:03d}.txt", f"""
    Fatigue Test Summary: Ti-6Al-4V ELI, Run FT-{7000+i}

    Specimen geometry: cylindrical, gauge diameter {random.uniform(5,8):.1f} mm,
    gauge length {random.uniform(15,25):.0f} mm. Surface finish: {random.uniform(0.2,0.8):.1f} um Ra.

    Testing parameters:
    - Stress amplitude: {random.uniform(350,550):.0f} MPa
    - Stress ratio R = {random.choice(['-1', '0', '0.1'])}
    - Frequency: {random.uniform(10,30):.0f} Hz
    - Environment: lab air, {random.uniform(20,25):.0f} degC

    The specimen endured {random.randint(50000,5000000)} cycles before failure.
    The corresponding stress amplitude of {random.uniform(350000,550000):.0f} kPa places this
    result {random.choice(['above', 'below', 'near'])} the published S-N curve for
    mill-annealed Ti-6Al-4V.

    Fractographic analysis revealed {random.choice(['surface-initiated', 'subsurface-initiated'])}
    cracking. The fracture surface showed fatigue striations with spacing of
    approximately {random.uniform(0.5,5):.1f} um in the stable growth region.
    Crack propagation followed a Paris-law relationship da/dN = C * (DeltaK)^m with
    m = {random.uniform(2.5,4.5):.2f}. Maximum stress intensity factor at failure was
    approximately {random.uniform(50,100):.0f} MPa*sqrt(m). Test duration:
    {random.uniform(0.5,100):.1f} h.
    """),
)

# 7 - creep test
_mat_templates.append(
    lambda i: (f"mat_{i:03d}.txt", f"""
    Creep Test Report: Inconel 718, Sample CR-{8000+i}

    Test conditions:
    - Temperature: {random.uniform(600,750):.0f} degC
    - Applied stress: {random.uniform(500,700):.0f} MPa
    - Environment: Argon atmosphere at {random.uniform(80,120):.0f} kPa

    Specimen: cylindrical, diameter {random.uniform(6,10):.1f} mm, gauge length
    {random.uniform(25,50):.0f} mm. The specimen was solution treated at 980 degC
    for 1 h followed by double aging ({random.choice(['720','620'])} degC).

    Results:
    - Primary creep duration: {random.uniform(10,50):.0f} h
    - Minimum creep rate: {random.uniform(1e-9, 1e-7):.2e} s^{{-1}}
    - Rupture time: {random.uniform(100,5000):.0f} h
    - Elongation at rupture: {random.uniform(5,25):.1f}%

    The steady-state creep rate of {random.uniform(1e-9, 1e-7):.2e} s^{{-1}} is consistent
    with diffusion-controlled dislocation climb mechanisms. Activation energy was
    estimated at {random.uniform(250,350):.0f} kJ/mol via Arrhenius analysis across
    three temperatures. Grain boundary cavitation was observed in post-test metallography.
    The applied stress of {random.uniform(500000,700000):.0f} kPa exceeds the threshold
    for power-law breakdown. Specimen mass before test: {random.uniform(40,80):.1f} g;
    after test: {random.uniform(39,79):.1f} g (minor oxidation loss).
    """),
)


def gen_materials_science():
    for i in range(40):
        template = _mat_templates[i % len(_mat_templates)]
        fname, text = template(i)
        write_doc("materials_science", fname, text)


# =========================================================================
# ATMOSPHERIC SCIENCE (40 docs) -- pressure in Pa/kPa/hPa, wind, temp
# =========================================================================

_atmo_templates = [
    # 0 - weather station
    lambda i: (f"atmo_{i:03d}.txt", f"""
    Weather Station Report: Station WX-{100+i}, {random.choice(['Great Plains','Appalachian Ridge','Coastal California','Lake Michigan Shore','Intermountain West'])}

    Observation period: {random.randint(1,28):02d}/{random.randint(1,12):02d}/2025, 00:00-23:59 UTC

    Surface observations (hourly averages):
    - Barometric pressure: {random.uniform(990,1030):.1f} hPa (station level)
    - Sea-level pressure: {random.uniform(995,1035):.1f} hPa
    - Temperature range: {random.uniform(-5,15):.1f} to {random.uniform(15,35):.1f} degC
    - Relative humidity: {random.uniform(30,95):.0f}%
    - Wind speed (10 m): {random.uniform(1,15):.1f} m/s, gusts to {random.uniform(10,30):.1f} m/s
    - Wind direction: {random.randint(0,360)} deg (meteorological convention)
    - Precipitation: {random.uniform(0,25):.1f} mm

    The barometric pressure in SI units was {random.uniform(99000,103000):.0f} Pa. A cold front
    passage at approximately {random.randint(8,18):02d}:00 UTC produced a pressure drop
    of {random.uniform(2,8):.1f} hPa over {random.uniform(2,6):.1f} hours. Peak wind
    during frontal passage was {random.uniform(40,80):.1f} km/h from the
    {random.choice(['northwest','north','west'])}. The 24-hour pressure tendency was
    {random.choice(['+','-'])}{random.uniform(1,5):.1f} hPa.

    Data quality: all sensors passed daily calibration checks. Anemometer height: 10 m AGL.
    Barometer elevation: {random.uniform(100,2000):.0f} m MSL.
    """),

    # 1 - radiosonde
    lambda i: (f"atmo_{i:03d}.txt", f"""
    Radiosonde Profile: Launch {200+i}, {random.choice(['Norman OK','Denver CO','Miami FL','Fairbanks AK'])}

    Launch time: {random.randint(0,23):02d}:00 UTC
    Surface conditions: pressure {random.uniform(850,1020):.1f} hPa, temperature {random.uniform(-10,35):.1f} degC,
    dewpoint {random.uniform(-20,25):.1f} degC

    Upper-air data (selected levels):
    - 850 hPa: height {random.uniform(1400,1600):.0f} m, temp {random.uniform(-5,20):.1f} degC, wind {random.uniform(5,25):.0f} m/s from {random.randint(180,360)}
    - 700 hPa: height {random.uniform(2900,3200):.0f} m, temp {random.uniform(-10,10):.1f} degC
    - 500 hPa: height {random.uniform(5400,5900):.0f} m, temp {random.uniform(-25,-10):.1f} degC, wind {random.uniform(15,50):.0f} m/s
    - 300 hPa: height {random.uniform(8900,9600):.0f} m, temp {random.uniform(-50,-35):.1f} degC, wind {random.uniform(20,80):.0f} m/s
    - 200 hPa: height {random.uniform(11500,12500):.0f} m, wind {random.uniform(25,100):.0f} m/s (jet stream level)

    The tropopause was identified at {random.uniform(150,250):.0f} hPa ({random.uniform(10500,14000):.0f} m).
    CAPE was {random.uniform(0,4000):.0f} J/kg, CIN was {random.uniform(0,200):.0f} J/kg. The
    precipitable water was {random.uniform(10,60):.0f} mm. Lifted index: {random.uniform(-8,5):.1f}.
    The surface pressure converted to Pa was {random.uniform(85000,102000):.0f} Pa.

    The 500 hPa wind speed in surface-relative units corresponds to approximately
    {random.uniform(54,180):.0f} km/h. The sounding reached a maximum altitude of
    {random.uniform(25000,35000):.0f} m before balloon burst.
    """),

    # 2 - severe weather
    lambda i: (f"atmo_{i:03d}.txt", f"""
    Severe Weather Event Summary: Case SW-{300+i}

    Date: {random.randint(1,28):02d}/{random.choice(['04','05','06','07'])}/2025
    Location: {random.choice(['Central Oklahoma','Eastern Kansas','Western Texas','Northern Illinois'])}
    Event type: {random.choice(['supercell','squall line','derecho','tornado-warned supercell'])}

    Pre-storm environment:
    - Surface pressure: {random.uniform(98,102):.1f} kPa
    - Surface temperature: {random.uniform(25,38):.0f} degC
    - Surface dewpoint: {random.uniform(18,28):.0f} degC
    - 0-6 km shear: {random.uniform(15,40):.0f} m/s ({random.uniform(54,144):.0f} km/h)
    - CAPE: {random.uniform(2000,5000):.0f} J/kg
    - Storm-relative helicity (0-3 km): {random.uniform(100,500):.0f} m^2/s^2

    Peak observations during event:
    - Minimum pressure: {random.uniform(94,99):.1f} kPa ({random.uniform(94000,99000):.0f} Pa)
    - Maximum wind gust: {random.uniform(25,50):.0f} m/s
    - Maximum hail diameter: {random.uniform(20,75):.0f} mm
    - Peak radar reflectivity: {random.uniform(55,75):.0f} dBZ

    The pressure perturbation of {random.uniform(2,6):.1f} kPa was detected by the
    Oklahoma Mesonet stations. {random.choice(['A tornado was confirmed','No tornado was observed','Brief tornado (EF-1) was reported'])}
    with estimated peak winds of {random.uniform(80,220):.0f} km/h. Storm motion was
    {random.uniform(30,60):.0f} km/h from {random.randint(200,280)} degrees.
    Event duration: approximately {random.uniform(1,4):.1f} h.
    """),

    # 3 - climate data
    lambda i: (f"atmo_{i:03d}.txt", f"""
    Climate Data Analysis: Monthly Summary for Station {400+i}

    Month: {random.choice(['January','February','March','April','May','June','July','August','September','October','November','December'])} 2025
    Latitude: {random.uniform(25,65):.2f}N, Longitude: {random.uniform(-125,-65):.2f}W
    Elevation: {random.uniform(10,2500):.0f} m MSL

    Monthly statistics:
    - Mean sea-level pressure: {random.uniform(1005,1025):.1f} hPa
    - Pressure range: {random.uniform(985,1000):.1f} to {random.uniform(1030,1050):.1f} hPa
    - Mean temperature: {random.uniform(-15,30):.1f} degC
    - Total precipitation: {random.uniform(5,200):.1f} mm
    - Mean wind speed: {random.uniform(2,10):.1f} m/s ({random.uniform(7.2,36):.1f} km/h)
    - Peak wind gust: {random.uniform(15,40):.0f} m/s

    Anomalies relative to 1991-2020 baseline:
    - Pressure anomaly: {random.choice(['+','-'])}{random.uniform(0.5,5):.1f} hPa
    - Temperature anomaly: {random.choice(['+','-'])}{random.uniform(0.2,3):.1f} degC

    The mean station pressure of {random.uniform(700,1020):.0f} hPa converts to
    {random.uniform(70000,102000):.0f} Pa. Cloud cover averaged {random.uniform(30,80):.0f}%
    with {random.randint(5,25)} clear days. Solar radiation: {random.uniform(50,300):.0f} W/m^2
    daily average. Soil temperature (10 cm depth): {random.uniform(2,25):.1f} degC.

    This station has been operational since {random.randint(1940,2000)} with
    {random.uniform(95,99.9):.1f}% data completeness.
    """),

    # 4 - boundary layer
    lambda i: (f"atmo_{i:03d}.txt", f"""
    Atmospheric Boundary Layer Study: Campaign ABL-{500+i}

    Site: Flat terrain, fetch > 5 km, crop stubble surface
    Measurement period: {random.uniform(3,14):.0f} days in {random.choice(['summer','autumn','winter','spring'])}

    Tower instrumentation (10 levels from 2 m to 60 m):
    - Sonic anemometers: 3-D wind at {random.choice(['10','20'])} Hz
    - Temperature/humidity: aspirated thermohygrometers
    - Pressure: Vaisala PTB330 at 2 m, accuracy +/- 0.1 hPa

    Selected results (convective afternoon case):
    - Mixed layer depth: {random.uniform(1000,2500):.0f} m
    - Surface heat flux: {random.uniform(100,400):.0f} W/m^2
    - Friction velocity u*: {random.uniform(0.2,0.6):.2f} m/s
    - 10-m wind speed: {random.uniform(3,8):.1f} m/s
    - Surface pressure: {random.uniform(96,102):.1f} kPa
    - Roughness length z0: {random.uniform(0.01,0.1):.3f} m
    - Monin-Obukhov length: {random.uniform(-50,-500):.0f} m (unstable)

    Turbulence statistics: the velocity variance sigma_u was {random.uniform(0.5,2.0):.2f} m/s
    and the turbulent kinetic energy TKE = 0.5*(sigma_u^2 + sigma_v^2 + sigma_w^2)
    was approximately {random.uniform(0.5,3):.2f} m^2/s^2. The surface pressure
    in Pa was {random.uniform(96000,102000):.0f} Pa with a diurnal range of
    {random.uniform(100,500):.0f} Pa.

    The thermal wind estimated from the pressure gradient was {random.uniform(1,5):.1f} m/s
    at 1 km height. Entrainment zone thickness: {random.uniform(100,500):.0f} m.
    """),
]

def gen_atmospheric_science():
    for i in range(40):
        template = _atmo_templates[i % len(_atmo_templates)]
        fname, text = template(i)
        write_doc("atmospheric_science", fname, text)


# =========================================================================
# FLUID DYNAMICS (40 docs) -- velocity, pressure, pipe diameter, Re
# =========================================================================

_fluid_templates = [
    # 0 - pipe flow
    lambda i: (f"fluid_{i:03d}.txt", f"""
    Pipe Flow Experiment: Run PF-{100+i}

    A fully developed turbulent pipe flow experiment was conducted in a {random.uniform(25,100):.0f} mm
    inner diameter smooth pipe. The pipe length was {random.uniform(3,10):.1f} m, giving an L/D ratio
    of {random.uniform(30,200):.0f}, sufficient for fully developed conditions. Working fluid: water
    at {random.uniform(18,25):.1f} degC (kinematic viscosity approximately 1.0e-6 m^2/s).

    Flow conditions:
    - Bulk velocity: {random.uniform(0.5,5):.2f} m/s
    - Reynolds number: Re = {random.randint(5000,200000)}
    - Volume flow rate: {random.uniform(0.1,20):.2f} L/s
    - Inlet pressure: {random.uniform(150,500):.0f} kPa (gauge)
    - Outlet pressure: {random.uniform(100,200):.0f} kPa (gauge)
    - Pressure drop: {random.uniform(10,300):.1f} kPa over the test section

    The pressure drop of {random.uniform(10000,300000):.0f} Pa corresponds to a Darcy friction
    factor of f = {random.uniform(0.015,0.045):.4f}, in good agreement with the Moody diagram
    prediction. Wall shear stress was estimated at {random.uniform(5,100):.1f} Pa using the
    relation tau_w = (delta_P * D) / (4 * L).

    PIV measurements in a cross-sectional plane at x/D = {random.uniform(50,150):.0f} revealed
    the expected logarithmic velocity profile. The centerline velocity was
    {random.uniform(0.6,6):.2f} m/s ({random.uniform(2.16,21.6):.1f} km/h).
    Turbulence intensity at the centerline: {random.uniform(3,8):.1f}%.
    """),

    # 1 - nozzle flow
    lambda i: (f"fluid_{i:03d}.txt", f"""
    Turbulent Flow Simulation: Run T-{2000+i}

    This simulation models turbulent flow through a converging nozzle at Reynolds
    number Re = {random.randint(10000,100000)}. The inlet velocity was set to
    {random.uniform(5,30):.1f} m/s with a turbulence intensity of {random.uniform(2,10):.0f}%.
    The nozzle throat diameter was {random.uniform(15,40):.0f} mm, expanding to
    {random.uniform(40,80):.0f} mm at the outlet.

    Boundary conditions: inlet pressure {random.uniform(200,500):.0f} kPa (gauge),
    outlet pressure {random.uniform(100,105):.1f} kPa (atmospheric). The simulation ran
    for {random.uniform(1,5):.1f} seconds of physical time using a time step of
    {random.uniform(0.0005,0.005):.4f} s. Peak velocity at the throat reached
    {random.uniform(50,150):.1f} m/s, consistent with the isentropic relation
    v = sqrt(2 * delta_P / rho).

    The pressure drop across the nozzle was {random.uniform(100,400):.1f} kPa,
    corresponding to {random.uniform(100000,400000):.0f} Pa. Grid convergence was verified
    using three mesh levels ({random.uniform(0.3,0.8):.1f} mm, {random.uniform(0.8,1.5):.1f} mm,
    {random.uniform(1.5,3.0):.1f} mm spacing). The k-epsilon turbulence model was employed with
    standard wall functions. Total cell count: {random.uniform(0.5,5):.1f} million.

    The throat velocity of {random.uniform(180,540):.0f} km/h indicates compressibility effects
    may become relevant. Mach number at throat: {random.uniform(0.15,0.5):.2f}.
    """),

    # 2 - open channel
    lambda i: (f"fluid_{i:03d}.txt", f"""
    Open Channel Flow Measurement: Site OC-{3000+i}

    A rectangular open channel with width {random.uniform(0.3,2.0):.2f} m and bed slope
    {random.uniform(0.001,0.01):.4f} was instrumented for velocity profiling. The channel
    lining is smooth concrete (Manning's n = {random.uniform(0.011,0.015):.4f}).

    Measured conditions at the test cross-section:
    - Water depth: {random.uniform(100,500):.0f} mm ({random.uniform(0.1,0.5):.3f} m)
    - Mean velocity: {random.uniform(0.3,2.0):.2f} m/s
    - Flow rate: {random.uniform(10,500):.0f} L/s
    - Froude number: Fr = {random.uniform(0.3,1.5):.2f}
    - Reynolds number: Re = {random.randint(10000,500000)}

    Velocity was measured using an acoustic Doppler velocimeter (ADV) at
    {random.choice(['25','50','100'])} Hz. The vertical velocity profile was sampled at
    {random.randint(10,30)} points from {random.uniform(5,15):.0f} mm above the bed to the
    free surface. Maximum velocity of {random.uniform(0.4,2.5):.2f} m/s occurred at
    approximately {random.uniform(0.5,0.8):.1f} of the flow depth.

    The hydraulic gradient corresponds to a pressure difference of {random.uniform(50,500):.0f} Pa
    per meter of channel length. The boundary shear stress was {random.uniform(1,20):.1f} Pa.
    A hydraulic jump was observed at {random.uniform(3,10):.1f} m downstream with a conjugate
    depth ratio of {random.uniform(1.5,4):.1f}. Upstream Froude number: {random.uniform(1.1,3.0):.1f}.
    Downstream velocity: {random.uniform(0.2,1.0):.2f} m/s ({random.uniform(0.72,3.6):.1f} km/h).
    """),

    # 3 - CFD validation
    lambda i: (f"fluid_{i:03d}.txt", f"""
    CFD Validation Study: Backward-Facing Step, Case BFS-{4000+i}

    A RANS simulation of flow over a backward-facing step was performed to validate
    the {random.choice(['k-omega SST','k-epsilon RNG','Spalart-Allmaras','RSM'])} turbulence model.
    The step height was h = {random.uniform(10,50):.0f} mm and the expansion ratio was
    {random.uniform(1.1,2.0):.2f}.

    Inlet conditions:
    - Freestream velocity: {random.uniform(5,30):.1f} m/s
    - Turbulence intensity: {random.uniform(0.5,5):.1f}%
    - Reynolds number (based on step height): Re_h = {random.randint(5000,80000)}

    Computational details:
    - Domain: {random.uniform(20,40):.0f}h upstream, {random.uniform(40,80):.0f}h downstream
    - Mesh: structured, {random.uniform(0.5,3):.1f} million cells
    - Near-wall spacing: y+ < {random.uniform(0.5,5):.1f}
    - Time step: {random.uniform(0.0001,0.005):.4f} s (for URANS)

    Results:
    - Reattachment length: {random.uniform(5,8):.1f}h (experiment: {random.uniform(6,7):.1f}h)
    - Peak reversed flow velocity: {random.uniform(-0.3,-0.1):.2f} times U_inlet
    - Pressure coefficient at reattachment: Cp = {random.uniform(0.1,0.3):.3f}
    - Wall pressure recovery: {random.uniform(60,90):.0f}% of theoretical

    The static pressure upstream of the step was {random.uniform(100,200):.1f} kPa while
    the base pressure (immediately behind step) dropped to {random.uniform(95,100):.1f} kPa.
    This pressure difference of {random.uniform(1000,10000):.0f} Pa drives the recirculation.
    Simulation wall-clock time: {random.uniform(2,48):.0f} h on {random.randint(16,256)} cores.
    """),

    # 4 - wind tunnel
    lambda i: (f"fluid_{i:03d}.txt", f"""
    Wind Tunnel Test: Airfoil NACA {random.choice(['0012','2412','4415','6412'])}, Run WT-{5000+i}

    Test section: {random.uniform(0.5,2):.1f} m x {random.uniform(0.5,1.5):.1f} m,
    turbulence intensity < {random.uniform(0.1,0.5):.2f}%.
    Model chord: {random.uniform(150,300):.0f} mm, span: {random.uniform(300,600):.0f} mm.
    End plates installed to enforce 2-D flow.

    Test conditions:
    - Freestream velocity: {random.uniform(15,60):.1f} m/s ({random.uniform(54,216):.0f} km/h)
    - Dynamic pressure: {random.uniform(100,2000):.0f} Pa (q = 0.5 * rho * V^2)
    - Reynolds number: {random.uniform(0.2,2.0):.1f} x 10^6
    - Angle of attack sweep: -4 to +20 deg in 2-deg increments
    - Stagnation pressure: {random.uniform(101,105):.1f} kPa

    Key results at alpha = {random.uniform(6,12):.0f} deg:
    - Lift coefficient: Cl = {random.uniform(0.5,1.4):.3f}
    - Drag coefficient: Cd = {random.uniform(0.006,0.030):.4f}
    - Pitching moment: Cm = {random.uniform(-0.15,-0.02):.4f}
    - Stall angle: approximately {random.uniform(12,18):.0f} deg

    Surface pressure distribution was measured using {random.randint(32,128)} pressure taps
    connected to a Scanivalve system sampling at {random.randint(100,1000)} Hz. The minimum
    surface pressure (suction peak) was {random.uniform(-3000,-500):.0f} Pa gauge, occurring
    at x/c = {random.uniform(0.02,0.15):.3f}. Wake survey at x/c = 2 confirmed the drag
    measurement within {random.uniform(2,8):.0f}% uncertainty.
    """),
]

def gen_fluid_dynamics():
    for i in range(40):
        template = _fluid_templates[i % len(_fluid_templates)]
        fname, text = template(i)
        write_doc("fluid_dynamics", fname, text)


# =========================================================================
# CHEMISTRY / REACTIONS (30 docs) -- mass, time, formulas
# =========================================================================

_chem_templates = [
    # 0 - synthesis
    lambda i: (f"chem_{i:03d}.txt", f"""
    Synthesis Report: {random.choice(['Gold Nanoparticle','Silver Nanorod','ZnO Quantum Dot','TiO2 Nanotube'])} Preparation

    Batch ID: SYN-{100+i}
    Method: {random.choice(['Turkevich citrate reduction','sol-gel','hydrothermal','co-precipitation'])}

    Reagent preparation:
    - Precursor: {random.uniform(50,500):.1f} mg dissolved in {random.uniform(10,100):.0f} mL deionized water
    - Reducing agent: {random.uniform(10,200):.1f} mg in {random.uniform(5,50):.0f} mL solvent
    - Stabilizer: {random.uniform(5,50):.1f} mg ({random.uniform(0.005,0.05):.3f} g)

    The reaction was carried out at {random.uniform(60,100):.0f} degC for {random.uniform(30,180):.0f} min
    ({random.uniform(1800,10800):.0f} s) under continuous magnetic stirring at {random.uniform(200,600):.0f} rpm.
    The pH was adjusted to {random.uniform(4,11):.1f} using NaOH/HCl.

    After synthesis, the product was centrifuged at {random.uniform(5000,15000):.0f} rpm for
    {random.uniform(10,30):.0f} min and washed three times with ethanol. Final product mass:
    {random.uniform(20,200):.1f} mg (yield: {random.uniform(40,90):.0f}%).

    Characterization:
    - UV-Vis absorption peak: {random.uniform(400,800):.0f} nm
    - DLS mean diameter: {random.uniform(5,100):.0f} nm
    - Zeta potential: {random.uniform(-50,-10):.0f} mV
    - XRD confirmed {random.choice(['fcc','wurtzite','anatase','rutile'])} crystal structure

    The total reagent mass was {random.uniform(0.08,0.75):.3f} g. Using the Arrhenius equation
    k = A * e^{{-Ea/RT}}, the estimated rate constant at the synthesis temperature is consistent
    with literature values for Ea = {random.uniform(40,120):.0f} kJ/mol.
    """),

    # 1 - kinetics
    lambda i: (f"chem_{i:03d}.txt", f"""
    Reaction Kinetics Study: {random.choice(['Ester Hydrolysis','Saponification','Diels-Alder','Suzuki Coupling'])}

    Experiment ID: KIN-{200+i}

    Reaction conditions:
    - Temperature: {random.uniform(25,80):.1f} degC
    - Solvent: {random.choice(['water','ethanol','THF','DMF','toluene'])}
    - Initial concentration: {random.uniform(0.01,1.0):.3f} M
    - Catalyst loading: {random.uniform(1,10):.1f} mol%

    Aliquots ({random.uniform(0.5,2):.1f} mL) were withdrawn at t = 0, {random.uniform(30,120):.0f},
    {random.uniform(120,300):.0f}, {random.uniform(300,600):.0f}, {random.uniform(600,1800):.0f},
    and {random.uniform(1800,3600):.0f} s. Conversion was monitored by {random.choice(['GC-MS','HPLC','NMR','UV-Vis'])}.

    Kinetic analysis:
    - Reaction order: {random.choice(['first','second','pseudo-first'])}
    - Rate constant k = {random.uniform(1e-4,1e-2):.4e} s^{{-1}} (at {random.uniform(25,80):.1f} degC)
    - Half-life: {random.uniform(60,3600):.0f} s ({random.uniform(1,60):.1f} min)
    - Activation energy Ea = {random.uniform(40,120):.0f} kJ/mol (from Arrhenius plot)

    The Arrhenius equation k = A * e^{{-Ea/RT}} with pre-exponential factor
    A = {random.uniform(1e6,1e12):.2e} s^{{-1}} fits the data with R^2 = {random.uniform(0.95,0.999):.3f}.
    The enthalpy of activation Delta_H_ddagger was {random.uniform(35,115):.0f} kJ/mol via
    Eyring analysis. Total reaction time to 95% conversion: {random.uniform(20,120):.0f} min.
    Product isolated mass: {random.uniform(0.5,5):.2f} g ({random.uniform(500,5000):.0f} mg).
    """),

    # 2 - electrochemistry
    lambda i: (f"chem_{i:03d}.txt", f"""
    Electrochemical Characterization: {random.choice(['LiFePO4','LiCoO2','Graphite','Si/C Composite'])} Electrode

    Cell ID: EC-{300+i}
    Configuration: {random.choice(['coin cell (CR2032)','pouch cell','three-electrode'])}

    Electrode preparation:
    - Active material: {random.uniform(1,10):.2f} mg
    - Conductive additive (Super P): {random.uniform(0.1,1):.2f} mg
    - Binder (PVDF): {random.uniform(0.1,1):.2f} mg
    - Total electrode mass: {random.uniform(1.5,12):.2f} mg
    - Active material loading: {random.uniform(1,5):.1f} mg/cm^2
    - Electrode thickness: {random.uniform(30,100):.0f} um ({random.uniform(0.03,0.1):.3f} mm)

    Cycling conditions:
    - Voltage window: {random.uniform(2.5,3.0):.1f} to {random.uniform(4.0,4.5):.1f} V
    - C-rate: C/{random.choice(['10','5','2','1'])}
    - Temperature: {random.uniform(20,30):.0f} degC

    Results (after {random.randint(50,500)} cycles):
    - Initial capacity: {random.uniform(120,180):.0f} mAh/g
    - Capacity retention: {random.uniform(70,99):.1f}%
    - Coulombic efficiency: {random.uniform(98,99.9):.1f}%
    - Impedance (Nyquist): R_ct = {random.uniform(10,200):.0f} Ohm

    The energy density E = q * V relationship gives {random.uniform(300,700):.0f} Wh/kg.
    Cyclic voltammetry at {random.uniform(0.1,1):.1f} mV/s showed redox peaks at
    {random.uniform(3.2,3.6):.2f}/{random.uniform(3.0,3.4):.2f} V. EIS was performed from
    100 kHz to 10 mHz. Total test duration: {random.uniform(100,2000):.0f} h.
    """),

    # 3 - thermochemistry
    lambda i: (f"chem_{i:03d}.txt", f"""
    Calorimetry Report: {random.choice(['Combustion','Dissolution','Neutralization','Mixing'])} Enthalpy

    Experiment: CAL-{400+i}
    Calorimeter type: {random.choice(['bomb (adiabatic)','solution (coffee-cup)','differential scanning (DSC)','isothermal titration (ITC)'])}

    Sample: {random.uniform(100,1000):.1f} mg ({random.uniform(0.1,1.0):.3f} g) of
    {random.choice(['benzoic acid','sucrose','naphthalene','urea','KNO3'])}
    in {random.uniform(50,200):.0f} mL of {random.choice(['water','HCl solution','NaOH solution'])}

    Measurement:
    - Initial temperature: {random.uniform(22,26):.3f} degC
    - Final temperature: {random.uniform(24,35):.3f} degC
    - Temperature rise: {random.uniform(1,10):.3f} degC
    - Calorimeter constant: {random.uniform(400,600):.1f} J/degC
    - Measured heat: {random.uniform(1,10):.2f} kJ
    - Molar enthalpy: {random.uniform(-500,-50):.1f} kJ/mol

    Using the relation q = m * c * Delta_T and correcting for heat losses, the
    enthalpy of {random.choice(['combustion','dissolution','neutralization'])} was determined.
    The ideal gas law PV = nRT was used to correct for gas-phase products where
    applicable. At the reaction pressure of {random.uniform(99,103):.1f} kPa, the molar
    volume correction is negligible.

    Uncertainty analysis: the combined standard uncertainty is {random.uniform(0.5,3):.1f}%
    (k=2), dominated by the temperature measurement uncertainty of {random.uniform(0.005,0.02):.3f} degC.
    Replicate measurements (n={random.randint(3,6)}): mean = {random.uniform(-500,-50):.1f} kJ/mol,
    s.d. = {random.uniform(1,10):.1f} kJ/mol. Duration of each run: {random.uniform(5,30):.0f} min.
    """),

    # 4 - spectroscopy
    lambda i: (f"chem_{i:03d}.txt", f"""
    Mass Spectrometry Analysis: Protein Digest, Run MS-{500+i}

    Sample preparation:
    - Protein extract: {random.uniform(10,100):.0f} ug ({random.uniform(0.01,0.1):.3f} mg)
    - Trypsin digestion: {random.uniform(12,18):.0f} h at 37 degC
    - Peptide cleanup: C18 SPE, eluted with 80% ACN
    - Final sample: {random.uniform(5,50):.0f} uL at {random.uniform(0.1,2):.1f} ug/uL

    LC-MS/MS conditions:
    - Column: C18, {random.uniform(75,150):.0f} um ID x {random.uniform(150,500):.0f} mm
    - Flow rate: {random.uniform(200,400):.0f} nL/min
    - Gradient: {random.uniform(60,120):.0f} min, 5-35% B
    - MS: {random.choice(['Orbitrap','Q-TOF','triple quad'])} in DDA mode
    - Resolution: {random.uniform(60000,240000):.0f} (at m/z 200)

    Results:
    - Peptide spectrum matches (PSMs): {random.randint(5000,50000)}
    - Unique peptides: {random.randint(2000,20000)}
    - Protein groups: {random.randint(500,5000)}
    - FDR: < 1% (target-decoy approach)

    The Einstein relation E = mc^2 is not directly applicable here, but mass accuracy
    of {random.uniform(1,5):.1f} ppm allows unambiguous identification. Total sample
    mass injected was {random.uniform(0.5,5):.1f} ug ({random.uniform(0.0005,0.005):.4f} mg).
    Acquisition time per run: {random.uniform(60,150):.0f} min ({random.uniform(3600,9000):.0f} s).
    """),

    # 5 - gas phase
    lambda i: (f"chem_{i:03d}.txt", f"""
    Gas-Phase Reaction Study: {random.choice(['NO + O3 -> NO2 + O2','2NO2 <-> N2O4','CH4 + 2O2 -> CO2 + 2H2O','H2 + I2 -> 2HI'])}

    Reactor: {random.choice(['flow tube','static reactor','CSTR'])} with volume {random.uniform(0.5,5):.1f} L
    Temperature: {random.uniform(200,600):.0f} K
    Total pressure: {random.uniform(50,200):.0f} kPa ({random.uniform(50000,200000):.0f} Pa)

    Using the ideal gas law PV = nRT:
    - n = PV/RT = ({random.uniform(50,200):.0f} kPa * {random.uniform(0.5,5):.1f} L) / (8.314 J/(mol*K) * {random.uniform(200,600):.0f} K)
    - n = {random.uniform(0.01,0.5):.4f} mol

    Reactant partial pressures: {random.uniform(1,50):.1f} kPa each, balance {random.choice(['N2','Ar','He'])}
    Residence time: {random.uniform(0.1,60):.1f} s
    Conversion at exit: {random.uniform(10,95):.0f}%

    The rate constant was determined as k = {random.uniform(1e-3,1e2):.3e} cm^3/(molecule*s)
    at {random.uniform(200,600):.0f} K, giving an Arrhenius activation energy
    Ea = {random.uniform(10,80):.1f} kJ/mol from the relation k = A * e^{{-Ea/RT}}.
    The pre-exponential factor A = {random.uniform(1e-11,1e-9):.2e} cm^3/(molecule*s).

    Product analysis by FTIR showed complete consumption of the limiting reagent.
    Mass balance closure: {random.uniform(95,100):.1f}%. Total reactant mass:
    {random.uniform(0.5,10):.2f} g. Reactor wall temperature uniformity: +/- {random.uniform(1,5):.0f} K.
    """),
]

def gen_chemistry():
    for i in range(30):
        template = _chem_templates[i % len(_chem_templates)]
        fname, text = template(i)
        write_doc("chemistry", fname, text)


# =========================================================================
# HPC SIMULATION METADATA (30 docs)
# =========================================================================

_hpc_templates = [
    # 0 - weather forecast
    lambda i: (f"hpc_{i:03d}.txt", f"""
    HPC Job Summary: Weather Forecast Run WRF-{1000+i}

    Job ID: {random.randint(100000,999999)}
    Queue: {random.choice(['normal','large','premium','debug'])}
    Nodes: {random.choice([16,32,64,128,256])}
    Cores per node: {random.choice([36,48,64,128])}
    Wall-clock time: {random.uniform(0.5,12):.1f} h ({random.uniform(1800,43200):.0f} s)

    Model configuration:
    - Domain: {random.uniform(1000,5000):.0f} km x {random.uniform(1000,5000):.0f} km
    - Horizontal resolution: {random.uniform(1,15):.0f} km ({random.uniform(1000,15000):.0f} m)
    - Vertical levels: {random.randint(40,80)}
    - Top of atmosphere: {random.uniform(30,50):.0f} hPa ({random.uniform(3000,5000):.0f} Pa)
    - Time step: {random.uniform(10,60):.0f} s
    - Simulation length: {random.uniform(24,120):.0f} h

    Physics options:
    - Microphysics: {random.choice(['Thompson','Morrison','WSM6'])}
    - Radiation: {random.choice(['RRTMG','CAM','Dudhia'])}
    - PBL: {random.choice(['YSU','MYJ','MYNN2.5'])}
    - Convection: {random.choice(['Kain-Fritsch','Grell-3D','off (convection-permitting)'])}

    Output:
    - Surface pressure fields: {random.uniform(95,105):.1f} kPa range
    - Surface pressure in Pa: {random.uniform(95000,105000):.0f} Pa
    - 10-m wind speed: max {random.uniform(10,40):.1f} m/s ({random.uniform(36,144):.0f} km/h)
    - Total precipitation: max {random.uniform(10,200):.0f} mm
    - Output interval: {random.uniform(1,6):.0f} h
    - Total output size: {random.uniform(50,2000):.0f} GB

    Performance: {random.uniform(50,500):.0f} simulation-seconds per wall-second.
    Memory usage: {random.uniform(50,95):.0f}% of available. MPI communication overhead:
    {random.uniform(5,25):.0f}%.
    """),

    # 1 - CFD simulation
    lambda i: (f"hpc_{i:03d}.txt", f"""
    HPC Job Report: CFD Simulation RANS-{2000+i}

    Job ID: {random.randint(100000,999999)}
    Cluster: {random.choice(['Summit','Frontera','Perlmutter','Polaris'])}
    GPUs: {random.choice([4,8,16,32,64])} x {random.choice(['A100','V100','MI250X'])}
    Wall time: {random.uniform(2,48):.1f} h

    Mesh statistics:
    - Total cells: {random.uniform(1,500):.1f} million
    - Mesh type: {random.choice(['structured hexahedral','unstructured tetrahedral','polyhedral','hybrid'])}
    - Minimum cell size: {random.uniform(0.01,1):.3f} mm
    - Maximum cell size: {random.uniform(5,50):.1f} mm
    - Average y+ on walls: {random.uniform(0.5,30):.1f}

    Solver settings:
    - Algorithm: {random.choice(['SIMPLE','SIMPLEC','PISO','coupled'])}
    - Time step: {random.uniform(1e-5,1e-2):.1e} s
    - Total physical time: {random.uniform(0.1,10):.1f} s
    - Turbulence model: {random.choice(['k-omega SST','k-epsilon','LES Smagorinsky','DES'])}

    Convergence:
    - Residuals: < 1e-{random.randint(4,7)} for all equations
    - Iterations: {random.randint(1000,50000)}
    - Pressure range in domain: {random.uniform(80,120):.0f} kPa to {random.uniform(200,800):.0f} kPa
    - Velocity range: {random.uniform(0,5):.1f} to {random.uniform(10,200):.1f} m/s

    The maximum pressure of {random.uniform(200000,800000):.0f} Pa occurs at the stagnation
    point. Mesh spacing at critical regions was {random.uniform(0.05,0.5):.2f} mm to resolve
    the boundary layer. Grid independence was verified by Richardson extrapolation.
    """),

    # 2 - molecular dynamics
    lambda i: (f"hpc_{i:03d}.txt", f"""
    HPC Simulation Log: Molecular Dynamics, Run MD-{3000+i}

    System: {random.choice(['Protein in water','Lipid bilayer','Polymer melt','Metal nanoparticle'])}
    Atom count: {random.randint(50000,5000000)}
    Force field: {random.choice(['AMBER ff19SB','CHARMM36m','OPLS-AA','ReaxFF'])}

    Simulation parameters:
    - Ensemble: {random.choice(['NVT','NPT','NVE'])}
    - Temperature: {random.uniform(280,400):.0f} K
    - Pressure (if NPT): {random.uniform(0.1,1.0):.1f} MPa ({random.uniform(100,1000):.0f} kPa)
    - Time step: {random.uniform(1,4):.0f} fs ({random.uniform(1e-15,4e-15):.1e} s)
    - Total simulation time: {random.uniform(10,1000):.0f} ns
    - Steps: {random.randint(5000000,500000000)}

    Compute resources:
    - Nodes: {random.choice([1,2,4,8,16])}
    - GPUs: {random.choice([1,2,4,8])} x {random.choice(['A100','V100','RTX 3090'])}
    - Wall time: {random.uniform(6,168):.1f} h ({random.uniform(21600,604800):.0f} s)
    - Performance: {random.uniform(10,500):.0f} ns/day

    Output:
    - Trajectory frames: every {random.uniform(1,100):.0f} ps
    - Energy log: every {random.uniform(0.1,10):.1f} ps
    - Average pressure: {random.uniform(99,102):.1f} kPa (+/- {random.uniform(0.5,5):.1f} kPa)
    - RMSD from initial: {random.uniform(0.1,3):.1f} nm
    - Total output: {random.uniform(5,500):.0f} GB

    The target pressure of {random.uniform(99000,102000):.0f} Pa was maintained by the
    Parrinello-Rahman barostat. Electrostatics treated with PME, cutoff {random.uniform(0.9,1.4):.1f} nm.
    """),

    # 3 - climate model
    lambda i: (f"hpc_{i:03d}.txt", f"""
    HPC Run Summary: Climate Model CESM2, Experiment {4000+i}

    Configuration: {random.choice(['Historical','SSP2-4.5','SSP5-8.5','piControl','1pctCO2'])}
    Resolution: {random.choice(['1 deg','0.5 deg','0.25 deg'])} atmosphere,
    {random.choice(['1 deg','0.1 deg'])} ocean
    Simulation period: {random.randint(50,500)} years

    Compute allocation:
    - Cores: {random.choice([4096,8192,16384,32768])}
    - Wall time: {random.uniform(100,2000):.0f} h
    - Throughput: {random.uniform(2,20):.1f} simulated years per wall day
    - Storage: {random.uniform(5,100):.0f} TB

    Global mean diagnostics (last decade):
    - Surface temperature: {random.uniform(13,17):.2f} degC
    - Sea-level pressure: {random.uniform(1012,1014):.1f} hPa ({random.uniform(101200,101400):.0f} Pa)
    - Precipitation: {random.uniform(2.5,3.5):.2f} mm/day
    - TOA radiation imbalance: {random.uniform(0.5,2.0):.2f} W/m^2
    - Ocean heat uptake: {random.uniform(0.5,1.5):.2f} W/m^2

    The global mean surface pressure of {random.uniform(101200,101400):.0f} Pa shows
    {random.choice(['no significant','a slight positive','a slight negative'])} trend.
    Arctic surface wind speeds averaged {random.uniform(3,8):.1f} m/s with extremes
    reaching {random.uniform(20,40):.0f} m/s ({random.uniform(72,144):.0f} km/h).
    The model's time step for the atmosphere component is {random.uniform(600,1800):.0f} s
    ({random.uniform(10,30):.0f} min).
    """),

    # 4 - structural FEA
    lambda i: (f"hpc_{i:03d}.txt", f"""
    HPC Job: Finite Element Analysis, Case FEA-{5000+i}

    Analysis type: {random.choice(['static structural','modal','transient dynamic','fatigue'])}
    Component: {random.choice(['turbine blade','pressure vessel','bridge deck','automotive frame'])}
    Software: {random.choice(['ABAQUS','ANSYS Mechanical','LS-DYNA','OpenFOAM'])}

    Model details:
    - Nodes: {random.uniform(0.5,10):.1f} million
    - Elements: {random.uniform(0.3,8):.1f} million ({random.choice(['C3D8R','C3D10M','S4R','SHELL181'])})
    - Minimum element size: {random.uniform(0.5,5):.1f} mm
    - Maximum element size: {random.uniform(10,100):.0f} mm
    - Material: {random.choice(['steel','aluminum','titanium','composite'])}

    Loading:
    - Applied pressure: {random.uniform(0.5,50):.1f} MPa ({random.uniform(500,50000):.0f} kPa) on inner surface
    - Temperature load: {random.uniform(20,500):.0f} degC
    - Boundary conditions: fixed at {random.choice(['flange','base','supports'])}

    Results:
    - Max von Mises stress: {random.uniform(100,800):.0f} MPa
    - Max displacement: {random.uniform(0.1,10):.2f} mm ({random.uniform(0.0001,0.01):.4f} m)
    - Factor of safety: {random.uniform(1.2,4):.1f}
    - First natural frequency: {random.uniform(10,500):.0f} Hz

    The peak stress of {random.uniform(100000,800000):.0f} kPa occurs at the fillet radius.
    Mesh convergence study: stress changed < {random.uniform(1,5):.0f}% between final two
    refinement levels. Wall time: {random.uniform(1,24):.1f} h on {random.randint(8,128)} cores.
    Total output size: {random.uniform(2,100):.0f} GB.
    """),

    # 5 - DNS turbulence
    lambda i: (f"hpc_{i:03d}.txt", f"""
    HPC Report: Direct Numerical Simulation of Turbulence, DNS-{6000+i}

    Configuration: {random.choice(['Channel flow','Isotropic turbulence','Boundary layer','Mixing layer'])}
    Reynolds number: Re_tau = {random.randint(180,5200)}

    Grid:
    - Nx x Ny x Nz = {random.choice([256,512,1024])} x {random.choice([192,256,384])} x {random.choice([256,512,1024])}
    - Total grid points: {random.uniform(12,500):.0f} million
    - Streamwise spacing: delta_x+ = {random.uniform(5,15):.1f}
    - Wall-normal (min): delta_y+ = {random.uniform(0.3,1):.2f}
    - Spanwise spacing: delta_z+ = {random.uniform(3,8):.1f}
    - Physical domain: {random.uniform(2,12):.0f} x 2 x {random.uniform(1,6):.0f} (in delta units)

    Time advancement:
    - Time step: {random.uniform(1e-5,1e-3):.1e} s
    - CFL: < {random.uniform(0.3,0.8):.1f}
    - Simulation time: {random.uniform(50,500):.0f} delta/u_tau
    - Statistical averaging: {random.uniform(20,200):.0f} delta/u_tau

    Resources:
    - Cores: {random.choice([4096,8192,16384,65536])}
    - Wall time: {random.uniform(50,500):.0f} h ({random.uniform(180000,1800000):.0f} s)
    - Memory: {random.uniform(1,20):.1f} TB
    - Output: {random.uniform(10,200):.0f} TB

    Key statistics:
    - Bulk velocity: {random.uniform(1,20):.1f} m/s
    - Friction velocity: {random.uniform(0.03,0.5):.3f} m/s
    - Wall shear stress: {random.uniform(0.5,50):.1f} Pa
    - Mean pressure gradient: {random.uniform(-10,-0.1):.2f} Pa/m

    Grid spacing at walls was {random.uniform(0.005,0.05):.3f} mm to resolve the viscous sublayer.
    The pressure difference between inlet and outlet was {random.uniform(0.1,5):.2f} kPa.
    """),
]

def gen_hpc_simulation():
    for i in range(30):
        template = _hpc_templates[i % len(_hpc_templates)]
        fname, text = template(i)
        write_doc("hpc_simulation", fname, text)


# =========================================================================
# MIXED / CROSS-DOMAIN (20 docs)
# =========================================================================

_mixed_templates = [
    # 0 - wind energy
    lambda i: (f"mixed_{i:03d}.txt", f"""
    Wind Energy Assessment: Turbine Site {100+i}

    This multi-disciplinary report covers the aerodynamic, structural, and atmospheric
    analysis for a proposed wind turbine installation.

    Atmospheric conditions (annual averages):
    - Mean wind speed at hub height (80 m): {random.uniform(5,12):.1f} m/s ({random.uniform(18,43.2):.1f} km/h)
    - Wind power density: {random.uniform(200,800):.0f} W/m^2
    - Extreme gust (50-year return): {random.uniform(40,70):.0f} m/s
    - Barometric pressure (site elevation {random.uniform(100,2000):.0f} m): {random.uniform(80,101):.1f} kPa
    - Air density: {random.uniform(1.0,1.25):.3f} kg/m^3

    Turbine specifications:
    - Rotor diameter: {random.uniform(80,170):.0f} m ({random.uniform(80000,170000):.0f} mm)
    - Blade length: {random.uniform(38,83):.0f} m
    - Blade root thickness: {random.uniform(40,80):.0f} mm
    - Tower height: {random.uniform(60,120):.0f} m
    - Rated wind speed: {random.uniform(10,15):.1f} m/s
    - Cut-in speed: {random.uniform(3,5):.0f} m/s

    Structural analysis:
    - Maximum blade root bending moment: {random.uniform(5,30):.1f} MN*m
    - Blade tip deflection: {random.uniform(3,8):.1f} m ({random.uniform(3000,8000):.0f} mm)
    - Tower base von Mises stress: {random.uniform(100,250):.0f} MPa ({random.uniform(100000,250000):.0f} kPa)
    - Foundation design pressure: {random.uniform(200,500):.0f} kPa on soil

    The power output P = 0.5 * rho * A * Cp * v^3 at rated conditions gives
    approximately {random.uniform(2,8):.1f} MW. Annual energy production estimate:
    {random.uniform(5,25):.0f} GWh at a capacity factor of {random.uniform(25,50):.0f}%.
    Fatigue life assessment: 20+ years at {random.uniform(1e7,1e9):.1e} cycles.
    """),

    # 1 - aerospace testing
    lambda i: (f"mixed_{i:03d}.txt", f"""
    Aerospace Component Test: Thermal-Structural Qualification, Test {200+i}

    Component: {random.choice(['Rocket nozzle extension','Heat shield tile','Turbopump housing','Combustion liner'])}
    Material: {random.choice(['Inconel 718','C/SiC composite','Hastelloy X','Niobium alloy C-103'])}

    Thermal loading:
    - Gas temperature: {random.uniform(800,2500):.0f} degC
    - Heat flux: {random.uniform(0.5,10):.1f} MW/m^2
    - Cooling fluid: {random.choice(['GN2','LN2','water','helium'])}
    - Coolant pressure: {random.uniform(1,30):.0f} MPa ({random.uniform(1000,30000):.0f} kPa)
    - Coolant flow velocity: {random.uniform(5,50):.0f} m/s ({random.uniform(18,180):.0f} km/h)

    Structural measurements:
    - Peak thermal stress: {random.uniform(200,800):.0f} MPa
    - Membrane stress: {random.uniform(50,300):.0f} MPa
    - Wall thickness: {random.uniform(1,10):.1f} mm
    - Displacement at tip: {random.uniform(0.5,5):.1f} mm
    - Component mass: {random.uniform(2,50):.1f} kg ({random.uniform(2000,50000):.0f} g)

    Pressure measurements:
    - Chamber pressure: {random.uniform(2,20):.0f} MPa
    - Nozzle exit pressure: {random.uniform(10,101):.0f} kPa ({random.uniform(10000,101000):.0f} Pa)
    - External ambient: {random.uniform(0.1,101):.1f} kPa

    Test duration: {random.uniform(30,600):.0f} s ({random.uniform(0.5,10):.1f} min). The component
    survived {random.randint(1,50)} thermal cycles without cracking. Post-test inspection
    by DPI and X-ray revealed no defects exceeding the acceptance criteria of
    {random.uniform(0.1,1):.1f} mm indication length.
    """),

    # 2 - biomedical
    lambda i: (f"mixed_{i:03d}.txt", f"""
    Biomedical Device Testing: Blood Pump Characterization, Unit BP-{300+i}

    A centrifugal blood pump was tested on a mock circulatory loop to characterize
    hydrodynamic performance and hemolysis.

    Test loop configuration:
    - Tubing ID: {random.uniform(6,12):.0f} mm ({random.uniform(0.006,0.012):.3f} m)
    - Reservoir volume: {random.uniform(0.5,2):.1f} L
    - Working fluid: {random.choice(['bovine blood','glycerol-water (40%)','porcine blood'])}
    - Temperature: {random.uniform(35,38):.1f} degC
    - Viscosity: {random.uniform(2.5,4.5):.1f} cP

    Pump performance at {random.randint(2000,4000)} rpm:
    - Flow rate: {random.uniform(2,8):.1f} L/min
    - Pressure rise: {random.uniform(60,150):.0f} mmHg ({random.uniform(8,20):.1f} kPa)
    - Outlet pressure: {random.uniform(100,130):.0f} mmHg ({random.uniform(13.3,17.3):.1f} kPa)
    - Flow velocity at outlet: {random.uniform(0.5,3):.1f} m/s
    - Maximum shear stress: {random.uniform(50,500):.0f} Pa

    Hemolysis assessment:
    - Test duration: {random.uniform(2,6):.0f} h ({random.uniform(7200,21600):.0f} s)
    - Plasma-free hemoglobin increase: {random.uniform(5,50):.0f} mg/dL
    - Normalized index of hemolysis: {random.uniform(0.001,0.05):.4f} g/100L

    The pump housing wall thickness was {random.uniform(2,5):.1f} mm (polycarbonate, sterilizable).
    Mass of pump head: {random.uniform(100,500):.0f} g. Bearing gap: {random.uniform(0.05,0.2):.2f} mm.
    Reynolds number in the impeller passage: approximately {random.randint(5000,50000)}.
    """),

    # 3 - geotechnical
    lambda i: (f"mixed_{i:03d}.txt", f"""
    Geotechnical Site Investigation: Borehole BH-{400+i}

    Location: {random.choice(['Downtown construction site','Highway bridge abutment','Wind farm foundation','Seaport expansion'])}
    Depth: {random.uniform(10,50):.1f} m below ground level

    Soil profile:
    - 0-{random.uniform(2,5):.1f} m: Fill material, unit weight {random.uniform(16,19):.0f} kN/m^3
    - {random.uniform(2,5):.1f}-{random.uniform(8,15):.1f} m: {random.choice(['Soft clay','Medium clay','Stiff clay','Loose sand'])}
    - {random.uniform(8,15):.1f}-{random.uniform(20,40):.1f} m: {random.choice(['Dense sand','Gravel','Weathered rock','Stiff glacial till'])}
    - Below {random.uniform(20,40):.1f} m: {random.choice(['Bedrock (limestone)','Bedrock (sandstone)','Hardpan'])}

    In-situ testing:
    - SPT N-values: {random.randint(2,10)} (soft layer) to {random.randint(30,100)} (dense layer)
    - CPT tip resistance: {random.uniform(0.5,30):.1f} MPa ({random.uniform(500,30000):.0f} kPa)
    - Pore water pressure at {random.uniform(5,15):.0f} m: {random.uniform(50,150):.0f} kPa ({random.uniform(50000,150000):.0f} Pa)
    - Groundwater table: {random.uniform(1,8):.1f} m below surface

    Laboratory tests:
    - Unconfined compressive strength: {random.uniform(20,200):.0f} kPa
    - Consolidation: Cc = {random.uniform(0.1,0.8):.2f}, preconsolidation pressure {random.uniform(100,500):.0f} kPa
    - Permeability: {random.uniform(1e-9,1e-5):.1e} m/s ({random.uniform(0.001,10):.3f} mm/s)

    The effective overburden pressure at {random.uniform(10,20):.0f} m depth is approximately
    {random.uniform(100,300):.0f} kPa. Foundation design: mat foundation, bearing capacity
    {random.uniform(200,600):.0f} kPa (allowable). Settlement estimate: {random.uniform(10,100):.0f} mm
    over {random.uniform(5,30):.0f} years. Sample mass extracted: {random.uniform(2,10):.1f} kg.
    """),
]

def gen_mixed():
    for i in range(20):
        template = _mixed_templates[i % len(_mixed_templates)]
        fname, text = template(i)
        write_doc("mixed", fname, text)


# =========================================================================
# NEGATIVE DOCUMENTS (10 docs -- no extractable measurements)
# =========================================================================

_neg_texts = [
    """
    Literature Review: History of Turbulence Modeling

    The study of turbulence has captivated fluid dynamicists for over a century.
    Reynolds' seminal decomposition of flow variables into mean and fluctuating
    components laid the groundwork for statistical approaches. Boussinesq's eddy
    viscosity hypothesis, while remarkably useful, remains an approximation that
    fails in many complex flows.

    Prandtl's mixing length theory provided the first practical closure model,
    enabling engineering calculations of boundary layers and pipe flows. The
    subsequent development of two-equation models (k-epsilon by Launder and
    Spalding, k-omega by Wilcox) represented a major advance in predictive
    capability. These models remain workhorses of industrial CFD today.

    More recent developments include large eddy simulation (LES), where the
    large energy-containing scales are resolved directly while only the small
    scales require modeling. Direct numerical simulation (DNS) resolves all
    scales but remains computationally prohibitive for most engineering flows.

    The quest for a universal turbulence model continues. Machine learning
    approaches show promise for data-driven closure models, potentially
    combining the efficiency of RANS with improved accuracy in separated and
    recirculating flows. However, generalizability remains an open question.
    """,

    """
    Editorial: The Reproducibility Crisis in Materials Science

    Recent studies have highlighted concerning trends in the reproducibility of
    published materials science results. A survey of leading journals found that
    fewer than half of experimental studies provided sufficient detail for
    independent replication. Key issues include incomplete reporting of synthesis
    conditions, ambiguous characterization protocols, and selective reporting of
    favorable results.

    The community has responded with initiatives such as FAIR data principles
    (Findable, Accessible, Interoperable, Reusable) and mandatory data
    deposition requirements. Some journals now require authors to submit raw
    data and analysis scripts alongside manuscripts.

    However, cultural change is slow. The incentive structure in academia still
    rewards novelty over rigor, and replication studies struggle to find
    publication venues. Several grassroots efforts have emerged, including
    community-organized replication challenges and open-source materials databases.

    Moving forward, the integration of automated experiment platforms and
    electronic lab notebooks promises to improve documentation and traceability.
    The development of standardized reporting guidelines specific to different
    subdisciplines would also help. Ultimately, the credibility of materials
    science depends on our collective commitment to transparency and rigor.
    """,

    """
    Tutorial: Introduction to Finite Element Method

    The finite element method (FEM) is a numerical technique for solving partial
    differential equations by discretizing a continuous domain into a finite
    number of elements. Each element is described by shape functions that
    approximate the variation of the unknown field variable within that element.

    The key steps in a finite element analysis are:
    First, the governing equations are cast in weak form using the principle of
    virtual work or the Galerkin method. Second, the domain is discretized into
    elements (triangles, quadrilaterals in two dimensions; tetrahedra, hexahedra
    in three dimensions). Third, shape functions are chosen to interpolate the
    solution within each element. Fourth, the element equations are assembled
    into a global system of algebraic equations. Finally, boundary conditions
    are applied and the system is solved.

    Convergence of the finite element solution depends on the mesh refinement
    (h-refinement), the polynomial order of shape functions (p-refinement),
    and the quality of the mesh. Poorly shaped elements (high aspect ratio,
    extreme angles) degrade accuracy and can cause numerical difficulties.

    Common applications include structural mechanics, heat transfer, fluid
    dynamics, and electromagnetics. Modern FEM software packages provide
    sophisticated pre-processing, solving, and post-processing capabilities.
    """,

    """
    Conference Announcement: International Symposium on Computational Mechanics

    We are pleased to announce the forthcoming International Symposium on
    Computational Mechanics, to be held at the Convention Center in the first
    quarter of next year.

    The symposium will feature plenary lectures by distinguished researchers,
    contributed presentations organized in parallel sessions, and a poster
    exhibition. Topics of interest include but are not limited to: multiscale
    modeling, uncertainty quantification, machine learning in mechanics,
    topology optimization, additive manufacturing simulation, and biomechanics.

    Abstract submission is now open through the symposium website. Authors are
    encouraged to submit extended abstracts of no more than two pages.
    Accepted abstracts will be published in the symposium proceedings. Selected
    contributions will be invited for a special issue in a peer-reviewed journal.

    Early registration is available at a reduced fee. Student travel grants are
    available through the organizing committee. The venue offers convenient
    access to public transportation and numerous accommodation options.

    We look forward to welcoming researchers, engineers, and students from
    around the world to this stimulating event. For further information,
    please visit the symposium website or contact the organizing committee.
    """,

    """
    Book Review: Fundamentals of Atmospheric Dynamics

    This comprehensive textbook provides an excellent introduction to the
    physical processes governing atmospheric motion. The author develops the
    governing equations from first principles, starting with the Navier-Stokes
    equations on a rotating sphere and systematically introducing the
    approximations that lead to the primitive equations.

    Particularly noteworthy is the treatment of quasi-geostrophic theory, which
    is presented with remarkable clarity. The chapter on baroclinic instability
    builds intuition through a series of progressively more realistic models,
    from the Eady problem to the Charney model. The inclusion of modern
    perspectives on potential vorticity dynamics is a welcome addition.

    The exercises range from straightforward derivations to challenging
    computational problems suitable for graduate-level courses. The appendices
    provide useful reference material on vector calculus in spherical
    coordinates and thermodynamic relations.

    Criticisms are minor: some figures could benefit from color, and the
    treatment of tropical dynamics, while adequate, lacks the depth of the
    midlatitude chapters. Overall, this is an outstanding resource for
    graduate students and researchers entering the field.
    """,

    """
    Safety Protocol: Laboratory Chemical Handling Procedures

    All laboratory personnel must complete chemical safety training before
    handling any hazardous materials. This document outlines the standard
    operating procedures for common laboratory chemicals.

    Personal protective equipment (PPE) requirements vary by chemical class.
    At minimum, safety glasses, lab coat, and closed-toe shoes are required
    in all wet lab areas. Nitrile gloves are required when handling organic
    solvents, acids, or bases. A face shield is required when handling
    concentrated acids or when performing reactions that may splatter.

    Chemical storage follows the segregation matrix posted in each lab.
    Incompatible chemicals must never be stored together. Flammable solvents
    must be stored in approved flammable storage cabinets. Oxidizers must be
    stored separately from organic materials. Acids and bases must be stored
    in separate secondary containment.

    Waste disposal follows the institutional waste management plan. Chemical
    waste must be properly labeled with contents, hazards, and generator
    information. Mixed waste streams are generally more expensive to dispose
    of and should be avoided when possible. Sharps, broken glass, and
    contaminated materials have separate disposal streams.

    Emergency procedures: in case of chemical spill, alert nearby personnel,
    consult the SDS, and contact the safety office. For small spills of known
    chemicals, trained personnel may clean up using appropriate spill kits.
    """,

    """
    Grant Proposal Abstract: Advanced Manufacturing for Sustainable Energy

    This proposal seeks support for a three-year research program investigating
    novel manufacturing processes for next-generation energy conversion devices.
    The interdisciplinary team combines expertise in materials processing,
    computational modeling, and device characterization.

    The primary objective is to develop scalable fabrication routes for
    perovskite-silicon tandem solar cells that achieve both high efficiency
    and long-term stability. Current laboratory demonstrations have shown
    promising performance, but translating these results to manufacturing
    scale presents significant challenges in film uniformity, defect control,
    and encapsulation.

    The proposed approach integrates high-throughput experimentation with
    machine learning-guided process optimization. An automated deposition
    platform will explore a wide parameter space while minimizing material
    consumption. Physics-informed neural networks will be trained on the
    experimental data to develop predictive process-structure-property models.

    Broader impacts include training of graduate students and postdocs in
    advanced manufacturing techniques, development of open-source software
    tools for the community, and outreach activities targeting underrepresented
    groups in STEM. Industrial partnerships will facilitate technology transfer
    and ensure relevance of the research to manufacturing practice.

    The expected outcomes include demonstrated tandem cell efficiency exceeding
    current records, a validated digital twin for the manufacturing process,
    and at least ten peer-reviewed publications.
    """,

    """
    Meeting Minutes: Departmental Research Computing Committee

    The committee convened to discuss the upcoming cluster upgrade and resource
    allocation policies for the next fiscal year.

    Regarding the cluster upgrade, the chair reported that vendor proposals
    have been received and evaluated. The recommended configuration includes
    newer generation processors, increased memory per node, and additional
    GPU accelerators. The timeline for procurement and installation is
    approximately six months from approval.

    The committee discussed job scheduling policies. Several users have
    reported long queue wait times for large parallel jobs. The proposed
    solution involves dedicated time slots for capability computing jobs and
    improved backfill scheduling for smaller jobs. A fair-share algorithm
    will replace the current first-come-first-served policy.

    Software environment management was also discussed. The transition to
    a container-based workflow was endorsed to improve reproducibility and
    reduce dependency conflicts. Training workshops will be organized to
    help users adapt their workflows.

    Data storage remains a concern. The current parallel filesystem is
    approaching capacity. Options include expanding existing storage,
    implementing tiered storage with automatic migration, or establishing
    a data lifecycle management policy. The committee will draft a proposal
    for the next meeting.

    The next meeting is scheduled for three weeks from today.
    """,

    """
    Philosophical Reflections on Measurement and Uncertainty

    The act of measurement lies at the heart of empirical science. Yet the
    seemingly simple question of what it means to measure something reveals
    deep philosophical complexities. Every measurement is mediated by
    instruments, procedures, and theoretical frameworks that shape what
    can be observed and how it is interpreted.

    The concept of measurement uncertainty, formalized in the Guide to the
    Expression of Uncertainty in Measurement (GUM), provides a rigorous
    framework for quantifying the reliability of measurement results. Yet
    even this framework rests on assumptions about the nature of probability
    and the completeness of our uncertainty models.

    In quantum mechanics, the Heisenberg uncertainty principle imposes
    fundamental limits on the simultaneous knowledge of conjugate variables.
    This is not a limitation of our instruments but a feature of nature
    itself. The implications for the philosophy of science are profound:
    there exist questions about physical systems that are in principle
    unanswerable.

    In practice, most measurement uncertainty arises from more mundane
    sources: calibration drift, environmental fluctuations, operator
    variability, and modeling approximations. The art of good measurement
    lies in identifying and controlling these sources systematically.
    """,

    """
    Alumni Newsletter: Department of Mechanical Engineering

    Congratulations to our recent graduates! This year's class includes
    twenty doctoral and forty master's students who have gone on to
    positions in academia, industry, and national laboratories.

    Research highlights from the past year include new funded projects in
    robotics, clean energy, and advanced materials. Professor Smith's group
    published a highly cited paper on topology optimization, while Professor
    Jones received a prestigious early career award for work on micro-scale
    heat transfer.

    The department welcomed three new faculty members specializing in
    autonomous systems, computational biomechanics, and sustainable
    manufacturing. Their expertise complements existing strengths and opens
    new interdisciplinary research directions.

    Infrastructure improvements include a renovated manufacturing lab with
    new equipment for additive and subtractive processes, and an upgraded
    computing cluster that supports the department's growing computational
    research portfolio.

    The annual alumni reception will be held in conjunction with the
    professional society conference this fall. We encourage all alumni to
    attend and reconnect with faculty and fellow graduates. Updates on
    department activities are available on our website and social media.
    """,
]

def gen_negatives():
    for i, text in enumerate(_neg_texts):
        write_doc("negatives", f"neg_{i:03d}.txt", text)


# =========================================================================
# MAIN
# =========================================================================

def main():
    gen_materials_science()
    gen_atmospheric_science()
    gen_fluid_dynamics()
    gen_chemistry()
    gen_hpc_simulation()
    gen_mixed()
    gen_negatives()

    # Count
    total = 0
    for domain_dir in sorted(CORPUS_DIR.iterdir()):
        if domain_dir.is_dir():
            count = len(list(domain_dir.glob("*.txt")))
            print(f"  {domain_dir.name}: {count} docs")
            total += count
    print(f"  TOTAL: {total} documents")


if __name__ == "__main__":
    main()
