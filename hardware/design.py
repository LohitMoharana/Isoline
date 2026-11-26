"""Single source of truth for the Isoline analog front end (Architecture A).
Both hardware/sim/make_afe_sim.py (ngspice) and hardware/kicad/gen/make_netlist.py (KiCad) read these values.
"""

VSUPPLY = 3.0            # 2 x AAA, nominal (range ~2.0-3.2 V; all parts run from 1.8 V)
VREF = 1.5               # mid-rail, buffered divider

# In-amp: INA333, G = 1 + 100k / RG
INA_RG = "11.1k"         # G1 = 10.0
INA_G1 = 1 + 100e3 / 11.1e3

# DC servo integrator on the in-amp REF pin: f_hp = 1 / (2*pi*RS*CS) = 0.048 Hz
SERVO_R = "10MEG"
SERVO_C = "330n"
FAST_RESTORE_R = "100k"  # switched in parallel with SERVO_R by a MOSFET at power-up -> 100x faster settling

# Gain stage (non-inverting about VREF): G2 = 1 + RF / RG
G2_RG = "10k"
G2_RF = "290k"           # G2 = 30 -> total gain 300 (0.1 mV input step = 30 mV at the comparators)

# Unity-gain Sallen-Key Butterworth low-pass, fc = 40 Hz
SK_R1 = "100k"
SK_R2 = "100k"
SK_C1 = "56n"            # feedback capacitor
SK_C2 = "27n"            # to VREF

# Mid-rail reference
VREF_DIV_R = "1MEG"      # two of these from VBAT to GND, buffered
VREF_C = "1u"

# Input network
BIAS_R = "10MEG"         # each electrode input to VREF (DC path)
INPUT_R = "10k"          # series protection
RLD_R = "10k"            # right-leg / reference electrode to VREF

# Hardware-in-the-loop input: attenuator ~1/1000 (1 V from a DAC/audio out -> ~1 mV "ECG")
HIL_R_TOP = "100k"
HIL_R_BOT = "100"

TOTAL_GAIN = INA_G1 * (1 + 290e3 / 10e3)
