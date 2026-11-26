"""Generate the KiCad netlist + BOM for the Isoline Architecture-A carrier board (Nucleo-L432KC shield).

All component values come from hardware/design.py, the same file the ngspice simulation reads.
IC pinouts were checked against the TI datasheets (INA333 DGK, TLV8542 D, TLV3691 DCK, SN74LVC1G66 DCK).
Nucleo-32 header pin order follows UM1956 (Arduino Nano layout): CN3 = D1,D0,RST,GND,D2..D12; CN4 = VIN,GND,RST,
+5V,A7..A0,AREF,+3V3,D13.

Import into KiCad PCB editor: File > Import > Netlist... (footprints are standard KiCad library names).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import design as DZ  # noqa: E402
from skidl import SKIDL, Net, Part, Pin, generate_netlist, set_default_tool, KICAD8  # noqa: E402

set_default_tool(KICAD8)

FP_R = "Resistor_SMD:R_0805_2012Metric"
FP_C = "Capacitor_SMD:C_0805_2012Metric"
BOM = []


def mk(name, ref, fp, pins, value=None, desc=""):
    p = Part(tool=SKIDL, name=name, ref_prefix=ref, footprint=fp,
             pins=[Pin(num=n, name=nm) for n, nm in pins])
    p.value = value or name
    BOM.append((name, value or name, fp, desc))
    return p


def R(val, a, b, desc=""):
    r = mk("R", "R", FP_R, [(1, "1"), (2, "2")], val, desc); r[1] += a; r[2] += b; return r


def C(val, a, b, desc=""):
    c = mk("C", "C", FP_C, [(1, "1"), (2, "2")], val, desc); c[1] += a; c[2] += b; return c


def opamp_dual(desc):
    return mk("TLV8542", "U", "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm",
              [(1, "OUTA"), (2, "-INA"), (3, "+INA"), (4, "V-"), (5, "+INB"), (6, "-INB"), (7, "OUTB"), (8, "V+")],
              "TLV8542ID", desc)


# ---------------------------------------------------------------- nets
gnd, vcc, vref = Net("GND"), Net("VCC"), Net("VREF")
bat, nuc3v3 = Net("VBAT"), Net("NUC_3V3")
e_pos, e_neg, e_ref = Net("E_POS"), Net("E_NEG"), Net("E_REF")
in_p, in_n = Net("INA_INP"), Net("INA_INN")
rg_a, rg_b = Net("RG_A"), Net("RG_B")
ina_out, srv, s_in, fr = Net("INA_OUT"), Net("SERVO_OUT"), Net("SERVO_IN"), Net("FR_NODE")
g_inv, g_out, sk1, sk2 = Net("G2_INV"), Net("G2_OUT"), Net("SK1"), Net("SK2")
afe_out, th_up, th_dn = Net("AFE_OUT"), Net("TH_UP"), Net("TH_DN")
ev_up, ev_dn, fr_ctl, vdiv, afe_mon = Net("EV_UP"), Net("EV_DN"), Net("FAST_RESTORE"), Net("VREF_DIV"), Net("AFE_MON")
hil_sig, hil_ac, hil_r = Net("HIL_SIG"), Net("HIL_AC"), Net("HIL_ATT")

# ---------------------------------------------------------------- power
bt = mk("Battery_2xAAA", "BT", "Battery:BatteryHolder_Keystone_2468_2xAAA", [(1, "+"), (2, "-")], "2xAAA",
        "2 x AAA holder (Keystone 2468)")
bt[1] += bat; bt[2] += gnd
jp1 = mk("Jumper_3", "JP", "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical",
         [(1, "1"), (2, "2"), (3, "3")], "PWR_SEL", "1-2: battery powers board, 2-3: Nucleo 3V3 (USB) powers board")
jp1[1] += bat; jp1[2] += vcc; jp1[3] += nuc3v3
C("10u", vcc, gnd, "bulk"); C("100n", vcc, gnd, "bulk HF")

# mid-rail reference: divider + buffer (U3B)
R(DZ.VREF_DIV_R, vcc, vdiv, "VREF divider top"); R(DZ.VREF_DIV_R, vdiv, gnd, "VREF divider bottom")
C(DZ.VREF_C, vdiv, gnd, "VREF divider filter"); C("1u", vref, gnd, "VREF output")

# ---------------------------------------------------------------- electrodes + HIL input
j1 = mk("Conn_Electrodes", "J", "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical",
        [(1, "E_NEG"), (2, "E_POS"), (3, "E_REF")], "ELECTRODES", "1: RA (-), 2: LL (+), 3: RL (reference)")
j1[1] += e_neg; j1[2] += e_pos; j1[3] += e_ref
R(DZ.INPUT_R, e_pos, in_p, "input protection +"); R(DZ.INPUT_R, e_neg, in_n, "input protection -")
R(DZ.BIAS_R, in_p, vref, "bias +"); R(DZ.BIAS_R, in_n, vref, "bias -")
R(DZ.RLD_R, e_ref, vref, "reference electrode to VREF")
j2 = mk("Conn_HIL", "J", "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
        [(1, "SIG"), (2, "GND")], "HIL_IN", "hardware-in-the-loop input (DAC / audio out)")
j2[1] += hil_sig; j2[2] += gnd
C("1u", hil_sig, hil_ac, "HIL AC coupling"); R(DZ.HIL_R_TOP, hil_ac, e_pos, "HIL attenuator top (~1/1000)")
jp2 = mk("Jumper_2", "JP", "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical", [(1, "1"), (2, "2")],
         "HIL_ATT_EN", "fit only in HIL mode: 100 R across the inputs")
jp2[1] += e_pos; jp2[2] += hil_r
R(DZ.HIL_R_BOT, hil_r, e_neg, "HIL attenuator bottom")
jp3 = mk("Jumper_2", "JP", "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical", [(1, "1"), (2, "2")],
         "HIL_NEG_REF", "fit only in HIL mode: E_NEG to reference")
jp3[1] += e_neg; jp3[2] += e_ref

# ---------------------------------------------------------------- in-amp + servo + gain + LPF
u1 = mk("INA333", "U", "Package_SO:VSSOP-8_3x3mm_P0.65mm",
        [(1, "RG"), (2, "VIN-"), (3, "VIN+"), (4, "V-"), (5, "REF"), (6, "VOUT"), (7, "V+"), (8, "RG_")],
        "INA333AIDGK", "in-amp, G = 1 + 100k/RG")
u1["RG"] += rg_a; u1["RG_"] += rg_b; u1["VIN-"] += in_n; u1["VIN+"] += in_p
u1["V-"] += gnd; u1["V+"] += vcc; u1["REF"] += srv; u1["VOUT"] += ina_out
R(DZ.INA_RG, rg_a, rg_b, "INA gain resistor (G1 = 10), 0.1 %")
C("100n", vcc, gnd, "U1 decoupling")

u2 = opamp_dual("servo integrator (A) + gain x30 (B)")
u2["V-"] += gnd; u2["V+"] += vcc; C("100n", vcc, gnd, "U2 decoupling")
u2["+INA"] += vref; u2["-INA"] += s_in; u2["OUTA"] += srv
R(DZ.SERVO_R, ina_out, s_in, "servo R (0.05 Hz)"); C(DZ.SERVO_C, s_in, srv, "servo C, film/C0G-X7R low leakage")
R(DZ.FAST_RESTORE_R, ina_out, fr, "fast-restore R")
u6 = mk("SN74LVC1G66", "U", "Package_TO_SOT_SMD:SOT-353_SC-70-5",
        [(1, "A"), (2, "B"), (3, "GND"), (4, "C"), (5, "VCC")], "SN74LVC1G66DCK", "fast-restore analog switch")
u6["A"] += fr; u6["B"] += s_in; u6["GND"] += gnd; u6["VCC"] += vcc; u6["C"] += fr_ctl
R("100k", fr_ctl, gnd, "fast-restore pull-down (switch off by default)")
C("100n", vcc, gnd, "U6 decoupling")
u2["+INB"] += ina_out; u2["-INB"] += g_inv; u2["OUTB"] += g_out
R(DZ.G2_RG, g_inv, vref, "gain stage RG"); R(DZ.G2_RF, g_inv, g_out, "gain stage RF (x30)")

u3 = opamp_dual("Sallen-Key LPF (A) + VREF buffer (B)")
u3["V-"] += gnd; u3["V+"] += vcc; C("100n", vcc, gnd, "U3 decoupling")
R(DZ.SK_R1, g_out, sk1, "SK R1"); R(DZ.SK_R2, sk1, sk2, "SK R2")
C(DZ.SK_C1, sk1, afe_out, "SK C1 (feedback)"); C(DZ.SK_C2, sk2, vref, "SK C2")
u3["+INA"] += sk2; u3["-INA"] += afe_out; u3["OUTA"] += afe_out
u3["+INB"] += vdiv; u3["-INB"] += vref; u3["OUTB"] += vref

# ---------------------------------------------------------------- comparators (thresholds from MCU DAC)
for name, plus, minus, out, desc in (("UP", afe_out, th_up, ev_up, "up event: AFE above upper threshold"),
                                     ("DN", th_dn, afe_out, ev_dn, "down event: AFE below lower threshold")):
    u = mk("TLV3691", "U", "Package_TO_SOT_SMD:SOT-353_SC-70-5",
           [(1, "IN+"), (2, "GND"), (3, "IN-"), (4, "OUT"), (5, "VCC")], "TLV3691IDCK", desc)
    u["IN+"] += plus; u["IN-"] += minus; u["OUT"] += out; u["GND"] += gnd; u["VCC"] += vcc
    C("100n", vcc, gnd, f"comparator {name} decoupling")
C("10n", th_up, gnd, "DAC1 hold"); C("10n", th_dn, gnd, "DAC2 hold")
R("1k", afe_out, afe_mon, "AFE monitor to ADC (debug)")

# ---------------------------------------------------------------- Nucleo-L432KC headers (UM1956 Nano layout)
cn3 = mk("Nucleo32_CN3", "J", "Connector_PinSocket_2.54mm:PinSocket_1x15_P2.54mm_Vertical",
         [(i + 1, n) for i, n in enumerate(["D1", "D0", "RESET", "GND", "D2", "D3", "D4", "D5", "D6", "D7",
                                             "D8", "D9", "D10", "D11", "D12"])], "NUCLEO_CN3", "Nucleo-32 left header")
cn4 = mk("Nucleo32_CN4", "J", "Connector_PinSocket_2.54mm:PinSocket_1x15_P2.54mm_Vertical",
         [(i + 1, n) for i, n in enumerate(["VIN", "GND", "RESET", "+5V", "A7", "A6", "A5", "A4", "A3", "A2",
                                             "A1", "A0", "AREF", "+3V3", "D13"])], "NUCLEO_CN4",
         "Nucleo-32 right header, 15.24 mm from CN3")
cn3["GND"] += gnd; cn4["GND"] += gnd; cn4["+3V3"] += nuc3v3
cn3["D2"] += ev_up        # PA12: EXTI / timer capture for up events
cn3["D3"] += ev_dn        # PB0: EXTI / timer capture for down events
cn3["D6"] += fr_ctl       # PB1: fast-restore control
cn4["A3"] += th_up        # PA4 = DAC1_OUT1
cn4["A4"] += th_dn        # PA5 = DAC1_OUT2 (open SB16/SB18 on the Nucleo, see hardware/README.md)
cn4["A0"] += afe_mon      # PA0 = ADC (debug monitor only)

# test points
for net in (vref, ina_out, srv, afe_out, th_up, th_dn, ev_up, ev_dn):
    tp = mk("TestPoint", "TP", "TestPoint:TestPoint_Pad_D1.5mm", [(1, "1")], net.name, "test point")
    tp[1] += net

if __name__ == "__main__":
    generate_netlist(file_=str(HERE / "isoline_afe.net"))
    with open(HERE / "bom.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["part", "value", "footprint", "note"])
        for row in BOM:
            w.writerow(row)
    print(f"parts {len(BOM)}  -> {HERE / 'isoline_afe.net'}, {HERE / 'bom.csv'}")
