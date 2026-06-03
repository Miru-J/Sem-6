
import numpy as np
import scipy.signal as sp

# MINIMUM VARIANCE clearly identifies off=0 as correct.
# Now build the final corrected receiver and verify full BER curve.

FS = 44100;
FC = 5500;
SPS = 4;
ROLL = 0.35;
RRC_TAPS = 81;
LPF_TAPS = 81;
ZC_LEN = 127;
N_BITS = 100000


def make_rrc():
    beta, sps, ntaps = ROLL, SPS, RRC_TAPS
    t = np.arange(-ntaps // 2, ntaps // 2 + 1) / sps;
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-12:
            h[i] = (beta / np.sqrt(2)) * (
                        (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / (
                        np.pi * ti * (1 - (4 * beta * ti) ** 2))
    h /= np.sqrt(np.sum(h ** 2));
    return h


RRC = make_rrc();
LPF = sp.firwin(LPF_TAPS, min(1.2 * (FS / SPS) / (FS / 2), 0.99))
REFS = np.exp(1j * np.array([1, 3, 5, 7]) * np.pi / 4)
_ENC = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3};
_DEC = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
RRC_DELAY = len(RRC) // 2;
LPF_DELAY = LPF_TAPS // 2
TOTAL_RX_DELAY = LPF_DELAY + 2 * RRC_DELAY  # 122: LPF(40)+RRC_TX(41)+RRC_RX(41)


def diff_encode(bits):
    if len(bits) % 2: bits = np.append(bits, 0)
    pairs = bits.reshape(-1, 2);
    idx = np.zeros(len(pairs) + 1, dtype=int)
    for i, p in enumerate(pairs): idx[i + 1] = (idx[i] + _ENC[tuple(p)]) % 4
    return REFS[idx]


def diff_decode(syms):
    idx = np.argmin(np.abs(syms[:, None] - REFS[None, :]), axis=1);
    bits = []
    for i in range(1, len(idx)): step = int((idx[i] - idx[i - 1]) % 4); bits.extend(_DEC[step])
    return np.array(bits, dtype=int)


def make_zc_passband():
    n = np.arange(ZC_LEN);
    zc = np.real(np.exp(-1j * np.pi * n * (n + 1) / ZC_LEN))
    zc /= np.max(np.abs(zc));
    up = np.zeros(ZC_LEN * SPS);
    up[::SPS] = zc
    bb = np.convolve(up, RRC);
    t = np.arange(len(bb)) / FS
    pb = bb * np.cos(2 * np.pi * FC * t);
    return pb / np.max(np.abs(pb))


ZC_PASSBAND = make_zc_passband();
ZC_LEN_SAMP = len(ZC_PASSBAND)


def transmitter(bits):
    syms = diff_encode(bits)
    up = np.zeros(len(syms) * SPS, dtype=complex);
    up[::SPS] = syms
    I_bb = np.convolve(np.real(up), RRC);
    Q_bb = np.convolve(np.imag(up), RRC)
    t_d = np.arange(len(I_bb)) / FS
    tx = I_bb * np.cos(2 * np.pi * FC * t_d) - Q_bb * np.sin(2 * np.pi * FC * t_d)
    guard = np.zeros(int(0.02 * FS))
    frame = np.concatenate([guard, ZC_PASSBAND, tx, guard])
    return (frame / np.max(np.abs(frame))).astype(float)


import matplotlib.pyplot as plt

def receiver(rx, nbits, visualize=False):
    t_full = np.arange(len(rx)) / FS

    # -------------------------------------------------
    # Downconversion
    # -------------------------------------------------
    I_full = 2 * rx * np.cos(2 * np.pi * FC * t_full)
    Q_full = -2 * rx * np.sin(2 * np.pi * FC * t_full)

    # LPF
    I_lpf = sp.lfilter(LPF, 1, I_full)
    Q_lpf = sp.lfilter(LPF, 1, Q_full)

    # Matched filter
    I_mf = np.convolve(I_lpf, RRC)
    Q_mf = np.convolve(Q_lpf, RRC)

    iq_full = I_mf + 1j * Q_mf

    # -------------------------------------------------
    # Frame synchronization
    # -------------------------------------------------
    corr = np.correlate(rx, ZC_PASSBAND, mode='valid')
    corr_abs = np.abs(corr)

    peak = int(np.argmax(corr_abs))

    data_start_iq = peak + ZC_LEN_SAMP + TOTAL_RX_DELAY
    iq_data = iq_full[data_start_iq:]

    # -------------------------------------------------
    # Timing synchronization
    # -------------------------------------------------
    vars_metric = []

    best_off = 0
    best_var = np.inf

    for off in range(SPS):
        seg = iq_data[off::SPS][:500]

        if len(seg) < 50:
            vars_metric.append(np.nan)
            continue

        v = np.var(np.abs(seg))
        vars_metric.append(v)

        if v < best_var:
            best_var = v
            best_off = off

    # -------------------------------------------------
    # Symbol extraction
    # -------------------------------------------------
    n_syms = nbits // 2 + 1

    syms_raw = iq_data[:400]
    syms_sync = iq_data[best_off::SPS][:n_syms]

    bits_out = diff_decode(syms_sync)[:nbits]

    # -------------------------------------------------
    # Visualization
    # -------------------------------------------------
    if visualize:

        fig = plt.figure(figsize=(18, 12))

        # =============================================
        # 1. Received passband signal
        # =============================================
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(rx[:4000])
        ax1.set_title("Received Passband Signal")
        ax1.grid(True)

        # =============================================
        # 2. ZC Correlation
        # =============================================
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(corr_abs)
        ax2.axvline(peak, color='r', linestyle='--')
        ax2.set_title("ZC Correlation Peak")
        ax2.grid(True)

        # =============================================
        # 3. Matched filter outputs
        # =============================================
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(np.real(iq_full[:2000]), label='I')
        ax3.plot(np.imag(iq_full[:2000]), label='Q')
        ax3.set_title("Matched Filter Output")
        ax3.legend()
        ax3.grid(True)

        # =============================================
        # 4. Timing metric
        # =============================================
        ax4 = plt.subplot(3, 3, 4)
        ax4.stem(range(SPS), vars_metric)
        ax4.set_title("Timing Synchronization Metric")
        ax4.set_xlabel("Offset")
        ax4.set_ylabel("Variance")
        ax4.grid(True)

        # =============================================
        # 5. Constellation before sync
        # =============================================
        ax5 = plt.subplot(3, 3, 5)
        ax5.scatter(
            np.real(syms_raw[:1000]),
            np.imag(syms_raw[:1000]),
            s=5
        )
        ax5.set_title("Constellation Before Timing Sync")
        ax5.set_xlabel("In-Phase")
        ax5.set_ylabel("Quadrature")
        ax5.grid(True)
        ax5.axis('equal')

        # =============================================
        # 6. Final synchronized constellation
        # =============================================
        ax6 = plt.subplot(3, 3, 6)
        ax6.scatter(
            np.real(syms_sync[:2000]),
            np.imag(syms_sync[:2000]),
            s=5
        )

        # Ideal points
        ax6.scatter(
            np.real(REFS),
            np.imag(REFS),
            marker='x',
            s=120
        )

        ax6.set_title("Final QPSK Constellation")
        ax6.set_xlabel("In-Phase")
        ax6.set_ylabel("Quadrature")
        ax6.grid(True)
        ax6.axis('equal')

        # =============================================
        # 7. Eye diagram (I branch)
        # =============================================
        ax7 = plt.subplot(3, 3, 7)

        eye = np.real(iq_data[:3000])

        for i in range(0, len(eye) - 2*SPS, SPS):
            ax7.plot(eye[i:i+2*SPS], alpha=0.3)

        ax7.set_title("Eye Diagram (I Branch)")
        ax7.grid(True)

        # =============================================
        # 8. Spectrum
        # =============================================
        ax8 = plt.subplot(3, 3, 8)

        f, Pxx = sp.welch(rx, FS, nperseg=1024)

        ax8.semilogy(f, Pxx)
        ax8.set_title("Received Signal Spectrum")
        ax8.set_xlabel("Frequency (Hz)")
        ax8.grid(True)

        # =============================================
        # 9. IQ trajectory
        # =============================================
        ax9 = plt.subplot(3, 3, 9)

        traj = syms_sync[:200]

        ax9.plot(np.real(traj), np.imag(traj), alpha=0.7)
        ax9.scatter(np.real(traj), np.imag(traj), s=10)

        ax9.set_title("IQ Trajectory")
        ax9.grid(True)
        ax9.axis('equal')

        plt.tight_layout()
        plt.show()

    return bits_out

def awgn(x, snr_db):
    p = np.mean(x ** 2);
    return x + np.sqrt(p / 10 ** (snr_db / 10)) * np.random.randn(len(x))


# Zero noise test
np.random.seed(42)
bits = np.random.randint(0, 2, 1000)
tx = transmitter(bits);
rx = tx.copy()
rb = receiver(rx, len(bits))
print(
    f"Zero noise BER: {np.mean(bits != rb[:len(bits)]):.6f}  ({'PASS' if np.mean(bits != rb[:len(bits)]) == 0 else 'FAIL'})")

# Full BER curve
print(f"\n{'SNR':>6}  {'BER':>12}  Status")
print("-" * 35)
for snr in [0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30]:
    np.random.seed(snr)
    bits = np.random.randint(0, 2, N_BITS)
    tx = transmitter(bits);
    rx = awgn(tx, snr)
    rb = receiver(rx, N_BITS)
    L = min(len(bits), len(rb))
    ber = float(np.mean(bits[:L] != rb[:L]))
    s = "✓<1e-5" if ber < 1e-5 else "✓<1e-3" if ber < 1e-3 else "✓<1e-2" if ber < 1e-2 else "✗"
    print(f"{snr:>6}  {ber:>12.4e}  {s}")
# =====================================================
# 30 dB Visualization
# =====================================================

np.random.seed(30)

bits = np.random.randint(0, 2, N_BITS)

tx = transmitter(bits)

rx = awgn(tx, 30)

rb = receiver(rx, N_BITS, visualize=True)

ber = np.mean(bits != rb[:len(bits)])

print(f"\n30 dB BER = {ber:.6e}")