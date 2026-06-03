"""
receiver.py  —  QPSK Audio Receiver
=====================================
Usage (live mic):
    python receiver.py

    Set LIVE_RECORD = True  to record from microphone.
    Set LIVE_RECORD = False to process a saved WAV file (for testing).

Outputs: rx_bits.csv           (1 column, no header — submit this)
         rx_bits_reference.npy (for BER calc)

Pipeline:
    mic/WAV → Hilbert demod → LPF → RRC matched filter
            → ZC sync → min-variance timing → diff decode → CSV
"""

import numpy as np
import scipy.signal as sp
import sounddevice as sd
import soundfile as sf
import pandas as pd
import os, time

# ── Parameters (must match transmitter.py exactly) ─────────────
FS        = 44100
FC        = 5500
SPS       = 4
ROLL      = 0.35
RRC_TAPS  = 81
LPF_TAPS  = 81
ZC_LEN    = 127

LIVE_RECORD  = True          # True=mic, False=load INPUT_WAV
INPUT_WAV    = "tx_signal.wav"
RECORD_SECS  = 60            # seconds to record (set > transmission time)
OUTPUT_CSV   = "rx_bits.csv"
REF_NPY      = "tx_bits_reference.npy"   # if present, BER is computed

# ── Derived ────────────────────────────────────────────────────
RRC_DELAY      = RRC_TAPS // 2
LPF_DELAY      = LPF_TAPS  // 2
TOTAL_RX_DELAY = LPF_DELAY + 2*RRC_DELAY   # 40+41+41=122

# ── Filters ────────────────────────────────────────────────────
def make_rrc():
    beta,sps,ntaps=ROLL,SPS,RRC_TAPS
    t=np.arange(-ntaps//2,ntaps//2+1)/sps; h=np.zeros_like(t)
    for i,ti in enumerate(t):
        if abs(ti)<1e-12: h[i]=1-beta+4*beta/np.pi
        elif abs(abs(ti)-1/(4*beta))<1e-12:
            h[i]=(beta/np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*beta))+(1-2/np.pi)*np.cos(np.pi/(4*beta)))
        else: h[i]=(np.sin(np.pi*ti*(1-beta))+4*beta*ti*np.cos(np.pi*ti*(1+beta)))/(np.pi*ti*(1-(4*beta*ti)**2))
    h/=np.sqrt(np.sum(h**2)); return h

RRC = make_rrc()
LPF = sp.firwin(LPF_TAPS, min(1.2*(FS/SPS)/(FS/2), 0.99))

# ── Constellation & differential decoding ──────────────────────
REFS = np.exp(1j*np.array([1,3,5,7])*np.pi/4)
_DEC = {0:(0,0),1:(0,1),2:(1,1),3:(1,0)}

def diff_decode(syms):
    idx=np.argmin(np.abs(syms[:,None]-REFS[None,:]),axis=1); bits=[]
    for i in range(1,len(idx)): bits.extend(_DEC[int((idx[i]-idx[i-1])%4)])
    return np.array(bits,dtype=int)

# ── ZC passband reference ──────────────────────────────────────
def make_zc_passband():
    n=np.arange(ZC_LEN); zc=np.real(np.exp(-1j*np.pi*n*(n+1)/ZC_LEN))
    zc/=np.max(np.abs(zc)); up=np.zeros(ZC_LEN*SPS); up[::SPS]=zc
    bb=np.convolve(up,RRC); t=np.arange(len(bb))/FS
    pb=bb*np.cos(2*np.pi*FC*t); return (pb/np.max(np.abs(pb))).astype(float)

ZC_PASSBAND = make_zc_passband()
ZC_LEN_SAMP = len(ZC_PASSBAND)

# ── Receiver pipeline ──────────────────────────────────────────
def receive(rx, nbits):
    # Step 1: IQ demodulation — classic multiply + LPF
    t_full=np.arange(len(rx))/FS
    I_full=2*rx*np.cos(2*np.pi*FC*t_full)
    Q_full=-2*rx*np.sin(2*np.pi*FC*t_full)
    I_full=sp.lfilter(LPF,1,I_full); Q_full=sp.lfilter(LPF,1,Q_full)

    # Step 2: RRC matched filter
    I_mf=np.convolve(I_full,RRC); Q_mf=np.convolve(Q_full,RRC)
    iq_full=I_mf+1j*Q_mf

    # Step 3: ZC sync on raw passband signal
    corr_abs=np.abs(np.correlate(rx,ZC_PASSBAND,mode='valid'))
    peak=int(np.argmax(corr_abs))
    print(f"  ZC sync peak at sample {peak} ({peak/FS*1000:.1f} ms)")

    data_start_iq=peak+ZC_LEN_SAMP+TOTAL_RX_DELAY
    iq_data=iq_full[data_start_iq:]
    print(f"  Data starts at IQ sample {data_start_iq}")
    print(f"  Samples available: {len(iq_data):,}  needed: {nbits//2*SPS:,}")

    # Step 4: Timing — minimum variance picks correct symbol instant
    best_off,best_var=0,np.inf
    for off in range(SPS):
        seg=iq_data[off::SPS][:500]
        if len(seg)<50: continue
        v=np.var(np.abs(seg))
        if v<best_var: best_var,best_off=v,off
    print(f"  Best timing offset: {best_off}")

    # Step 5: Downsample — no skip needed with exact delay
    n_syms=nbits//2+1
    syms=iq_data[best_off::SPS][:n_syms]

    # Step 6: Differential decode
    return diff_decode(syms)[:nbits]

# ── Main ───────────────────────────────────────────────────────
if __name__=="__main__":
    print("="*55); print("  QPSK AUDIO RECEIVER"); print("="*55)
    print(f"  FC={FC}Hz  SPS={SPS}  Rb={2*FS/SPS:.0f}bps")

    t_start=time.time()

    # Acquire signal
    if LIVE_RECORD:
        print(f"\n[1/4] Recording {RECORD_SECS}s from microphone ...")
        print("      → Make sure transmitter is playing NOW ←")
        raw=sd.rec(int(RECORD_SECS*FS),samplerate=FS,channels=1,dtype='float32')
        sd.wait()
        rx=raw[:,0].astype(float)
        sf.write("rx_captured.wav",rx.astype(np.float32),FS)
        print(f"      Captured audio saved → rx_captured.wav")
    else:
        print(f"\n[1/4] Loading '{INPUT_WAV}' ...")
        rx,fs_wav=sf.read(INPUT_WAV)
        if rx.ndim>1: rx=rx[:,0]
        assert fs_wav==FS, f"WAV sample rate {fs_wav} != {FS}"
        rx=rx.astype(float)
        print(f"      Loaded {len(rx)/FS:.2f}s of audio")

    # How many bits to decode
    if os.path.exists(REF_NPY):
        nbits=len(np.load(REF_NPY))
        print(f"\n  Reference found → decoding {nbits:,} bits")
    else:
        nbits=1_000_000
        print(f"\n  No reference found → decoding {nbits:,} bits (default)")

    # Decode
    print(f"\n[2/4] Decoding ...")
    rx_bits=receive(rx,nbits)

    # Save CSV
    print(f"\n[3/4] Saving '{OUTPUT_CSV}' ...")
    pd.DataFrame(rx_bits).to_csv(OUTPUT_CSV,index=False,header=False)
    print(f"      {len(rx_bits):,} bits saved.")

    T=time.time()-t_start

    # BER (if reference available)
    print(f"\n[4/4] Results")
    print("-"*45)
    if os.path.exists(REF_NPY):
        ref=np.load(REF_NPY)
        L=min(len(ref),len(rx_bits))
        n_err=int(np.sum(ref[:L]!=rx_bits[:L]))
        ber=n_err/nbits
        data_rate=nbits/T
        print(f"  Bits received : {nbits:,}")
        print(f"  Bit errors    : {n_err:,}")
        print(f"  BER           : {ber:.6f}")
        print(f"  Time T        : {T:.1f} s")
        print(f"  Data rate     : {data_rate:.0f} bps ({data_rate/1000:.2f} kbps)")
        print()
        if ber<0.00001 and T<150: print("  → Target: 30 marks (BER<1e-5, T<150s) ✓")
        elif ber<0.001 and T<300: print("  → Target: 25 marks (BER<1e-3, T<300s) ✓")
        elif ber<0.01:            print("  → Target: 20 marks (BER<0.01) ✓")
        else:                     print("  → BER too high — check volume and positioning")
    else:
        print(f"  No reference for BER. Output → {OUTPUT_CSV}")
        print(f"  Time T: {T:.1f} s")
    print("="*55)
