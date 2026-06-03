"""
transmitter.py  —  QPSK Audio Transmitter
==========================================
Usage:
    python transmitter.py

Reads  : bits_to_send.csv  (1 column, no header, 1 000 000 bits)
Outputs: tx_signal.wav     (play through speakers toward receiver mic)
         tx_bits_reference.npy  (saved for BER calculation)
"""

import numpy as np
import scipy.signal as sp
import soundfile as sf
import pandas as pd
import os, time

FS       = 44100
FC       = 5500
SPS      = 4
ROLL     = 0.35
RRC_TAPS = 81
ZC_LEN   = 127

INPUT_CSV  = "bits_to_send.csv"
OUTPUT_WAV = "tx_signal.wav"
REF_NPY    = "tx_bits_reference.npy"

def make_rrc():
    beta,sps,ntaps = ROLL,SPS,RRC_TAPS
    t = np.arange(-ntaps//2,ntaps//2+1)/sps; h=np.zeros_like(t)
    for i,ti in enumerate(t):
        if abs(ti)<1e-12: h[i]=1-beta+4*beta/np.pi
        elif abs(abs(ti)-1/(4*beta))<1e-12:
            h[i]=(beta/np.sqrt(2))*((1+2/np.pi)*np.sin(np.pi/(4*beta))+(1-2/np.pi)*np.cos(np.pi/(4*beta)))
        else: h[i]=(np.sin(np.pi*ti*(1-beta))+4*beta*ti*np.cos(np.pi*ti*(1+beta)))/(np.pi*ti*(1-(4*beta*ti)**2))
    h/=np.sqrt(np.sum(h**2)); return h

RRC  = make_rrc()
REFS = np.exp(1j*np.array([1,3,5,7])*np.pi/4)
_ENC = {(0,0):0,(0,1):1,(1,1):2,(1,0):3}

def diff_encode(bits):
    if len(bits)%2: bits=np.append(bits,0)
    pairs=bits.reshape(-1,2); idx=np.zeros(len(pairs)+1,dtype=int)
    for i,p in enumerate(pairs): idx[i+1]=(idx[i]+_ENC[tuple(p)])%4
    return REFS[idx]

def make_zc_passband():
    n=np.arange(ZC_LEN); zc=np.real(np.exp(-1j*np.pi*n*(n+1)/ZC_LEN))
    zc/=np.max(np.abs(zc)); up=np.zeros(ZC_LEN*SPS); up[::SPS]=zc
    bb=np.convolve(up,RRC); t=np.arange(len(bb))/FS
    pb=bb*np.cos(2*np.pi*FC*t); return (pb/np.max(np.abs(pb))).astype(float)

ZC_PASSBAND = make_zc_passband()

def transmit(bits):
    syms=diff_encode(bits)
    up=np.zeros(len(syms)*SPS,dtype=complex); up[::SPS]=syms
    I_bb=np.convolve(np.real(up),RRC); Q_bb=np.convolve(np.imag(up),RRC)
    t_d=np.arange(len(I_bb))/FS
    tx_data=I_bb*np.cos(2*np.pi*FC*t_d)-Q_bb*np.sin(2*np.pi*FC*t_d)
    guard=np.zeros(int(0.05*FS))
    frame=np.concatenate([guard,ZC_PASSBAND,tx_data,guard])
    return (frame/(np.max(np.abs(frame))+1e-12)).astype(float)

if __name__=="__main__":
    print("="*55); print("  QPSK AUDIO TRANSMITTER"); print("="*55)
    print(f"  FC={FC}Hz  SPS={SPS}  Rb={2*FS/SPS:.0f}bps  ({2*FS/SPS/1000:.2f}kbps)")

    if not os.path.exists(INPUT_CSV):
        print(f"\nERROR: '{INPUT_CSV}' not found. Place it here and re-run."); exit(1)

    print(f"\n[1/3] Loading '{INPUT_CSV}' ...")
    bits=pd.read_csv(INPUT_CSV,header=None).iloc[:,0].values.astype(int)
    print(f"      {len(bits):,} bits loaded.")
    np.save(REF_NPY,bits); print(f"      Reference saved → {REF_NPY}")

    print(f"\n[2/3] Building QPSK frame ...")
    t0=time.time(); frame=transmit(bits); dur=len(frame)/FS
    print(f"      Duration={dur:.2f}s  Samples={len(frame):,}")

    print(f"\n[3/3] Writing '{OUTPUT_WAV}' ...")
    sf.write(OUTPUT_WAV,frame,FS)
    print(f"      Done in {time.time()-t0:.1f}s")
    print(f"\n  → Play '{OUTPUT_WAV}' through speakers at 1m from receiver mic.")
    print(f"  → Keep volume ≤ 50 dB.")
    print(f"  → Estimated transmission time: {len(bits)/(2*FS/SPS):.1f} s")
    print("="*55)
