"""Log-mel of a synthetic nonlinguistic signal: illustration, no corpus data."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.signal import butter, sosfilt

P = Path(__file__).resolve().parent

def example():
    sr, duration, n_fft, hop, n_mels = 16000, 1.6, 400, 160, 80
    rng = np.random.default_rng(731)
    t = np.arange(int(sr * duration)) / sr
    f0 = 135 + 23*np.sin(2*np.pi*1.05*t) + 7*np.sin(2*np.pi*4.2*t)
    phase = np.cumsum(f0)*(2*np.pi/sr)
    y = np.zeros_like(t)
    for center, formants in [(.18,[650,1150,2450]),(.56,[350,2000,2800]),
                             (.99,[480,1050,2350]),(1.38,[330,2450,3150])]:
        envelope = np.exp(-.5*((t-center)/.105)**4)
        sound = np.zeros_like(t)
        for k in range(1,58):
            frequency = k*f0
            response = .015+sum(np.exp(-.5*((frequency-fc)/bw)**2)
                                for fc,bw in zip(formants,[85,140,200]))
            sound += response*np.sin(k*phase+rng.uniform(0,.2))/k**.8
        y += sound*envelope
    noise = sosfilt(butter(3,2400,'highpass',fs=sr,output='sos'),rng.normal(size=len(t)))
    y += .025*noise*(np.exp(-((t-.38)/.06)**2)+.8*np.exp(-((t-.80)/.045)**2)
                     +.5*np.exp(-((t-1.19)/.045)**2))
    y += .0002*rng.normal(size=len(t)); y /= np.max(np.abs(y))
    frames = np.lib.stride_tricks.sliding_window_view(np.pad(y,n_fft//2,mode='reflect'),n_fft)[::hop][:-1]
    power = np.abs(np.fft.rfft(frames*np.hanning(n_fft+1)[:-1],axis=1))**2
    def hz_to_mel(x):
        x=np.asarray(x,dtype=float)
        return np.where(x<1000,x/(200/3),15+np.log(np.maximum(x,1)/1000)/(np.log(6.4)/27))
    def mel_to_hz(x):
        x=np.asarray(x,dtype=float)
        return np.where(x<15,x*(200/3),1000*np.exp((x-15)*np.log(6.4)/27))
    edges=mel_to_hz(np.linspace(0,hz_to_mel(sr/2),n_mels+2))
    freqs=np.fft.rfftfreq(n_fft,1/sr)
    filters=np.maximum(0,np.minimum((freqs[None,:]-edges[:-2,None])/np.diff(edges)[:-1,None],
                                   (edges[2:,None]-freqs[None,:])/np.diff(edges)[1:,None]))
    filters *= (2/(edges[2:]-edges[:-2]))[:,None]
    log_mel=np.log10(np.maximum(filters@power.T,1e-10))
    log_mel=np.maximum(log_mel,log_mel.max()-8)
    assert log_mel.shape==(80,160)
    identity='F7_AUDIO_BLOCKS_V1|synthetic|figure1'
    seed=int(hashlib.sha256(identity.encode()).hexdigest()[:16],16)
    order=np.random.default_rng(seed).permutation(8)
    if np.array_equal(order,np.arange(8)):order=np.roll(order,1)
    permuted=np.concatenate([log_mel[:,i*20:(i+1)*20] for i in order],axis=1)
    np.savez_compressed(P/'synthetic_logmel.npz',clean=log_mel,permuted=permuted,waveform=y,sample_rate=sr,block_order=order)
    palette=['000004','1B0C41','4F127B','812581','B53679','E55063','FB8761','FDC98C','FCFDBF']
    stops=np.array([[int(c[i:i+2],16) for i in (0,2,4)] for c in palette])
    z=(log_mel-(log_mel.max()-8))/8
    rgb=np.round(np.stack([np.interp(z,np.linspace(0,1,len(stops)),stops[:,k]) for k in range(3)],axis=-1)).astype(np.uint8)
    rgb_perm=np.concatenate([rgb[:,i*20:(i+1)*20] for i in order],axis=1)
    meta={'kind':'Actual log-mel transform of a synthetic nonlinguistic signal; illustration only',
          'corpus_audio_used':False,'ASR_inference_or_evaluation':False,'sample_rate_hz':sr,
          'duration_seconds':duration,'n_fft':n_fft,'hop_samples':hop,'n_mels':n_mels,'frames':160,
          'mel_filter':'Slaney scale and area normalization','window':'Periodic Hann; reflect-centered STFT; last frame removed',
          'display':'Purple-to-gold sequential palette; identical fixed eight-decade log-power range for both views',
          'block_frames':20,'permutation_identity':identity,'permuted_order_zero_based':order.tolist(),
          'exact_block_reuse':all(np.array_equal(permuted[:,j*20:(j+1)*20],log_mel[:,i*20:(i+1)*20]) for j,i in enumerate(order)),
          'array_file_sha256':hashlib.sha256((P/'synthetic_logmel.npz').read_bytes()).hexdigest()}
    (P/'synthetic_logmel_provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    return y,rgb,rgb_perm,order.tolist(),meta

if __name__=='__main__':
    wave,clean,permuted,order,meta=example()
    (P/'synthetic_logmel_render.json').write_text(json.dumps({
        'waveform_display':wave[::100].tolist(),'clean_rgb':clean.tolist(),
        'permuted_rgb':permuted.tolist(),'order':order,'provenance':meta},separators=(',',':'))+'\n')
