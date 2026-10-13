"""Measure a soundtrack WAV against the animation's cue sheet.

Usage: python3 analyze.py track.wav [--png out.png] [--json out.json]

Reports: format, duration, sample/true peak, integrated loudness (LUFS),
clipping, DC offset, stereo correlation, band balance, long silences,
and for every cue how strong the audio onset is right at that moment.

Onset strength = peak of the spectral-flux envelope within [t-40ms, t+60ms],
divided by the track's median flux. A ratio >= 4 reads as a clear hit,
>= 2.5 as an audible accent. 'offset_ms' is where that peak sits relative
to the cue (negative = early).
"""
import json
import sys
import numpy as np
from scipy.io import wavfile
from scipy import signal
import pyloudnorm as pyln

HERE = __file__.rsplit('/', 1)[0] if '/' in __file__ else '.'


def load(path):
    sr, x = wavfile.read(path)
    if x.dtype == np.int16:
        x = x.astype(np.float64) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float64) / 2147483648.0
    else:
        x = x.astype(np.float64)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    return sr, x


def onset_env(mono, sr, hop=256, nfft=1024):
    f, t, Z = signal.stft(mono, sr, nperseg=nfft, noverlap=nfft - hop, boundary=None, padded=False)
    mag = np.log1p(np.abs(Z) * 100)
    flux = np.maximum(0, np.diff(mag, axis=1)).sum(axis=0)
    times = t[1:]
    return times, flux


def main():
    args = sys.argv[1:]
    path = args[0]
    png = args[args.index('--png') + 1] if '--png' in args else None
    jout = args[args.index('--json') + 1] if '--json' in args else None
    cues = json.load(open(HERE + '/cues.json'))
    sr, x = load(path)
    mono = x.mean(axis=1)
    dur = len(x) / sr
    rep = {'file': path, 'sample_rate': sr, 'channels': 2, 'duration_s': round(dur, 3),
           'target_duration_s': cues['duration']}

    peak = np.max(np.abs(x))
    rep['sample_peak_dbfs'] = round(20 * np.log10(peak + 1e-12), 2)
    up = signal.resample_poly(x, 4, 1, axis=0)
    rep['true_peak_dbtp'] = round(20 * np.log10(np.max(np.abs(up)) + 1e-12), 2)
    rep['clipped_samples'] = int(np.sum(np.abs(x) >= 0.999))
    rep['dc_offset'] = [round(float(x[:, 0].mean()), 5), round(float(x[:, 1].mean()), 5)]
    meter = pyln.Meter(sr)
    rep['integrated_lufs'] = round(float(meter.integrated_loudness(x)), 2)
    l, r = x[:, 0], x[:, 1]
    rep['stereo_correlation'] = round(float(np.corrcoef(l, r)[0, 1]), 3) if np.std(l) > 0 and np.std(r) > 0 else 1.0

    spec = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(len(mono), 1 / sr)
    bands = {'sub_<60': (0, 60), 'low_60-250': (60, 250), 'mid_250-2k': (250, 2000),
             'high_2k-8k': (2000, 8000), 'air_>8k': (8000, sr / 2)}
    tot = spec.sum() + 1e-12
    rep['band_energy_pct'] = {k: round(100 * spec[(freqs >= a) & (freqs < b)].sum() / tot, 1) for k, (a, b) in bands.items()}

    # short-term loudness curve (400 ms windows, 100 ms hop) and long silences
    win, hop = int(0.4 * sr), int(0.1 * sr)
    rms_t, rms_db = [], []
    for i in range(0, len(mono) - win, hop):
        seg = mono[i:i + win]
        rms_t.append((i + win / 2) / sr)
        rms_db.append(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-9))
    rms_t, rms_db = np.array(rms_t), np.array(rms_db)
    quiet = rms_db < -45
    silences, start = [], None
    for tt, q in zip(rms_t, quiet):
        if q and start is None:
            start = tt
        if not q and start is not None:
            if tt - start > 0.5:
                silences.append([round(start, 2), round(tt, 2)])
            start = None
    if start is not None and rms_t[-1] - start > 0.5:
        silences.append([round(start, 2), round(float(rms_t[-1]), 2)])
    rep['silences_over_0.5s'] = silences
    rep['loudness_by_section'] = [
        {'name': s['name'], 't': s['t'], 'end': s['end'],
         'rms_db': round(float(np.mean(rms_db[(rms_t >= s['t']) & (rms_t < s['end'])])), 1) if np.any((rms_t >= s['t']) & (rms_t < s['end'])) else None}
        for s in cues['sections']]

    ot, flux = onset_env(mono, sr)
    med = np.median(flux) + 1e-9
    cue_rows = []
    for c in cues['cues']:
        t, kind = c['t'], c['kind']
        row = {'t': t, 'kind': kind, 'label': c['label']}
        if kind in ('hit', 'downbeat', 'accent'):
            m = (ot >= t - 0.04) & (ot <= t + 0.06)
            if np.any(m):
                i = np.argmax(np.where(m, flux, -1))
                row['onset_ratio'] = round(float(flux[i] / med), 2)
                row['offset_ms'] = round(float((ot[i] - t) * 1000), 1)
                need = 4.0 if kind in ('hit', 'downbeat') else 2.5
                row['ok'] = bool(row['onset_ratio'] >= need)
        elif kind == 'riser_end':
            # high-frequency energy should rise across the 0.8 s before the cue
            seg = mono[int(max(0, t - 0.8) * sr):int(t * sr)]
            if len(seg) > 1000:
                hp = signal.sosfilt(signal.butter(4, 2000, 'hp', fs=sr, output='sos'), seg)
                k = len(hp) // 4
                e = [20 * np.log10(np.sqrt(np.mean(hp[j * k:(j + 1) * k] ** 2)) + 1e-9) for j in range(4)]
                row['hf_db_quarters'] = [round(v, 1) for v in e]
                row['ok'] = bool(e[3] - e[0] >= 3)
        elif kind == 'swell':
            m1 = (rms_t >= t) & (rms_t < t + 0.6)
            m2 = (rms_t >= c['end'] - 0.6) & (rms_t < c['end'])
            row['rms_start_db'] = round(float(np.mean(rms_db[m1])), 1)
            row['rms_end_db'] = round(float(np.mean(rms_db[m2])), 1)
            row['ok'] = bool(row['rms_end_db'] - row['rms_start_db'] >= 2)
        elif kind == 'stop':
            before = flux[(ot >= t - 1.0) & (ot < t - 0.05)]
            after = flux[(ot >= t + 0.05) & (ot < t + 1.0)]
            row['flux_before'] = round(float(before.mean() / med), 2)
            row['flux_after'] = round(float(after.mean() / med), 2)
            row['ok'] = bool(abs(row['flux_before'] - row['flux_after']) / max(row['flux_before'], 1e-6) > 0.3)
        elif kind == 'fade_end':
            tail = mono[int((t - 0.15) * sr):int(t * sr)]
            row['last_150ms_db'] = round(float(20 * np.log10(np.sqrt(np.mean(tail ** 2)) + 1e-9)), 1) if len(tail) else None
            row['ok'] = bool(row['last_150ms_db'] is not None and row['last_150ms_db'] < -40)
        cue_rows.append(row)
    rep['cues'] = cue_rows
    checked = [r for r in cue_rows if 'ok' in r]
    rep['cue_pass'] = f"{sum(r['ok'] for r in checked)}/{len(checked)}"
    rep['cue_fail'] = [f"{r['t']} {r['kind']} {r['label']}" for r in checked if not r['ok']]

    problems = []
    if abs(dur - cues['duration']) > 0.05:
        problems.append(f'duration {dur:.2f}s, expected {cues["duration"]}s')
    if rep['true_peak_dbtp'] > -1.0:
        problems.append(f'true peak {rep["true_peak_dbtp"]} dBTP > -1.0')
    if not (-16.5 <= rep['integrated_lufs'] <= -12.5):
        problems.append(f'loudness {rep["integrated_lufs"]} LUFS outside -16.5..-12.5')
    if rep['clipped_samples']:
        problems.append(f'{rep["clipped_samples"]} clipped samples')
    if rep['stereo_correlation'] < 0.2:
        problems.append(f'stereo correlation {rep["stereo_correlation"]} (mono-compatibility risk)')
    if sr != cues['sample_rate']:
        problems.append(f'sample rate {sr}, expected {cues["sample_rate"]}')
    rep['technical_problems'] = problems

    if png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(26, 11), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        f, tt, S = signal.spectrogram(mono, sr, nperseg=2048, noverlap=1536)
        ax[0].pcolormesh(tt, f, 10 * np.log10(S + 1e-12), shading='auto', cmap='magma', vmin=-110, vmax=-30)
        ax[0].set_yscale('symlog', linthresh=200)
        ax[0].set_ylim(30, 16000)
        ax[0].set_ylabel('Hz')
        ax[1].plot(ot, flux / med, lw=.6, color='k')
        ax[1].set_ylabel('onset/median')
        ax[1].set_ylim(0, max(12, float(np.percentile(flux / med, 99.5))))
        ax[2].plot(rms_t, rms_db, color='tab:blue')
        ax[2].set_ylabel('RMS dB')
        ax[2].set_ylim(-60, 0)
        colors = {'hit': 'red', 'downbeat': 'lime', 'accent': 'cyan', 'riser_end': 'yellow', 'swell': 'orange', 'stop': 'magenta', 'fade_end': 'white'}
        for r in cue_rows:
            for a in ax:
                a.axvline(r['t'], color=colors[r['kind']], lw=.8 if r['kind'] == 'accent' else 1.4, alpha=.8,
                          ls='-' if r.get('ok', True) else '--')
        for s in cues['sections']:
            ax[0].text(s['t'] + .05, 13000, s['name'][:22], color='w', fontsize=7, va='top')
        ax[2].set_xlabel('seconds  (red=hit, lime=downbeat, cyan=accent, yellow=riser end, dashed=cue failed)')
        ax[2].set_xticks(np.arange(0, cues['duration'] + 1, 1))
        fig.tight_layout()
        fig.savefig(png, dpi=60)
        rep['png'] = png

    out = json.dumps(rep, ensure_ascii=False, indent=1)
    if jout:
        open(jout, 'w').write(out)
    print(json.dumps({k: rep[k] for k in ('duration_s', 'integrated_lufs', 'true_peak_dbtp', 'clipped_samples', 'stereo_correlation', 'band_energy_pct', 'silences_over_0.5s', 'cue_pass', 'cue_fail', 'technical_problems')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
