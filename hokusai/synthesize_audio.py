import math
import struct
import random
import os

def generate_wave_sound(physics_log_path, output_wav_path, sample_rate=44100):
    metrics = []
    if not os.path.exists(physics_log_path):
        print(f"Error: {physics_log_path} not found.")
        return
        
    with open(physics_log_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                metrics.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), int(parts[4])))
                
    if not metrics:
        print("Error: No physical metrics logged.")
        return
        
    total_frames = len(metrics)
    fps = 30
    sim_duration = total_frames / fps
    total_samples = int(sample_rate * sim_duration)
    
    # -------------------------------------------------------------------------
    # STATIC OPEN-BAND LPF SYSTEM
    # -------------------------------------------------------------------------
    class StaticOnePoleLPF:
        def __init__(self, fc, fs):
            self.y1 = 0.0
            self.alpha = math.exp(-2.0 * math.pi * fc / fs)
        def process(self, x):
            y = (1.0 - self.alpha) * x + self.alpha * self.y1
            self.y1 = y
            return y

    # --- CALIBRATED OUTDOOR SPATIAL REVERBERATION ---
    # The "Karaoke Hall / Echo Chamber" effect was caused by:
    # 1. Reverb mix ratio being too high (50% Wet).
    # 2. Feedback parameters (0.94) holding the sound for too long.
    # 3. Macro delay sizes (up to 150ms) causing distinct artificial slapbacks.
    # To fix this and restore natural outdoor cliff reflection:
    # 1. WET mix is reduced to 15% (85% Dry / 15% Wet) to serve as a subtle acoustic cushion.
    # 2. Feedback parameters are lowered to 0.65 - 0.78 to target a tight 0.9s decay tail.
    # 3. Delay times are packed closer (53ms - 89ms) for a dense, diffuse, non-tonal space presence.
    # -------------------------------------------------------------------------
    class PrimeDelayLine:
        def __init__(self, delay_ms, sample_rate):
            self.size = int(delay_ms * sample_rate / 1000.0)
            self.buf = [0.0] * self.size
            self.idx = 0
        def process(self, x, feedback):
            tap = self.buf[self.idx]
            w = x + feedback * tap
            self.buf[self.idx] = w
            self.idx = (self.idx + 1) % self.size
            return tap

    # Medium-tight prime delay times for dense outdoor diffusion
    reverb_l1 = PrimeDelayLine(53.3, sample_rate)
    reverb_l2 = PrimeDelayLine(67.7, sample_rate)
    reverb_l3 = PrimeDelayLine(79.3, sample_rate)
    reverb_l4 = PrimeDelayLine(89.7, sample_rate)
    
    reverb_r1 = PrimeDelayLine(54.1, sample_rate)
    reverb_r2 = PrimeDelayLine(66.1, sample_rate)
    reverb_r3 = PrimeDelayLine(81.1, sample_rate)
    reverb_r4 = PrimeDelayLine(88.3, sample_rate)

    # LPF constants
    wind_lp_l = StaticOnePoleLPF(450.0, sample_rate)
    wind_lp_r = StaticOnePoleLPF(450.0, sample_rate)
    
    buff_lp_l = StaticOnePoleLPF(20.0, sample_rate)
    buff_lp_r = StaticOnePoleLPF(20.0, sample_rate)
    
    crash_lp_l = StaticOnePoleLPF(850.0, sample_rate)
    crash_lp_r = StaticOnePoleLPF(850.0, sample_rate)
    
    fizz_lp_l = StaticOnePoleLPF(16500.0, sample_rate)
    fizz_lp_r = StaticOnePoleLPF(16500.0, sample_rate)

    # -------------------------------------------------------------------------
    # Pure Natural Noise Generators (Zero White Noise, Zero Sine waves)
    # -------------------------------------------------------------------------
    pink_rows = [0.0] * 12
    pink_sum = 0.0
    def next_pink():
        nonlocal pink_sum
        r = random.randint(0, 11)
        pink_sum -= pink_rows[r]
        pink_rows[r] = random.uniform(-1.0, 1.0)
        pink_sum += pink_rows[r]
        return (pink_sum + random.uniform(-1.0, 1.0)) * 0.0833

    brown_l, brown_r = 0.0, 0.0
    def next_brown_stereo():
        nonlocal brown_l, brown_r
        brown_l += random.uniform(-0.04, 0.04)
        brown_r += random.uniform(-0.04, 0.04)
        brown_l *= 0.993
        brown_r *= 0.993
        return brown_l * 0.35, brown_r * 0.35

    out_samples = []
    
    # Envelopes
    crash_env = 0.0
    fizz_env = 0.0
    
    print("Synthesizing organic wave audio: adjusting reverb to dense outdoor cliff mix (85% dry / 15% wet)...")
    
    for n in range(total_samples):
        t_sec = n / sample_rate
        f_idx = int(t_sec * fps)
        f_idx = min(f_idx, total_frames - 1)
        w_interp = (t_sec * fps) - f_idx
        f_idx_next = min(f_idx + 1, total_frames - 1)
        
        _, _, gust_curr, _, nEmit_curr = metrics[f_idx]
        _, _, gust_next, _, nEmit_next = metrics[f_idx_next]
        
        gust = gust_curr + (gust_next - gust_curr) * w_interp
        nEmit = nEmit_curr + (nEmit_next - nEmit_curr) * w_interp
        
        # --- (A) WIND ROAR & BUFFETING ---
        noise_wl = next_pink()
        noise_wr = next_pink()
        
        wind_l = wind_lp_l.process(noise_wl) * 0.015 * 1.50
        wind_r = wind_lp_r.process(noise_wr) * 0.015 * 1.50
        
        # Low frequency pressure buffeting
        br_l, br_r = next_brown_stereo()
        buff_l = buff_lp_l.process(br_l) * 0.28
        buff_r = buff_lp_r.process(br_r) * 0.28
        
        # --- (B) ORGANIC WAVE CRASH & OUTDOOR DIFFUSE REVERB ---
        wave_energy_ratio = clamp(nEmit / 3500.0, 0.0, 1.0)
        target_crash = wave_energy_ratio * (1.0 + 0.35 * math.sin(t_sec * 0.12))
        target_crash = min(target_crash, 1.0)
        
        if target_crash > crash_env:
            crash_env += (target_crash - crash_env) * 0.18
        else:
            crash_env += (target_crash - crash_env) * 0.00009
            
        br_c_l, br_c_r = next_brown_stereo()
        
        # Dry crash signals (scaled by slow crash_env)
        dry_crash_l = crash_lp_l.process(br_c_l) * 5.50 * crash_env
        dry_crash_r = crash_lp_r.process(br_c_r) * 5.50 * crash_env
        
        # Feed the DYNAMIC dry_crash into the delay buffers
        # Feedback gain lowered to 0.65 - 0.78 to target a tight, natural 0.9s decay tail
        rev_l = (reverb_l1.process(dry_crash_l, 0.78) +
                 reverb_l2.process(dry_crash_l, 0.74) +
                 reverb_l3.process(dry_crash_l, 0.70) +
                 reverb_l4.process(dry_crash_l, 0.66)) * 0.25
                 
        rev_r = (reverb_r1.process(dry_crash_r, 0.78) +
                 reverb_r2.process(dry_crash_r, 0.74) +
                 reverb_r3.process(dry_crash_r, 0.70) +
                 reverb_r4.process(dry_crash_r, 0.66)) * 0.25
                 
        # Output mix: Calibrated to 85% Dry / 15% Wet for outdoor space presence without echo clutter
        crash_l = dry_crash_l * 0.85 + rev_l * 0.15
        crash_r = dry_crash_r * 0.85 + rev_r * 0.15
        
        # --- (C) CRISP CRYSTALLINE FOAM FIZZ (16.5kHz, White/Pink noise) ---
        target_fizz = (nEmit / 2200.0 + 0.08)
        
        if target_fizz > fizz_env:
            fizz_env += (target_fizz - fizz_env) * 0.12
        else:
            fizz_env += (target_fizz - fizz_env) * 0.00015
            
        pink_fizz_l = next_pink()
        pink_fizz_r = next_pink()
        white_fizz_l = random.uniform(-1.0, 1.0)
        white_fizz_r = random.uniform(-1.0, 1.0)
        
        mixed_fizz_l = pink_fizz_l * 0.75 + white_fizz_l * 0.25
        mixed_fizz_r = pink_fizz_r * 0.75 + white_fizz_r * 0.25
        
        # Process fizz: LPF 16.5kHz
        fizz_l = fizz_lp_l.process(mixed_fizz_l) * 0.055 * fizz_env
        fizz_r = fizz_lp_r.process(mixed_fizz_r) * 0.055 * fizz_env
        
        # --- MIXING ---
        mix_l = wind_l + buff_l + crash_l + fizz_l
        mix_r = wind_r + buff_r + crash_r + fizz_r
        
        # Soft master saturation
        mix_l = math.tanh(mix_l * 0.70) * 0.95
        mix_r = math.tanh(mix_r * 0.70) * 0.95
        
        out_samples.append((int(mix_l * 32767), int(mix_r * 32767)))
        
    # Write WAV file
    with open(output_wav_path, 'wb') as f:
        f.write(b'RIFF')
        data_size = len(out_samples) * 4
        f.write(struct.pack('<I', 36 + data_size))
        f.write(b'WAVE')
        f.write(b'fmt ')
        f.write(struct.pack('<I', 16))
        f.write(struct.pack('<H', 1))
        f.write(struct.pack('<H', 2))
        f.write(struct.pack('<I', sample_rate))
        f.write(struct.pack('<I', sample_rate * 4))
        f.write(struct.pack('<H', 4))
        f.write(struct.pack('<H', 16))
        f.write(b'data')
        f.write(struct.pack('<I', data_size))
        
        for left, right in out_samples:
            f.write(struct.pack('<hh', left, right))
            
    print(f"Successfully generated high-fidelity physical sound track: {output_wav_path}")

def clamp(val, min_val, max_val):
    return max(min_val, min(val, max_val))

if __name__ == '__main__':
    generate_wave_sound('data/audio_physics.txt', 'data/hokusai_sea_ambient.wav')
