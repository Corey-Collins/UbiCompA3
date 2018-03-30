import aubio

SAMPLE_RATE = 32

def extract_pitch(file, rate=44096):
    window = int(rate / SAMPLE_RATE)
    audio_src = aubio.source(file, rate, window)
    pitch_obj = aubio.pitch('yin', window, window, rate)
    pitch_obj.set_unit("Hz")
    pitch_obj.set_silence(-40)
    pitches = []
    total_frames = 0
    while True:
        samples, read = audio_src()
        pitch = pitch_obj(samples)[0]
        if pitch > 1000:  # Not likely, just zero it out...
            pitch = 0
        pitches.append(pitch)
        total_frames += read
        if read < window: break
    return pitches

print(extract_pitch('example.mp3'))
