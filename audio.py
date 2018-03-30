import librosa
import numpy as np
import math

def detect_pitch(y, sr, t):
    index = magnitudes[:, t].argmax()

    pitch = pitches[index, t]
    return pitch

def detect_pitch_yield(y, sr, rate):
    for y_1 in y:
        #print('TYPE')
        #print(len(y_1))
        pitches, magnitudes = librosa.piptrack(y=y_1, sr=sr)
        print(np.shape(magnitudes))
        #np.savetxt('foo.csv', pitches, delimiter=',')
        #print(len(pitches[0]))
        index = magnitudes.argmax()
        pitch = pitches[index, len(pitches[0])-1]
        yielded = False
        print(y_1)
        print(pitches[index])
        for p in pitches[index]:
            if p > 0.0 and yielded is False:
                yielded = True
                yield p
        #rate+= rate
        if yielded is False:
            yield pitch

y, sr = librosa.load(librosa.util.example_audio_file())
#y, sr = librosa.load('example.mp3')

#print(type(y), type(sr))
time = librosa.get_duration(y=y, sr=sr)
print('TYPE1', type(y[0]))
print(len(y))
counter = 0
print(y)
for i in np.nditer(y[0]):
    print(i)
    counter += 1

print('y = '+str(counter))
print(time)
print(y.shape, sr)

pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

RR = librosa.time_to_frames(np.arange(0, 1, 0.1),sr=22050, hop_length=512)
print(RR)
split_time = math.floor(time/0.032)

# n=duration/0.032 rounded down
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

chunkedY = list(chunks(y, split_time))

print(len(chunkedY[0]))
print(len(chunkedY))
print('type obj' + str(type(chunkedY[0][0])))
print(len(chunkedY[-1]))
print(type(chunkedY))
pitches_32 = list(detect_pitch_yield(np.asarray(chunkedY), 22050, 1920))
print("PITCHES")
print(pitches_32)
'''try:
    for i in range(0, 3000):
        #print(i)
        detect_pitch(y,sr,i)
except Exception as e:
    print(e)
'''
#print(pitches)

#numpy.savetxt('foo.csv', pitches, delimiter=',')
