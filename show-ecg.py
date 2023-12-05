import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, ifft
import pywt 
from scipy.ndimage import gaussian_filter1d



database = './ecgiddb/Person_01/rec_1.dat'
#database = './ecgiddb/Person_01/rec_10.dat'
#database = './ecgiddb/Person_01/rec_7.dat'
#database = './ecgiddb/Person_01/rec_8.dat'
#database = './ecgiddb/Person_01/rec_12.dat'
ecg_data = np.fromfile(database, dtype=np.int16)

sampling_rate = 1000
duration = 20
time_axis = np.arange(0, duration, 1 / sampling_rate)


#####################################       LOWPASS FILTER     #################################################

cutoff_frequency = 20
order = 1
b, a = signal.butter(order, cutoff_frequency / (sampling_rate / 2), btype='low')

# Applicazione del filtro al segnale ECG
filtered_ecg_data = signal.filtfilt(b, a, ecg_data)

# Mostra i grafici
r = plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(time_axis, ecg_data)
plt.title('Segnale ECG Originale')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_axis, filtered_ecg_data)
plt.title('Segnale ECG Filtrato (Passa-Basso)')
plt.grid(True)

plt.tight_layout()
plt.show()

r.savefig('filtered.png')


#####################################     HIGHPASS FILTER UN PO' INUTILE    #################################################

cutoff_frequency = 200
order = 1
b, a = signal.butter(order, cutoff_frequency / (sampling_rate / 2), btype='high')

filtered_ecg_data = signal.filtfilt(b, a, ecg_data)

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(time_axis, ecg_data)
plt.title('Segnale ECG Originale')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_axis, filtered_ecg_data)
plt.title('Segnale ECG Filtrato (Passa-Basso)')
plt.grid(True)

plt.tight_layout()
plt.show()


#####################################       BANDPASS FILTER     #################################################

lowcut = 2
highcut = 20
order = 1
b, a = signal.butter(order, [lowcut / (sampling_rate / 2), highcut / (sampling_rate / 2)], btype='band')

filtered_ecg_data = signal.filtfilt(b, a, ecg_data)

# Segnale ECG originale
plt.subplot(2, 1, 1)
plt.plot(time_axis, ecg_data)
plt.title('Segnale ECG Originale')
plt.ylabel('Ampiezza')
plt.grid(True)

# Segnale ECG filtrato
plt.subplot(2, 1, 2)
plt.plot(time_axis, filtered_ecg_data)
plt.title('Segnale ECG Filtrato (Passa-Banda) con cutoff ' +str(lowcut) +' ' +str(highcut))
plt.ylabel('Ampiezza')
plt.grid(True)

plt.tight_layout()
plt.show()



#####################################       WAVELET TRANSFORM     #################################################


wavelet= 'db4'
coeffs = pywt.wavedec(ecg_data, wavelet)

threshold = 10
coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

# Segnale con la trasformata wavelet
filtered_ecg_data = pywt.waverec(coeffs_thresholded, wavelet)

plt.figure(figsize=(12, 8))

# Segnale ECG originale
plt.subplot(2, 1, 1)
plt.plot(time_axis, ecg_data)
plt.title('Segnale ECG Originale')
plt.xlabel('Tempo (s)')
plt.ylabel('Ampiezza')
plt.grid(True)

# Segnale ECG filtrato con trasformata wavelet
plt.subplot(2, 1, 2)
plt.plot(time_axis, filtered_ecg_data)
plt.title('Segnale ECG Filtrato (Wavelet)')
plt.xlabel('Tempo (s)')
plt.ylabel('Ampiezza')
plt.grid(True)

plt.tight_layout()
plt.show()


#####################################       GAUSSIAN FILTER     #################################################


sigma = 8
smoothed_ecg_data = gaussian_filter1d(ecg_data, sigma=sigma)

plt.figure(figsize=(12, 8))

# Segnale ECG originale
plt.subplot(2, 1, 1)
plt.plot(time_axis, ecg_data)
plt.title('Segnale ECG Originale')
plt.xlabel('Tempo (s)')
plt.ylabel('Ampiezza')
plt.grid(True)

# Segnale ECG con smoothing gaussiano
plt.subplot(2, 1, 2)
plt.plot(time_axis, smoothed_ecg_data)
plt.title('Segnale ECG con Smoothing Gaussiano')
plt.ylabel('Ampiezza')
plt.grid(True)

plt.tight_layout()
plt.show()


#####################################       Fourier Transform     #################################################

#In realtà forse la trasformata di Fourier è poco interessante in quanto si filtrano le frequenze come si farebbe per un passa-banda

# Calcola la trasformata (fast)
ecg_fft = fft(ecg_data)

cutoff_frequency_low = 2
cutoff_frequency_high = 200 

# Trova gli indici delle frequenze corrispondenti ai tagli
lowcut_idx = int(cutoff_frequency_low * len(ecg_fft) / sampling_rate)
highcut_idx = int(cutoff_frequency_high * len(ecg_fft) / sampling_rate)

ecg_fft[:lowcut_idx] = 0
ecg_fft[highcut_idx:] = 0

# Calcola la Trasformata inversa di Fourier
filtered_ecg_data = ifft(ecg_fft)

# Plot del segnale ECG originale e del segnale filtrato
plt.figure(figsize=(12, 8))

# Segnale ECG originale
plt.subplot(2, 1, 1)
plt.plot(time_axis, ecg_data)
plt.title('Segnale ECG Originale')
plt.ylabel('Ampiezza')
plt.grid(True)

# Segnale ECG filtrato con Trasformata di Fourier
plt.subplot(2, 1, 2)
plt.plot(time_axis, filtered_ecg_data)
plt.title('Segnale ECG Filtrato (Fourier)')
plt.ylabel('Ampiezza')
plt.grid(True)

plt.tight_layout()
plt.show()


#####################################       Wavelet e Passa basso     #################################################


wavelet = 'db4'  
coeffs = pywt.wavedec(ecg_data, wavelet)

threshold = 10
coeffs_thresholded = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

# Wavelet
filtered_ecg_data_wavelet = pywt.waverec(coeffs_thresholded, wavelet)

# filtro passa-basso
cutoff_frequency = 20 
order = 4 
b, a = signal.butter(order, cutoff_frequency / (sampling_rate / 2), btype='low')

filtered_ecg_data_combined = signal.filtfilt(b, a, filtered_ecg_data_wavelet)

plt.figure(figsize=(12, 12))

# Segnale ECG originale
plt.subplot(3, 1, 1)
plt.plot(time_axis, ecg_data)
plt.title('Segnale ECG Originale')
plt.ylabel('Ampiezza')
plt.grid(True)

# Segnale ECG filtrato con trasformata wavelet
plt.subplot(3, 1, 2)
plt.plot(time_axis, filtered_ecg_data_wavelet)
plt.title('Segnale ECG Filtrato (Wavelet)')
plt.ylabel('Ampiezza')
plt.grid(True)

# Segnale ECG filtrato con trasformata wavelet e filtro passa-basso
plt.subplot(3, 1, 3)
plt.plot(time_axis, filtered_ecg_data_combined)
plt.title('Segnale ECG Filtrato (Wavelet + Passa-Basso)')
plt.ylabel('Ampiezza')
plt.grid(True)

plt.tight_layout()
plt.show()


#####################################       Filtro ellittico     #################################################


cutoff_frequency_low = 1  
cutoff_frequency_high = 20
rp = 1  
rs = 200

b, a = signal.ellip(4, rp, rs, [cutoff_frequency_low / (sampling_rate / 2), cutoff_frequency_high / (sampling_rate / 2)], btype='band')

filtered_ecg_data_combined = signal.filtfilt(b, a, ecg_data)

plt.figure(figsize=(12, 12))

plt.subplot(2, 1, 1)
plt.plot(time_axis, ecg_data)
plt.title('Segnale ECG Originale')
plt.ylabel('Ampiezza')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(time_axis, filtered_ecg_data_combined)
plt.title('Segnale ECG Filtrato (Ellittico)')
plt.ylabel('Ampiezza')
plt.grid(True)

plt.tight_layout()
plt.show()