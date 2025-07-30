import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt

WAVELET_PARAM = {
    'low': 1.0,
    'high': 100.0,
    'step': 1.0,
    'n_cycles': 7,
    'length': 1.0,
    'addtaper': False,
    'taperlength': 0.1
}

def calculateTFDPower(
    signal: np.ndarray,
    si: float,
    params: dict
) -> np.ndarray:
    """    Calculates the time-frequency distribution power of a signal using wavelet convolution.
    Args:
        signal (np.ndarray): The input signal to be analyzed.
        si (float): Sampling interval of the signal.
        params (dict): Dictionary containing parameters for the wavelet transform.
            Expected keys: 'low', 'high', 'step', 'n_cycles', 'length', 'addtaper', 'taperlength'.
    Returns:
        np.ndarray: The power matrix of the signal after wavelet convolution.
    """

    frequencies = np.arange(params['low'], params['high'] + params['step'], params['step'])
    powers = []
    for frequency in frequencies:
        wavelet  = createMorletWavelet(
           frequency,
           si,
           params['n_cycles'],
           params['length']
        )
        conv_fft = convolveWithFft(signal, wavelet)
        #   taperlength=round(parameters.taperlength/parameters.interval);
        #   conv_fft = convolveWithFft([inputSignal(taperlength:-1:1,:);inputSignal;inputSignal(end:-1:end-taperlength,:)], wavelet); % ha a taperlength hosszabb, mint az inputSignal, akkor gebasz van, de ilyen elvileg nem fordulhat el�..
        #   conv_fft =conv_fft(:,taperlength+1:taperlength+size(inputSignal,1));

        powers.append(np.abs(conv_fft) ** 2)

    power_matrix = np.stack(powers, axis=0)
    return power_matrix

def convolveWithFft(signal: np.ndarray, response: np.ndarray) -> np.ndarray:
    """
    Performs convolution of the signal with the response using FFT.
    Args:
        signal (np.ndarray): The input signal to be convolved.
        response (np.ndarray): The response function to convolve with the signal.
    Returns:
        np.ndarray: The convolved signal.
    """
    new_length = len(signal) + len(response) - 1
    new_length_pow = 2 ** int(np.ceil(np.log2(new_length)))
    half_response_length = round((len(response)-1)/2)
        
    signal = np.nan_to_num(signal, nan=0.0).squeeze()
    response = np.nan_to_num(response, nan=0.0).squeeze()

    signal_fft = fft(signal, new_length_pow)
    response_fft = fft(response, new_length_pow)
    conv_fft = ifft(signal_fft * response_fft)
    conv_fft = conv_fft[:new_length]
    conv_fft = conv_fft[half_response_length:-half_response_length]

    return conv_fft


def createMorletWavelet(
    frequency: float,
    si: float,
    n_cycles: int = 7,
    length: float = 1
) -> np.ndarray:
    """
    Creates a complex Morlet wavelet based on the given base frequency and parameters.
    """

    sigma_f = frequency / n_cycles
    sigma_t = 1 / (2 * np.pi * sigma_f)
    sigma_t_sq = sigma_t ** 2
    A = np.sqrt(1 / (sigma_t * np.sqrt(np.pi)))

    t = np.arange(-length, length + si, si)
    t_sq = t ** 2

    wavelet = A * np.exp(-t_sq / (2 * sigma_t_sq)) * np.exp(2 * 1j * np.pi * frequency * t)
    return wavelet


class Wavelet:
    """A class that represents a generated wavelet."""

    def __init__(self, response: np.ndarray):
        self._response = np.nan_to_num(response, nan=0.0).squeeze()

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        new_length = len(signal) + len(self._response) - 1
        new_length_pow = 2 ** int(np.ceil(np.log2(new_length)))
        half_response_length = round((len(self._response)-1)/2)

        signal = np.nan_to_num(signal, nan=0.0).squeeze()

        signal_fft = fft(signal, new_length_pow)
        response_fft = fft(self._response, new_length_pow)
        conv_fft = ifft(signal_fft * response_fft)
        conv_fft = conv_fft[:new_length]
        conv_fft = conv_fft[half_response_length:-half_response_length]

        return conv_fft


t = np.linspace(0, 10, 10001)

theta = np.where(
    (4 <= t) & (t < 7),
    np.sin(2 * np.pi * 7.5 * t),
    0
)
spindle = np.where(
    (2<=t) & (t<2.75),
    np.sin(2 * np.pi * 15 * t),
    0
)
gamma = np.where(
    (5<=t) & (t<5.2),
    np.sin(2 * np.pi * 70 * t),
    0
)

signal = (
    0.1 * np.sin(2 * np.pi * 1 * t) +
    spindle +
    gamma +
    theta +
    np.random.normal(0, 0.1, len(t))
)

spectrogram = calculateTFDPower(
    signal,
    si=0.001,
    params=WAVELET_PARAM
)

fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(t, signal)
axs[0].set_title('Artificial Spindle and Gamma with noise')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Amplitude')

# Use pcolormesh as the counterpart of Matlab's pcolor
im = axs[1].pcolormesh(
    t,
    np.arange(WAVELET_PARAM['low'], WAVELET_PARAM['high'] + WAVELET_PARAM['step'], WAVELET_PARAM['step']),
    spectrogram,
    shading='auto',
    cmap='jet'
)
fig.colorbar(im, ax=axs[1], label='Power')
axs[1].set_title('Time-Frequency Distribution (Spectrogram)')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Frequency (Hz)')

plt.tight_layout()
plt.show()
