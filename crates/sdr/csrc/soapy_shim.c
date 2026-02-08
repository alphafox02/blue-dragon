// Thin C shim for SoapySDR API calls.
// Direct Rust FFI to SoapySDR causes segfaults in readStream with the
// SoapyUHD module. Routing through C wrappers resolves the issue.

#include <SoapySDR/Device.h>
#include <stddef.h>
#include <string.h>

SoapySDRDevice *soapy_shim_make(const char *args) {
    return SoapySDRDevice_makeStrArgs(args);
}

void soapy_shim_unmake(SoapySDRDevice *dev) {
    SoapySDRDevice_unmake(dev);
}

int soapy_shim_set_sample_rate(SoapySDRDevice *dev, double rate) {
    return SoapySDRDevice_setSampleRate(dev, SOAPY_SDR_RX, 0, rate);
}

int soapy_shim_set_frequency(SoapySDRDevice *dev, double freq) {
    return SoapySDRDevice_setFrequency(dev, SOAPY_SDR_RX, 0, freq, NULL);
}

int soapy_shim_set_gain(SoapySDRDevice *dev, double gain) {
    return SoapySDRDevice_setGain(dev, SOAPY_SDR_RX, 0, gain);
}

int soapy_shim_set_bandwidth(SoapySDRDevice *dev, double bw) {
    return SoapySDRDevice_setBandwidth(dev, SOAPY_SDR_RX, 0, bw);
}

SoapySDRStream *soapy_shim_setup_stream(SoapySDRDevice *dev, const char *format) {
    size_t chan = 0;
    return SoapySDRDevice_setupStream(dev, SOAPY_SDR_RX, format, &chan, 1, NULL);
}

int soapy_shim_activate(SoapySDRDevice *dev, SoapySDRStream *stream) {
    return SoapySDRDevice_activateStream(dev, stream, 0, 0, 0);
}

void soapy_shim_deactivate(SoapySDRDevice *dev, SoapySDRStream *stream) {
    SoapySDRDevice_deactivateStream(dev, stream, 0, 0);
}

void soapy_shim_close(SoapySDRDevice *dev, SoapySDRStream *stream) {
    SoapySDRDevice_closeStream(dev, stream);
}

size_t soapy_shim_get_mtu(SoapySDRDevice *dev, SoapySDRStream *stream) {
    return SoapySDRDevice_getStreamMTU(dev, stream);
}

int soapy_shim_read(SoapySDRDevice *dev, SoapySDRStream *stream,
                    void *buf, size_t num_samples) {
    void *buffs[] = { buf };
    int flags = 0;
    long long timeNs = 0;
    return SoapySDRDevice_readStream(dev, stream, buffs, num_samples,
                                     &flags, &timeNs, 100000);
}

// Query the native stream format's full-scale value.
// bladeRF SC16_Q11 -> 2048, USRP int16 -> 32768, HackRF int8 -> 128, etc.
double soapy_shim_get_full_scale(SoapySDRDevice *dev) {
    double fullScale = 0;
    SoapySDRDevice_getNativeStreamFormat(dev, SOAPY_SDR_RX, 0, &fullScale);
    return (fullScale > 0) ? fullScale : 32768.0;
}

// Enumerate: returns count, fills serial/driver arrays
size_t soapy_shim_enumerate(char labels[][64], char drivers[][32], size_t max_devices) {
    size_t length = 0;
    SoapySDRKwargs *results = SoapySDRDevice_enumerate(NULL, &length);
    if (!results || length == 0) return 0;
    if (length > max_devices) length = max_devices;

    for (size_t i = 0; i < length; i++) {
        labels[i][0] = '\0';
        drivers[i][0] = '\0';
        for (size_t j = 0; j < results[i].size; j++) {
            if (strcmp(results[i].keys[j], "label") == 0) {
                strncpy(labels[i], results[i].vals[j], 63);
                labels[i][63] = '\0';
            }
            if (strcmp(results[i].keys[j], "driver") == 0) {
                strncpy(drivers[i], results[i].vals[j], 31);
                drivers[i][31] = '\0';
            }
        }
    }

    SoapySDRKwargsList_clear(results, length);
    return length;
}
