# GPU vs CPU Comparison at -C 40

Date: 2026-02-22
Hardware: Intel i7-12700H / NVIDIA RTX 3060 Laptop GPU / Intel UHD iGPU
SDR: USRP B210 (USB 3.0)
Config: -C 40 (20 BLE channels, 2422-2460 MHz), --check-crc --stats
Signal: WHAD ButteRFly dongle advertising LE 1M through 30 dB attenuator
        plus ambient BLE traffic

## Results (30-second mark)

| Backend | BLE pkts | CRC% | Bursts | Overflow | BLE/sec |
|---------|----------|------|--------|----------|---------|
| NVIDIA RTX 3060 (OpenCL) | 1,486 | 95.2% | 48,983 | 1 | ~50 |
| Intel UHD iGPU (OpenCL) | 870 | 92.3% | 32,646 | 0 | ~29 |
| CPU-only (AVX2) | 1,820 | 94.0% | 47,824 | 0 | ~61 |

## Notes

- All three backends handle -C 40 without sample drops
- CPU AVX2 path is competitive with NVIDIA discrete GPU at this channel count
- Intel iGPU works but processes fewer bursts (slower OpenCL throughput)
- The iGPU lower burst/packet count may be due to OpenCL submit latency
  rather than raw compute -- the GPU path batches differently than CPU
- Higher channel counts (-C 60+) not tested here; GPU likely needed there
- The transmitter was briefly turned on OTA during NVIDIA test which may
  have slightly affected those numbers
