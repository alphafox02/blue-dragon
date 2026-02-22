# PHY-Layer Sensitivity Test Results

Conducted cable test: nRF52840 (Butterfly/WHAD) -> 30 dB attenuator -> SDR RX.
BLE advertising on channel 38 (2426 MHz), 15 seconds per gain step.

## bladeRF 2.0 micro

- Date: 2026-02-21
- Mode: SC16_Q11 (normal), 12 MHz bandwidth, 12 channels
- Firmware: 2.4.0, FPGA: 0.15.3

| TX Power | RX Gain | BLE Pkts | CRC % | Valid/Total | Bursts |
|----------|---------|----------|-------|-------------|--------|
| high     | 60 dB   | 382      | 75.4  | 288/382     | 99547  |
| high     | 50 dB   | 15       | 100.0 | 13/13       | 4928   |
| high     | 40 dB   | 29       | 96.6  | 28/29       | 3523   |
| high     | 30 dB   | 175      | 95.4  | 167/175     | 2452   |
| high     | 20 dB   | 214      | 97.7  | 209/214     | 4216   |
| high     | 10 dB   | 200      | 97.5  | 195/200     | 210    |
| high     | 5 dB    | 95       | 95.8  | 91/95       | 216    |
| high     | 0 dB    | 0        | -     | 0/0         | 224    |
| medium   | 60 dB   | 242      | 76.4  | 185/242     | 85385  |
| medium   | 50 dB   | 43       | 81.4  | 35/43       | 4390   |
| medium   | 40 dB   | 42       | 90.5  | 38/42       | 2963   |
| medium   | 30 dB   | 181      | 91.2  | 165/181     | 1918   |
| medium   | 20 dB   | 197      | 97.0  | 191/197     | 4030   |
| medium   | 10 dB   | 196      | 96.4  | 189/196     | 620    |
| medium   | 5 dB    | 95       | 96.8  | 92/95       | 209    |
| medium   | 0 dB    | 0        | -     | 0/0         | 217    |
| low      | 60 dB   | 286      | 67.0  | 191/285     | 86057  |
| low      | 50 dB   | 21       | 90.5  | 19/21       | 4362   |
| low      | 40 dB   | 31       | 90.3  | 28/31       | 3023   |
| low      | 30 dB   | 185      | 96.2  | 178/185     | 1929   |
| low      | 20 dB   | 199      | 97.5  | 194/199     | 4174   |
| low      | 10 dB   | 205      | 98.5  | 202/205     | 640    |
| low      | 5 dB    | 87       | 97.7  | 85/87       | 210    |
| low      | 0 dB    | 1        | 100.0 | 1/1         | 214    |

### Analysis

- **Sweet spot**: 10-20 dB gain, ~200 pkts/15s, 97-98% CRC
- **Sensitivity floor**: ~5 dB gain (signal drops out at 0 dB)
- **Clipping onset**: 60 dB gain (CRC drops to 67-76%, burst count explodes from noise)
- **Estimated input power**: -26 to -30 dBm (BLE TX 0 to +4 dBm minus 30 dB attenuator)
- **Effective sensitivity**: approximately -31 dBm at 12 MHz bandwidth
- **Zero overflows** across all 24 test runs

### Notes

- TX source: nRF52840 with SMA connector (Butterfly firmware via WHAD)
- 30 dB fixed SMA attenuator between TX and RX
- Burst count at high gain reflects noise-triggered false detections, not real packets
- Non-monotonic packet count at 40-50 dB likely due to burst detector threshold interaction

## USRP B210 (LibreSDR clone)

- Date: 2026-02-21
- Mode: SC8 over USB, 12 MHz bandwidth, 12 channels
- UHD: 4.9.0

| TX Power | RX Gain | BLE Pkts | CRC % | Valid/Total | Bursts |
|----------|---------|----------|-------|-------------|--------|
| high     | 60 dB   | 422      | 92.2  | 389/422     | 6835   |
| high     | 50 dB   | 185      | 85.9  | 159/185     | 46266  |
| high     | 40 dB   | 168      | 95.2  | 160/168     | 13783  |
| high     | 30 dB   | 165      | 95.7  | 155/162     | 6383   |
| high     | 20 dB   | 181      | 96.7  | 175/181     | 8149   |
| high     | 10 dB   | 209      | 96.2  | 201/209     | 1549   |
| high     | 5 dB    | 196      | 98.0  | 192/196     | 1271   |
| high     | 0 dB    | 199      | 96.0  | 191/199     | 1131   |
| medium   | 60 dB   | 456      | 91.6  | 417/455     | 9332   |
| medium   | 50 dB   | 191      | 89.0  | 170/191     | 38432  |
| medium   | 40 dB   | 168      | 94.6  | 159/168     | 12857  |
| medium   | 30 dB   | 176      | 97.1  | 169/174     | 5960   |
| medium   | 20 dB   | 179      | 96.6  | 173/179     | 8267   |
| medium   | 10 dB   | 202      | 98.5  | 199/202     | 1663   |
| medium   | 5 dB    | 198      | 97.5  | 193/198     | 1248   |
| medium   | 0 dB    | 201      | 94.5  | 190/201     | 1146   |
| low      | 60 dB   | 555      | 90.3  | 500/554     | 9694   |
| low      | 50 dB   | 175      | 93.7  | 163/174     | 51103  |
| low      | 40 dB   | 160      | 96.9  | 155/160     | 13960  |
| low      | 30 dB   | 171      | 97.6  | 166/170     | 6338   |
| low      | 20 dB   | 175      | 97.1  | 170/175     | 8019   |
| low      | 10 dB   | 203      | 94.1  | 191/203     | 1613   |
| low      | 5 dB    | 202      | 97.0  | 196/202     | 1225   |
| low      | 0 dB    | 198      | 92.9  | 184/198     | 1242   |

### Analysis

- **Sweet spot**: 5-30 dB gain, ~170-200 pkts/15s, 95-98% CRC
- **Sensitivity floor**: none - still decoding at 0 dB gain (92-96% CRC)
- **Clipping onset**: 60 dB still functional (90-92% CRC), much better than bladeRF
- **Consistent packet count**: 160-555 pkts across full gain range (never drops to zero)
- **Overflows**: 1 at 60 dB gain per TX power level (negligible)

### Comparison: USRP B210 vs bladeRF 2.0

| Metric                  | USRP B210      | bladeRF 2.0    |
|-------------------------|----------------|----------------|
| Peak CRC %              | 98.5%          | 100% (small N) |
| Usable gain range       | 0-60 dB        | 5-30 dB        |
| Packets at 0 dB gain    | ~200           | 0              |
| Packets at 60 dB gain   | 420-555        | 240-380        |
| CRC at 60 dB gain       | 90-92%         | 67-76%         |
| False burst sensitivity | Moderate       | Very high       |
| Overall                 | Better dynamic range, more forgiving | Narrower sweet spot, clips harder |

### Notes

- USRP decodes packets across the entire 0-60 dB gain range - significantly wider dynamic range
- bladeRF drops out entirely at 0 dB and clips badly at 60 dB
- USRP burst counts are much lower at high gain (6k-9k vs 85k-99k), suggesting better noise rejection
- Both achieve 95-98% CRC in their respective sweet spots
