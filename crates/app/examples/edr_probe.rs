use bd_protocol::btbb::{self, BtHeader, ClassicBtPacket};
use bd_protocol::Timespec;
use num_complex::Complex32;
use std::fs;

fn main() {
    let path = std::env::args().nth(1).expect("capture path");
    let start: usize = std::env::args()
        .nth(2)
        .expect("start sample")
        .parse()
        .expect("numeric start sample");
    let raw = fs::read(path).expect("read capture");
    let iq = raw
        .chunks_exact(4)
        .map(|sample| {
            Complex32::new(
                i16::from_le_bytes([sample[0], sample[1]]) as f32,
                i16::from_le_bytes([sample[2], sample[3]]) as f32,
            )
        })
        .skip(start)
        .take(40_000)
        .collect::<Vec<_>>();
    let baseband = bd_dsp::edr::extract_edr_channel_4m(&iq, -1_000_000.0, 16_000_000);
    let two_pi = 2.0 * std::f32::consts::PI;
    let refined = bd_dsp::edr::refine_cfo(
        &baseband,
        4,
        two_pi * 300_000.0 / 4_000_000.0,
        two_pi * 5_000.0 / 4_000_000.0,
    );
    eprintln!(
        "samples={} baseband={} refine={:.0}Hz",
        iq.len(),
        baseband.len(),
        refined * 4_000_000.0 / two_pi
    );

    for alias in -3..=3 {
        let correction = refined + two_pi * (alias as f32 * 125_000.0) / 4_000_000.0;
        let mut corrected = baseband.clone();
        bd_dsp::edr::derotate(&mut corrected, correction);
        let matched = bd_dsp::edr::rrc_matched_filter(&corrected, 4, 0.4, 6);
        for reference_sample in 0..600 {
            let lock = bd_dsp::edr::SyncLock {
                score: 0.0,
                reference_sample,
                residual: 0.0,
                conjugated: false,
            };
            let bits =
                bd_dsp::edr::demod_dpsk_detrended_from_sync(&matched, lock, 4, 3);
            for clk6 in 0..64 {
                let Some(payload) = btbb::edr_payload_header(&bits, clk6) else {
                    continue;
                };
                if !(604..=612).contains(&payload.length) {
                    continue;
                }
            let header = BtHeader {
                lt_addr: 1,
                pkt_type: 0x0d,
                flow: 1,
                arqn: 0,
                seqn: 0,
                hec: 0,
                clk6,
            };
            let mut packet = ClassicBtPacket {
                lap: 0x00_00_00, // set to the piconet LAP under test
                ac_errors: 0,
                sync_offset: 0,
                rssi_db: 0,
                noise_db: 0,
                freq: 2440,
                timestamp: Timespec::default(),
                raw_header: [0; 7],
                has_header: true,
                payload: Vec::new(),
                uap: Some(0x00), // set to the piconet UAP under test
                uap_verified: true,
                nap: None,
                header: Some(header),
                decoded_payload: Vec::new(),
                crc_ok: false,
                clkn: None,
            };
            if btbb::enrich_edr_candidate(&mut packet, &bits, 0x00, header) {
                eprintln!(
                    "CRC PASS alias={} correction={:.0}Hz reference={} clk={} bytes={}",
                    alias,
                    correction * 4_000_000.0 / two_pi,
                    reference_sample,
                    clk6,
                    packet.decoded_payload.len()
                );
            }
            }
        }
    }
}
