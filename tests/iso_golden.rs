// SPDX-License-Identifier: AGPL-3.0-only
//! The Rust isometric kit (`engine::iso`) against the anatomy kit's own output: every case in
//! tests/fixtures/iso_golden.json (written by scripts/iso_golden.mjs from iso-kit.mjs) must come
//! out byte for byte the same.

use thme::engine::iso::*;

fn plans() -> [Plan; 3] {
    [
        Plan {
            x: 0.0,
            y: 0.0,
            w: 40.0,
            d: 30.0,
            r: 0.0,
        },
        Plan {
            x: -10.0,
            y: 5.0,
            w: 60.0,
            d: 24.0,
            r: 6.0,
        },
        Plan {
            x: 3.3,
            y: -7.7,
            w: 12.5,
            d: 12.5,
            r: 6.25,
        },
    ]
}

fn projections() -> [Projection; 3] {
    [
        Projection {
            origin: (200.0, 150.0),
            scale: 2.0,
            azimuth: 45.0,
            elevation: 30.0,
        },
        Projection {
            origin: (320.5, 180.25),
            scale: 3.7,
            azimuth: 35.0,
            elevation: 30.0,
        },
        Projection {
            origin: (100.0, 220.0),
            scale: 1.3,
            azimuth: 60.0,
            elevation: 44.0,
        },
    ]
}

/// The same cases, in the same order, as scripts/iso_golden.mjs.
pub fn cases() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (pi, p) in projections().iter().enumerate() {
        for (i, plan) in plans().iter().enumerate() {
            let fi = i as f64;
            out.push((
                format!("slab p{pi} plan{i}"),
                solid_svg(
                    &slab_of(plan, 2.0, 10.0 + fi, p, 8, fi * 0.6),
                    &SolidStyle::default(),
                ),
            ));
            out.push((
                format!("slab-lit-halo p{pi} plan{i}"),
                solid_svg(
                    &slab_of(plan, 0.0, 4.0, p, 8, 0.0),
                    &SolidStyle {
                        tone: Tone::Lo,
                        halo: true,
                        lit: true,
                        ..Default::default()
                    },
                ),
            ));
            out.push((
                format!("outline p{pi} plan{i}"),
                plan_outline(&inset_plan(plan, 3.0), 12.0, p, 8),
            ));
        }
        out.push((
            format!("cylinder p{pi}"),
            solid_svg(
                &cylinder(10.0, 12.0, 8.5, 1.0, 22.0, p, 32, 0.8),
                &SolidStyle {
                    crease: Some(Tone::Mid),
                    ..Default::default()
                },
            ),
        ));
        out.push((
            format!("thin-cylinder p{pi}"),
            solid_svg(
                &cylinder(-4.0, 2.0, 0.55, 0.0, 0.4, p, 12, 0.0),
                &SolidStyle {
                    flat: true,
                    tone: Tone::Mid,
                    ..Default::default()
                },
            ),
        ));
        out.push((
            format!("segment p{pi}"),
            line_svg(&segment((0.0, 0.0, 0.0), (12.3, -4.4, 9.9), p), Tone::Mid),
        ));
        out.push((
            format!("dots p{pi}"),
            dots_svg(
                &[iso((1.0, 2.0, 3.0), p), iso((-4.25, 7.5, 0.0), p)],
                0.5,
                Tone::Hi,
            ),
        ));
        out.push((
            format!("wire p{pi}"),
            wire_svg(
                iso((0.0, 0.0, 5.0), p),
                iso((30.0, 10.0, 5.0), p),
                &[iso((15.0, 0.0, 5.0), p)],
                Tone::Lo,
            ),
        ));
        let l_shape = [
            (0.0, 0.0),
            (20.0, 0.0),
            (20.0, 6.0),
            (8.0, 6.0),
            (8.0, 18.0),
            (0.0, 18.0),
        ];
        out.push((
            format!("extrude-concave p{pi}"),
            solid_svg(
                &extrude(
                    &l_shape,
                    0.0,
                    7.0,
                    p,
                    ExtrudeOptions {
                        convex: false,
                        bevel: 0.0,
                    },
                ),
                &SolidStyle::default(),
            ),
        ));
    }
    let fit = fit_projection(
        &[
            (0.0, 0.0, 0.0),
            (100.0, 60.0, 0.0),
            (0.0, 60.0, 40.0),
            (100.0, 0.0, 40.0),
        ],
        600.0,
        400.0,
        24.0,
        40.0,
        ELEVATION,
    );
    out.push((
        "fit".into(),
        format!("{:?}", (fit.origin.0, fit.origin.1, fit.scale)),
    ));
    out.push((
        "figure".into(),
        figure_svg(600.0, 400.0, "A <test> & figure", "<g/>", None),
    ));
    out.push((
        "plate".into(),
        plate_html(
            "Fig 1",
            "Rig",
            "Drag",
            "0.42 mm",
            "Caption & more",
            "<svg/>",
        ),
    ));
    out
}

#[test]
fn rust_port_matches_the_anatomy_kit_byte_for_byte() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/iso_golden.json"
    ))
    .unwrap();
    let golden: Vec<serde_json::Value> = serde_json::from_str(&text).unwrap();
    let ours = cases();
    assert_eq!(golden.len(), ours.len(), "case count");
    let mut mismatches = Vec::new();
    for (g, (name, svg)) in golden.iter().zip(&ours) {
        assert_eq!(g["name"].as_str().unwrap(), name, "case order");
        let want = g["svg"].as_str().unwrap();
        if name == "fit" {
            // Numbers, compared to 1e-9: JSON and Rust print floats differently.
            let v: serde_json::Value = serde_json::from_str(want).unwrap();
            let w = [
                v["origin"][0].as_f64().unwrap(),
                v["origin"][1].as_f64().unwrap(),
                v["scale"].as_f64().unwrap(),
            ];
            let got: Vec<f64> = svg
                .trim_matches(|c| c == '(' || c == ')')
                .split(", ")
                .map(|x| x.parse().unwrap())
                .collect();
            assert!(
                w.iter().zip(&got).all(|(a, b)| (a - b).abs() < 1e-9),
                "fit: {w:?} vs {got:?}"
            );
        } else if want != svg {
            mismatches.push(format!(
                "{name}\n  kit:  {}\n  rust: {}",
                &want[..want.len().min(300)],
                &svg[..svg.len().min(300)]
            ));
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} of {} cases differ:\n{}",
        mismatches.len(),
        ours.len(),
        mismatches.join("\n")
    );
}
