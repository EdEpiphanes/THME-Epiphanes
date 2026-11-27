// SPDX-License-Identifier: AGPL-3.0-only
//! Pure 3D projection, colour, and polygon maths helpers for the live view.

use eframe::egui::{Color32, Pos2, Rect, Vec2};

/// True when the screen quad turns the same way at every corner.
pub(super) fn is_convex(pts: &[Pos2; 4]) -> bool {
    let mut sign = 0.0f32;
    for k in 0..4 {
        let (a, b, c) = (pts[k], pts[(k + 1) % 4], pts[(k + 2) % 4]);
        let cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
        if cross.abs() > 1e-3 {
            if sign != 0.0 && cross.signum() != sign {
                return false;
            }
            sign = cross.signum();
        }
    }
    true
}

pub(super) fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub(super) fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(super) fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-9);
    [v[0] / len, v[1] / len, v[2] / len]
}

pub(super) struct Face {
    pub(super) depth: f32,
    pub(super) pts: [Pos2; 4],
    pub(super) fill: Color32,
    pub(super) dots: f32,
    pub(super) glint: bool,
    pub(super) center: Pos2,
    pub(super) patch: usize,
}

pub(super) struct Camera {
    pub(super) yaw: f32,
    pub(super) pitch: f32,
    pub(super) dist: f32,
    pub(super) rect: Rect,
}

impl Camera {
    /// World point to camera space (x right, y up, z = depth away from the eye).
    pub(super) fn view(&self, p: [f32; 3]) -> [f32; 3] {
        let (sy, cy) = self.yaw.sin_cos();
        let x1 = p[0] * cy - p[2] * sy;
        let z1 = p[0] * sy + p[2] * cy;
        let (sp, cp) = self.pitch.sin_cos();
        let y2 = p[1] * cp - z1 * sp;
        let z2 = p[1] * sp + z1 * cp + self.dist;
        [x1, y2, z2]
    }

    /// Camera-space point to screen (caller ensures depth > 0.2).
    pub(super) fn screen(&self, v: [f32; 3]) -> Pos2 {
        let f = self.rect.height() * 1.15;
        self.rect.center() + Vec2::new(v[0] * f / v[2], -v[1] * f / v[2])
    }

    /// Projects a world point; returns the screen position and camera-space depth.
    pub(super) fn project(&self, p: [f32; 3]) -> Option<(Pos2, f32)> {
        let v = self.view(p);
        if v[2] < 0.2 {
            return None;
        }
        Some((self.screen(v), v[2]))
    }

    pub(super) fn quad(&self, corners: [[f32; 3]; 4]) -> Option<(f32, [Pos2; 4])> {
        let mut pts = [Pos2::ZERO; 4];
        let mut depth = 0.0;
        for (slot, corner) in pts.iter_mut().zip(corners) {
            let (pos, z) = self.project(corner)?;
            *slot = pos;
            depth += z / 4.0;
        }
        Some((depth, pts))
    }
}

pub(super) fn inside(point: Pos2, pts: &[Pos2; 4]) -> bool {
    let mut sign = 0.0f32;
    for i in 0..4 {
        let a = pts[i];
        let b = pts[(i + 1) % 4];
        let cross = (b.x - a.x) * (point.y - a.y) - (b.y - a.y) * (point.x - a.x);
        if cross != 0.0 {
            if sign != 0.0 && cross.signum() != sign {
                return false;
            }
            sign = cross.signum();
        }
    }
    true
}

pub(super) fn lerp(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    let mix = |x: u8, y: u8| (f32::from(x) + (f32::from(y) - f32::from(x)) * t).round() as u8;
    Color32::from_rgb(mix(a.r(), b.r()), mix(a.g(), b.g()), mix(a.b(), b.b()))
}

pub(super) fn with_alpha(c: Color32, alpha: u8) -> Color32 {
    Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), alpha)
}

pub(super) fn hash01(i: usize) -> f32 {
    let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 31;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 29;
    (x % 10_000) as f32 / 10_000.0
}
