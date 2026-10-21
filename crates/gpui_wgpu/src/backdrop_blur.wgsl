// Backdrop blur: what the frame holds under a rectangle, halved a few times,
// blurred with a separable Gaussian at the smallest size and put back inside
// the rounded rectangle.

struct BlurPass {
    // The source texel a destination pixel reads: origin + position * scale.
    origin: vec2<f32>,
    scale: f32,
    sigma: f32,
    // Reads are clamped to these texels, so nothing past the used part of a
    // pooled texture leaks in.
    min_texel: vec2<f32>,
    max_texel: vec2<f32>,
    texel_size: vec2<f32>,
    direction: vec2<f32>,
}

@group(0) @binding(0) var<uniform> blur_pass: BlurPass;
@group(0) @binding(1) var t_source: texture_2d<f32>;
@group(0) @binding(2) var s_source: sampler;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vertex_id: u32) -> @builtin(position) vec4<f32> {
    let unit = vec2<f32>(f32((vertex_id << 1u) & 2u), f32(vertex_id & 2u));
    return vec4<f32>(unit * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
}

fn read(texel: vec2<f32>) -> vec4<f32> {
    let clamped = clamp(texel, blur_pass.min_texel, blur_pass.max_texel);
    return textureSampleLevel(t_source, s_source, clamped * blur_pass.texel_size, 0.0);
}

// Halves: each destination pixel covers 2x2 source texels. Five bilinear
// reads (dual Kawase) weigh a 4x4 neighbourhood, which keeps content that
// scrolls under a still rectangle from shimmering.
@fragment
fn fs_down(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let center = blur_pass.origin + position.xy * blur_pass.scale;
    let sum = read(center) * 4.0
        + read(center + vec2<f32>(-1.0, -1.0))
        + read(center + vec2<f32>(1.0, -1.0))
        + read(center + vec2<f32>(-1.0, 1.0))
        + read(center + vec2<f32>(1.0, 1.0));
    return sum / 8.0;
}

// One direction of the Gaussian, out to two standard deviations.
@fragment
fn fs_gaussian(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let center = blur_pass.origin + position.xy * blur_pass.scale;
    let sigma = blur_pass.sigma;
    var sum = read(center);
    var total = 1.0;
    let reach = i32(ceil(sigma * 2.0));
    for (var i = 1; i <= reach; i += 1) {
        let offset = f32(i);
        let weight = exp(-0.5 * offset * offset / (sigma * sigma));
        let step = blur_pass.direction * offset;
        sum += (read(center + step) + read(center - step)) * weight;
        total += weight * 2.0;
    }
    return sum / total;
}
