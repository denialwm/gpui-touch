// Backdrop blur, last step: the blurred texture put back into the frame,
// inside the rounded rectangle.

struct Composite {
    viewport_size: vec2<f32>,
    // The blurred texel a frame pixel reads: (position - origin) * scale.
    origin: vec2<f32>,
    min_texel: vec2<f32>,
    max_texel: vec2<f32>,
    texel_size: vec2<f32>,
    scale: f32,
    opacity: f32,
    bounds_origin: vec2<f32>,
    bounds_size: vec2<f32>,
    draw_origin: vec2<f32>,
    draw_size: vec2<f32>,
    // Top left, top right, bottom right, bottom left.
    corner_radii: vec4<f32>,
}

@group(0) @binding(0) var<uniform> composite: Composite;
@group(0) @binding(1) var t_source: texture_2d<f32>;
@group(0) @binding(2) var s_source: sampler;

struct CompositeVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) frame_position: vec2<f32>,
}

@vertex
fn vs_composite(@builtin(vertex_index) vertex_id: u32) -> CompositeVarying {
    let unit = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let frame_position = composite.draw_origin + unit * composite.draw_size;
    var out: CompositeVarying;
    out.position = vec4<f32>(
        frame_position / composite.viewport_size * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0),
        0.0,
        1.0,
    );
    out.frame_position = frame_position;
    return out;
}

// How much of the pixel the rounded rectangle covers, times the opacity.
fn coverage(point: vec2<f32>) -> f32 {
    let half_size = composite.bounds_size / 2.0;
    let center_to_point = point - (composite.bounds_origin + half_size);
    let radii = composite.corner_radii;
    var radius: f32;
    if (center_to_point.x < 0.0) {
        radius = select(radii.w, radii.x, center_to_point.y < 0.0);
    } else {
        radius = select(radii.z, radii.y, center_to_point.y < 0.0);
    }
    let corner_center_to_point = abs(center_to_point) - half_size + radius;
    let distance = length(max(vec2<f32>(0.0), corner_center_to_point))
        + min(0.0, max(corner_center_to_point.x, corner_center_to_point.y))
        - radius;
    return saturate(0.5 - distance) * composite.opacity;
}

// The blur replaces what is under it: one draw takes the covered share of
// the frame away, the next adds that share of the blur back.
@fragment
fn fs_composite_clear(input: CompositeVarying) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 0.0, 0.0, coverage(input.frame_position));
}

@fragment
fn fs_composite_add(input: CompositeVarying) -> @location(0) vec4<f32> {
    let texel = clamp(
        (input.frame_position - composite.origin) * composite.scale,
        composite.min_texel,
        composite.max_texel,
    );
    let blurred = textureSampleLevel(t_source, s_source, texel * composite.texel_size, 0.0);
    return blurred * coverage(input.frame_position);
}
