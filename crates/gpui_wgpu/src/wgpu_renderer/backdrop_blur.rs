//! Backdrop blurs: what the frame holds under a rounded rectangle, halved a
//! few times, blurred with a separable Gaussian at the smallest size and put
//! back. The frame must be sampleable; the main pass breaks around each
//! group of blurs, as it does around paths.

use bytemuck::{Pod, Zeroable};
use gpui::BackdropBlur;
use std::num::NonZeroU64;

/// The blurred texture is at most this many halvings below the frame.
const MAX_LEVELS: usize = 4;
/// Uniform records one blur takes: its halvings, two Gaussian passes and
/// the composite.
const RECORDS_PER_BLUR: u64 = MAX_LEVELS as u64 + 3;
/// Half floats keep dark gradients from banding once blurred.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Pooled textures grow in these steps, so a blur that moves or resizes
/// every frame does not reallocate every frame.
const POOL_STEP: u32 = 64;

/// Must match `BlurPass` in backdrop_blur.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlurPass {
    origin: [f32; 2],
    scale: f32,
    sigma: f32,
    min_texel: [f32; 2],
    max_texel: [f32; 2],
    texel_size: [f32; 2],
    direction: [f32; 2],
}

/// Must match `Composite` in backdrop_composite.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Composite {
    viewport_size: [f32; 2],
    origin: [f32; 2],
    min_texel: [f32; 2],
    max_texel: [f32; 2],
    texel_size: [f32; 2],
    scale: f32,
    opacity: f32,
    bounds_origin: [f32; 2],
    bounds_size: [f32; 2],
    draw_origin: [f32; 2],
    draw_size: [f32; 2],
    corner_radii: [f32; 4],
}

struct PooledTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl PooledTexture {
    /// A texture of at least `width` by `height`, kept in `slot`.
    fn fit<'a>(
        slot: &'a mut Option<PooledTexture>,
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> &'a PooledTexture {
        let fits = slot.as_ref().is_some_and(|pooled| {
            pooled.texture.width() >= width && pooled.texture.height() >= height
        });
        if !fits {
            let (old_width, old_height) = slot.as_ref().map_or((0, 0), |pooled| {
                (pooled.texture.width(), pooled.texture.height())
            });
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("backdrop_blur_texture"),
                size: wgpu::Extent3d {
                    width: width.max(old_width).next_multiple_of(POOL_STEP),
                    height: height.max(old_height).next_multiple_of(POOL_STEP),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            *slot = Some(PooledTexture { texture, view });
        }
        slot.as_ref().expect("the slot was just filled")
    }
}

fn texel_size(texture: &wgpu::Texture) -> [f32; 2] {
    [1.0 / texture.width() as f32, 1.0 / texture.height() as f32]
}

/// The textures of one blur in a group: one per halving and one for the
/// Gaussian's first direction.
#[derive(Default)]
struct Slot {
    levels: [Option<PooledTexture>; MAX_LEVELS],
    scratch: Option<PooledTexture>,
}

/// A blur ready under the frame, put back once the main pass resumes.
pub(super) struct Blurred(wgpu::BindGroup);

pub(super) struct BackdropBlurs {
    layout: wgpu::BindGroupLayout,
    down: wgpu::RenderPipeline,
    gaussian: wgpu::RenderPipeline,
    clear: wgpu::RenderPipeline,
    add: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    uniforms: wgpu::Buffer,
    alignment: u64,
    /// This frame's uniform records, written before it is submitted.
    staged: Vec<u8>,
    slots: Vec<Slot>,
}

impl BackdropBlurs {
    pub(super) fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("backdrop_blur_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("backdrop_blur_pipeline_layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let passes = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("backdrop_blur_shaders"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../backdrop_blur.wgsl").into()),
        });
        let composite = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("backdrop_composite_shaders"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../backdrop_composite.wgsl").into()),
        });

        let pipeline = |name: &str,
                        module: &wgpu::ShaderModule,
                        vs_entry: &str,
                        fs_entry: &str,
                        format: wgpu::TextureFormat,
                        blend: Option<wgpu::BlendComponent>,
                        topology: wgpu::PrimitiveTopology| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(name),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some(vs_entry),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(fs_entry),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: blend.map(|component| wgpu::BlendState {
                            color: component,
                            alpha: component,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        let triangles = wgpu::PrimitiveTopology::TriangleList;
        let strip = wgpu::PrimitiveTopology::TriangleStrip;
        let down = pipeline(
            "backdrop_blur_down",
            &passes,
            "vs_fullscreen",
            "fs_down",
            FORMAT,
            None,
            triangles,
        );
        let gaussian = pipeline(
            "backdrop_blur_gaussian",
            &passes,
            "vs_fullscreen",
            "fs_gaussian",
            FORMAT,
            None,
            triangles,
        );
        // The frame keeps what the blur does not cover...
        let clear = pipeline(
            "backdrop_blur_clear",
            &composite,
            "vs_composite",
            "fs_composite_clear",
            surface_format,
            Some(wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::Zero,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            }),
            strip,
        );
        // ...and takes the blur where it does.
        let add = pipeline(
            "backdrop_blur_add",
            &composite,
            "vs_composite",
            "fs_composite_add",
            surface_format,
            Some(wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            }),
            strip,
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("backdrop_blur_sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let alignment = u64::from(device.limits().min_uniform_buffer_offset_alignment);

        Self {
            layout,
            down,
            gaussian,
            clear,
            add,
            sampler,
            uniforms: Self::create_uniforms(device, alignment * RECORDS_PER_BLUR * 4),
            alignment,
            staged: Vec::new(),
            slots: Vec::new(),
        }
    }

    fn create_uniforms(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("backdrop_blur_uniforms"),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Makes room for the uniform records of `count` blurs.
    pub(super) fn begin_frame(&mut self, device: &wgpu::Device, count: usize) {
        self.staged.clear();
        let needed = count as u64 * RECORDS_PER_BLUR * self.alignment;
        if self.uniforms.size() < needed {
            self.uniforms = Self::create_uniforms(device, needed.next_power_of_two());
        }
    }

    /// Writes this frame's uniform records; call before submitting it.
    pub(super) fn finish_frame(&mut self, queue: &wgpu::Queue) {
        if !self.staged.is_empty() {
            queue.write_buffer(&self.uniforms, 0, &self.staged);
            self.staged.clear();
        }
    }

    fn push<T: Pod>(&mut self, record: &T) -> u64 {
        let offset = self.staged.len() as u64;
        self.staged.extend_from_slice(bytemuck::bytes_of(record));
        self.staged.resize(
            self.staged.len().next_multiple_of(self.alignment as usize),
            0,
        );
        debug_assert!(self.staged.len() as u64 <= self.uniforms.size());
        offset
    }

    fn bind_group<T>(
        &self,
        device: &wgpu::Device,
        offset: u64,
        source: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("backdrop_blur_bind_group"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniforms,
                        offset,
                        size: NonZeroU64::new(std::mem::size_of::<T>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(source),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn run(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        down: bool,
        record: BlurPass,
        source: &wgpu::TextureView,
        target: &wgpu::TextureView,
        (width, height): (u32, u32),
    ) {
        let offset = self.push(&record);
        let bind_group = self.bind_group::<BlurPass>(device, offset, source);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("backdrop_blur_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_viewport(0.0, 0.0, width as f32, height as f32, 0.0, 1.0);
        pass.set_pipeline(if down { &self.down } else { &self.gaussian });
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Blurs what `frame` holds under each of `blurs`, which must not
    /// overlap one another: one order of the scene.
    pub(super) fn blur(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        frame: &wgpu::Texture,
        blurs: &[BackdropBlur],
    ) -> Vec<Blurred> {
        let frame_size = [frame.width() as f32, frame.height() as f32];
        let frame_view = frame.create_view(&wgpu::TextureViewDescriptor::default());
        let mut slots = std::mem::take(&mut self.slots);
        slots.resize_with(slots.len().max(blurs.len()), Slot::default);

        let mut blurred = Vec::with_capacity(blurs.len());
        for (blur, slot) in blurs.iter().zip(&mut slots) {
            let shown = blur.bounds.intersect(&blur.content_mask.bounds);
            if shown.is_empty() || blur.opacity <= 0.0 {
                continue;
            }
            // The Gaussian reaches two standard deviations: the radius.
            let reach = blur.blur_radius.0;
            let sigma = reach / 2.0;
            let left = (shown.origin.x.0 - reach).floor().max(0.0);
            let top = (shown.origin.y.0 - reach).floor().max(0.0);
            let right = (shown.right().0 + reach).ceil().min(frame_size[0]);
            let bottom = (shown.bottom().0 + reach).ceil().min(frame_size[1]);
            if right <= left || bottom <= top {
                continue;
            }
            // Halve until the Gaussian left is a few texels wide.
            let levels = (sigma / 2.0).log2().floor().clamp(1.0, MAX_LEVELS as f32) as usize;

            let mut size = ((right - left) as u32, (bottom - top) as u32);
            let mut source = frame_view.clone();
            let mut record = BlurPass {
                origin: [left, top],
                scale: 2.0,
                sigma: 0.0,
                min_texel: [0.5, 0.5],
                max_texel: [frame_size[0] - 0.5, frame_size[1] - 0.5],
                texel_size: texel_size(frame),
                direction: [0.0, 0.0],
            };
            for level in &mut slot.levels[..levels] {
                size = (size.0.div_ceil(2), size.1.div_ceil(2));
                let target = PooledTexture::fit(level, device, size.0, size.1);
                self.run(device, encoder, true, record, &source, &target.view, size);
                source = target.view.clone();
                record = BlurPass {
                    origin: [0.0, 0.0],
                    min_texel: [0.5, 0.5],
                    max_texel: [size.0 as f32 - 0.5, size.1 as f32 - 0.5],
                    texel_size: texel_size(&target.texture),
                    ..record
                };
            }
            let smallest = slot.levels[levels - 1]
                .as_ref()
                .expect("every halving has its texture");
            let scratch = PooledTexture::fit(&mut slot.scratch, device, size.0, size.1);
            let shrink = (1u32 << levels) as f32;
            let across = BlurPass {
                scale: 1.0,
                sigma: sigma / shrink,
                direction: [1.0, 0.0],
                ..record
            };
            self.run(device, encoder, false, across, &source, &scratch.view, size);
            let down = BlurPass {
                texel_size: texel_size(&scratch.texture),
                direction: [0.0, 1.0],
                ..across
            };
            self.run(
                device,
                encoder,
                false,
                down,
                &scratch.view,
                &smallest.view,
                size,
            );

            let radii = blur.corner_radii;
            let offset = self.push(&Composite {
                viewport_size: frame_size,
                origin: [left, top],
                min_texel: record.min_texel,
                max_texel: record.max_texel,
                texel_size: record.texel_size,
                scale: 1.0 / shrink,
                opacity: blur.opacity,
                bounds_origin: [blur.bounds.origin.x.0, blur.bounds.origin.y.0],
                bounds_size: [blur.bounds.size.width.0, blur.bounds.size.height.0],
                draw_origin: [shown.origin.x.0, shown.origin.y.0],
                draw_size: [shown.size.width.0, shown.size.height.0],
                corner_radii: [
                    radii.top_left.0,
                    radii.top_right.0,
                    radii.bottom_right.0,
                    radii.bottom_left.0,
                ],
            });
            blurred.push(Blurred(self.bind_group::<Composite>(
                device,
                offset,
                &smallest.view,
            )));
        }

        self.slots = slots;
        blurred
    }

    /// Puts `blurred` back into the frame the main pass draws to.
    pub(super) fn composite(&self, blurred: &[Blurred], pass: &mut wgpu::RenderPass<'_>) {
        for Blurred(bind_group) in blurred {
            pass.set_bind_group(0, bind_group, &[]);
            pass.set_pipeline(&self.clear);
            pass.draw(0..4, 0..1);
            pass.set_pipeline(&self.add);
            pass.draw(0..4, 0..1);
        }
    }
}
