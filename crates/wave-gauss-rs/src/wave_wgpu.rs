//! Native cross-platform GPU acoustic wave forward & discrete adjoint engine using `wgpu`.
//! Reuses the verified 4th-order C-PML shaders (`splatForward.wgsl` and `splatAdjoint.wgsl`)
//! natively on Metal (Apple Silicon), Vulkan (Linux/Windows), or DX12.

use bytemuck::{Pod, Zeroable};
use geology_core::{cpml_profiles, ricker_wavelet, CELLS, FD_PAD, H, N};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

const FORWARD_WGSL: &str = include_str!("shaders/splatForward.wgsl");
const ADJOINT_WGSL: &str = include_str!("shaders/splatAdjoint.wgsl");

const RESIDUAL_WGSL: &str = r#"
struct ResidualParams {
  count: u32,
  scale: f32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<uniform> rp: ResidualParams;
@group(0) @binding(1) var<storage, read_write> traces: array<f32>;
@group(0) @binding(2) var<storage, read> observed: array<f32>;
@group(0) @binding(3) var<storage, read_write> partials: array<f32>;

var<workgroup> wg_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_index) li: u32,
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(num_workgroups) nwg: vec3<u32>
) {
  let w = wid.y * nwg.x + wid.x;
  let i = w * 256u + li;
  var e = 0.0;
  if (i < rp.count) {
    let d = traces[i] - observed[i];
    traces[i] = d * rp.scale;
    e = 0.5 * d * d * rp.scale;
  }
  wg_sums[li] = e;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s /= 2u) {
    if (li < s) {
      wg_sums[li] += wg_sums[li + s];
    }
    workgroupBarrier();
  }
  if (li == 0u) {
    partials[w] = wg_sums[0];
  }
}
"#;

const STEP_STRIDE: u64 = 256;
const STEPS_PER_SUBMIT: usize = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct WaveParams {
    nx: u32,
    nz: u32,
    rh: f32,
    dt2: f32,
    nrec: u32,
    nt: u32,
    region: u32,
    shots: u32,
    store_every: u32,
    nsnap: u32,
    store_offset: u32,
    region_adj: u32,
    src: u32,
    rec: u32,
    csr: u32,
    entries: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ResidualParams {
    count: u32,
    scale: f32,
    pad0: u32,
    pad1: u32,
}

pub struct AdjointResult {
    pub loss: f32,
    pub grad_mps: Vec<f32>, // [70 * 70] dJ / d(v in m/s)
}

#[allow(dead_code)]
pub struct NativeWaveEngine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pub shots: usize,
    pub nt: usize,
    pub nrec: usize,
    pub pml: usize,
    pub pad: usize,
    pub nx: usize,
    pub nz: usize,
    pub dt: f32,
    pub peak_hz: f32,
    pub vmin: f32,
    pub vmax: f32,
    pub sources: Vec<[f32; 2]>,
    pub wavelet: Vec<f32>,
    params_forward_buf: wgpu::Buffer,
    params_adjoint_buf: wgpu::Buffer,
    steps_buf: wgpu::Buffer,
    statics_buf: wgpu::Buffer,
    geometry_buf: wgpu::Buffer,
    u0_buf: wgpu::Buffer,
    u1_buf: wgpu::Buffer,
    psi0_buf: wgpu::Buffer,
    psi1_buf: wgpu::Buffer,
    zeta_buf: wgpu::Buffer,
    outputs_buf: wgpu::Buffer,
    l0_buf: wgpu::Buffer,
    l1_buf: wgpu::Buffer,
    lpsi_buf: wgpu::Buffer,
    lzeta_buf: wgpu::Buffer,
    grad_buf: wgpu::Buffer,
    observed_buf: wgpu::Buffer,
    res_params_buf: wgpu::Buffer,
    partials_buf: wgpu::Buffer,
    traces_read_buf: wgpu::Buffer,
    grad_read_buf: wgpu::Buffer,
    partials_read_buf: wgpu::Buffer,
    residual_groups: u32,
    forward_pipeline: wgpu::ComputePipeline,
    forward_bgs_no_store: [wgpu::BindGroup; 2],
    forward_bgs_store: [wgpu::BindGroup; 2],
    adjoint_pipeline: wgpu::ComputePipeline,
    adjoint_bgs: [wgpu::BindGroup; 2],
    residual_pipeline: wgpu::ComputePipeline,
    residual_bg: wgpu::BindGroup,
}

impl NativeWaveEngine {
    pub fn new(
        shots: usize,
        nt: usize,
        dt: f32,
        peak_hz: f32,
        pml: usize,
        vmin: f32,
        vmax: f32,
    ) -> Result<Self, String> {
        pollster::block_on(Self::new_async(shots, nt, dt, peak_hz, pml, vmin, vmax))
    }

    async fn new_async(
        shots: usize,
        nt: usize,
        dt: f32,
        peak_hz: f32,
        pml: usize,
        vmin: f32,
        vmax: f32,
    ) -> Result<Self, String> {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| "Failed to find a suitable GPU adapter via wgpu".to_string())?;

        let adapter_limits = adapter.limits();
        let pad = FD_PAD + pml;
        let nx = N + 2 * pad;
        let nz = N + 2 * pad;
        let cells = nx * nz;
        let nrec = N;
        let nsnap = nt;
        let outputs_words = shots * nrec * nt + shots * nsnap * cells;
        let needed_bytes = (outputs_words * 4) as u64;

        let mut limits = wgpu::Limits::default();
        limits.max_buffer_size = adapter_limits.max_buffer_size.max(needed_bytes + (1 << 20));
        limits.max_storage_buffer_binding_size = adapter_limits
            .max_storage_buffer_binding_size
            .max((needed_bytes as u32) + (1 << 20));

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("wave-gauss-rs-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| format!("Failed to request wgpu device: {e}"))?;

        let xmax = ((N - 1) as f32) * H;
        let sources: Vec<[f32; 2]> = (0..shots)
            .map(|k| {
                let x = if shots == 1 {
                    xmax * 0.5
                } else {
                    (xmax * (k as f32)) / ((shots - 1) as f32)
                };
                [x, H]
            })
            .collect();
        let receivers: Vec<[f32; 2]> = (0..nrec).map(|k| [(k as f32) * H, H]).collect();

        let corners = |pt: [f32; 2]| -> [(u32, f32); 4] {
            let gx = pt[0] / H + (pad as f32);
            let gz = pt[1] / H + (pad as f32);
            let x0 = (gx.floor() as usize).min(nx - 2);
            let z0 = (gz.floor() as usize).min(nz - 2);
            let fx = gx - (x0 as f32);
            let fz = gz - (z0 as f32);
            [
                ((z0 * nx + x0) as u32, (1.0 - fx) * (1.0 - fz)),
                ((z0 * nx + x0 + 1) as u32, fx * (1.0 - fz)),
                (((z0 + 1) * nx + x0) as u32, (1.0 - fx) * fz),
                (((z0 + 1) * nx + x0 + 1) as u32, fx * fz),
            ]
        };

        let mut entries: Vec<(u32, u32, f32)> = Vec::new();
        for (k, &r) in receivers.iter().enumerate() {
            for (cell, w) in corners(r) {
                if w != 0.0 {
                    entries.push((cell, k as u32, w));
                }
            }
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

        let src_off = 0usize;
        let rec_off = 8 * shots;
        let csr_off = rec_off + 8 * nrec;
        let list_off = csr_off + cells + 1;
        let mut geom_words = vec![0u32; list_off + 2 * entries.len()];

        for (k, &s) in sources.iter().enumerate() {
            for (j, (cell, w)) in corners(s).into_iter().enumerate() {
                geom_words[src_off + 4 * k + j] = cell;
                geom_words[src_off + 4 * shots + 4 * k + j] = w.to_bits();
            }
        }
        for (k, &r) in receivers.iter().enumerate() {
            for (j, (cell, w)) in corners(r).into_iter().enumerate() {
                geom_words[rec_off + 4 * k + j] = cell;
                geom_words[rec_off + 4 * nrec + 4 * k + j] = w.to_bits();
            }
        }
        let mut e_idx = 0usize;
        for cell in 0..=cells {
            while e_idx < entries.len() && (entries[e_idx].0 as usize) < cell {
                e_idx += 1;
            }
            geom_words[csr_off + cell] = e_idx as u32;
        }
        for (j, &(_, k, w)) in entries.iter().enumerate() {
            geom_words[list_off + 2 * j] = k;
            geom_words[list_off + 2 * j + 1] = w.to_bits();
        }

        let geometry_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("geometry"),
            contents: bytemuck::cast_slice(&geom_words),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let trace_words = (shots * nrec * nt) as u32;
        let base_params = WaveParams {
            nx: nx as u32,
            nz: nz as u32,
            rh: 1.0 / H,
            dt2: dt * dt,
            nrec: nrec as u32,
            nt: nt as u32,
            region: (pml + 2 * FD_PAD) as u32,
            shots: shots as u32,
            store_every: 0,
            nsnap: 0,
            store_offset: trace_words,
            region_adj: (pml + 3 * FD_PAD) as u32,
            src: src_off as u32,
            rec: rec_off as u32,
            csr: csr_off as u32,
            entries: list_off as u32,
        };
        let adjoint_params = WaveParams {
            store_every: 1,
            nsnap: nsnap as u32,
            ..base_params
        };

        let params_forward_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_forward"),
            contents: bytemuck::bytes_of(&base_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let params_adjoint_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params_adjoint"),
            contents: bytemuck::bytes_of(&adjoint_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let step_words_len = nt * (STEP_STRIDE as usize / 4);
        let mut step_words = vec![0u32; step_words_len];
        for t in 0..nt {
            step_words[t * (STEP_STRIDE as usize / 4)] = t as u32;
        }
        let steps_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("steps"),
            contents: bytemuck::cast_slice(&step_words),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let (mut ax, mut bx, mut dbx) = (vec![0.0f32; nx], vec![0.0f32; nx], vec![0.0f32; nx]);
        let (mut az, mut bz, mut dbz) = (vec![0.0f32; nz], vec![0.0f32; nz], vec![0.0f32; nz]);
        cpml_profiles(nx, H, dt, pml, vmax, peak_hz, &mut ax, &mut bx, &mut dbx);
        cpml_profiles(nz, H, dt, pml, vmax, peak_hz, &mut az, &mut bz, &mut dbz);

        let mut wavelet = vec![0.0f32; nt];
        ricker_wavelet(nt, dt, peak_hz, &mut wavelet);

        let mut statics_tail = Vec::with_capacity(3 * (nx + nz) + nt);
        statics_tail.extend_from_slice(&ax);
        statics_tail.extend_from_slice(&bx);
        statics_tail.extend_from_slice(&dbx);
        statics_tail.extend_from_slice(&az);
        statics_tail.extend_from_slice(&bz);
        statics_tail.extend_from_slice(&dbz);
        statics_tail.extend_from_slice(&wavelet);

        let statics_total_words = cells + statics_tail.len();
        let statics_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("statics"),
            size: (statics_total_words * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &statics_buf,
            (cells * 4) as u64,
            bytemuck::cast_slice(&statics_tail),
        );

        let storage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let make_buf = |label: &str, words: usize| -> wgpu::Buffer {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (words.max(4) * 4) as u64,
                usage: storage_rw,
                mapped_at_creation: false,
            })
        };

        let u0_buf = make_buf("u0", shots * cells);
        let u1_buf = make_buf("u1", shots * cells);
        let psi0_buf = make_buf("psi0", shots * 2 * cells);
        let psi1_buf = make_buf("psi1", shots * 2 * cells);
        let zeta_buf = make_buf("zeta", shots * 2 * cells);
        let outputs_buf = make_buf("outputs", outputs_words);

        let l0_buf = make_buf("l0", shots * cells);
        let l1_buf = make_buf("l1", shots * cells);
        let lpsi_buf = make_buf("lpsi", 2 * shots * 2 * cells);
        let lzeta_buf = make_buf("lzeta", 2 * shots * 2 * cells);
        let grad_buf = make_buf("grad", 2 * shots * cells);
        let observed_buf = make_buf("observed", shots * nrec * nt);

        let residual_groups = (trace_words + 255) / 256;
        let res_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("res_params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let partials_buf = make_buf("partials", residual_groups as usize);

        let map_read = wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST;
        let traces_read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("traces_read"),
            size: (shots * nrec * nt * 4) as u64,
            usage: map_read,
            mapped_at_creation: false,
        });
        let grad_read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("grad_read"),
            size: (shots * cells * 4) as u64,
            usage: map_read,
            mapped_at_creation: false,
        });
        let partials_read_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("partials_read"),
            size: (residual_groups.max(4) as u64) * 4,
            usage: map_read,
            mapped_at_creation: false,
        });

        let bgl_entry = |binding: u32, ty: wgpu::BufferBindingType, dynamic: bool| {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty,
                    has_dynamic_offset: dynamic,
                    min_binding_size: None,
                },
                count: None,
            }
        };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        let rw = wgpu::BufferBindingType::Storage { read_only: false };
        let uni = wgpu::BufferBindingType::Uniform;

        let forward_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("forward_bgl"),
            entries: &[
                bgl_entry(0, uni, false),
                bgl_entry(1, uni, true),
                bgl_entry(2, ro, false),
                bgl_entry(3, ro, false),
                bgl_entry(4, ro, false),
                bgl_entry(5, rw, false),
                bgl_entry(6, ro, false),
                bgl_entry(7, rw, false),
                bgl_entry(8, rw, false),
                bgl_entry(9, rw, false),
            ],
        });
        let forward_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("forward_pl"),
            bind_group_layouts: &[&forward_bgl],
            push_constant_ranges: &[],
        });
        let forward_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splatForward"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(FORWARD_WGSL)),
        });
        let forward_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("forward_pipeline"),
            layout: Some(&forward_pl),
            module: &forward_sm,
            entry_point: Some("update"),
            compilation_options: Default::default(),
            cache: None,
        });

        let make_forward_bg = |params_b: &wgpu::Buffer,
                               cur_b: &wgpu::Buffer,
                               prev_b: &wgpu::Buffer,
                               psi_b: &wgpu::Buffer,
                               psi_next_b: &wgpu::Buffer|
         -> wgpu::BindGroup {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("forward_bg"),
                layout: &forward_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &steps_buf,
                            offset: 0,
                            size: std::num::NonZeroU64::new(16),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: statics_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: geometry_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: cur_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: prev_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: psi_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: psi_next_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: zeta_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: outputs_buf.as_entire_binding(),
                    },
                ],
            })
        };

        let forward_bgs_no_store = [
            make_forward_bg(&params_forward_buf, &u0_buf, &u1_buf, &psi0_buf, &psi1_buf),
            make_forward_bg(&params_forward_buf, &u1_buf, &u0_buf, &psi1_buf, &psi0_buf),
        ];
        let forward_bgs_store = [
            make_forward_bg(&params_adjoint_buf, &u0_buf, &u1_buf, &psi0_buf, &psi1_buf),
            make_forward_bg(&params_adjoint_buf, &u1_buf, &u0_buf, &psi1_buf, &psi0_buf),
        ];

        let adjoint_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("adjoint_bgl"),
            entries: &[
                bgl_entry(0, uni, false),
                bgl_entry(1, uni, true),
                bgl_entry(2, ro, false),
                bgl_entry(3, ro, false),
                bgl_entry(4, ro, false),
                bgl_entry(5, rw, false),
                bgl_entry(6, rw, false),
                bgl_entry(7, rw, false),
                bgl_entry(8, ro, false),
                bgl_entry(9, rw, false),
            ],
        });
        let adjoint_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("adjoint_pl"),
            bind_group_layouts: &[&adjoint_bgl],
            push_constant_ranges: &[],
        });
        let adjoint_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("splatAdjoint"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(ADJOINT_WGSL)),
        });
        let adjoint_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("adjoint_pipeline"),
            layout: Some(&adjoint_pl),
            module: &adjoint_sm,
            entry_point: Some("adjoint"),
            compilation_options: Default::default(),
            cache: None,
        });

        let make_adjoint_bg = |cur_b: &wgpu::Buffer, prev_b: &wgpu::Buffer| -> wgpu::BindGroup {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("adjoint_bg"),
                layout: &adjoint_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_adjoint_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &steps_buf,
                            offset: 0,
                            size: std::num::NonZeroU64::new(16),
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: statics_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: geometry_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: cur_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: prev_b.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: lpsi_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: lzeta_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: outputs_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: grad_buf.as_entire_binding(),
                    },
                ],
            })
        };
        let adjoint_bgs = [
            make_adjoint_bg(&l0_buf, &l1_buf),
            make_adjoint_bg(&l1_buf, &l0_buf),
        ];

        let residual_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("residual_bgl"),
            entries: &[
                bgl_entry(0, uni, false),
                bgl_entry(1, rw, false),
                bgl_entry(2, ro, false),
                bgl_entry(3, rw, false),
            ],
        });
        let residual_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("residual_pl"),
            bind_group_layouts: &[&residual_bgl],
            push_constant_ranges: &[],
        });
        let residual_sm = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("residual_sm"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(RESIDUAL_WGSL)),
        });
        let residual_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("residual_pipeline"),
            layout: Some(&residual_pl),
            module: &residual_sm,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let residual_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("residual_bg"),
            layout: &residual_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: res_params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: outputs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: observed_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: partials_buf.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            device,
            queue,
            shots,
            nt,
            nrec,
            pml,
            pad,
            nx,
            nz,
            dt,
            peak_hz,
            vmin,
            vmax,
            sources,
            wavelet,
            params_forward_buf,
            params_adjoint_buf,
            steps_buf,
            statics_buf,
            geometry_buf,
            u0_buf,
            u1_buf,
            psi0_buf,
            psi1_buf,
            zeta_buf,
            outputs_buf,
            l0_buf,
            l1_buf,
            lpsi_buf,
            lzeta_buf,
            grad_buf,
            observed_buf,
            res_params_buf,
            partials_buf,
            traces_read_buf,
            grad_read_buf,
            partials_read_buf,
            residual_groups,
            forward_pipeline,
            forward_bgs_no_store,
            forward_bgs_store,
            adjoint_pipeline,
            adjoint_bgs,
            residual_pipeline,
            residual_bg,
        })
    }

    fn pad_velocity(&self, vel_mps: &[f32]) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(vel_mps.len(), CELLS);
        let cells = self.nx * self.nz;
        let mut padded = vec![0.0f32; cells];
        let mut clamp_mask = vec![0.0f32; cells];
        for z in 0..self.nz {
            let vz = (z as isize - self.pad as isize).clamp(0, (N - 1) as isize) as usize;
            for x in 0..self.nx {
                let vx = (x as isize - self.pad as isize).clamp(0, (N - 1) as isize) as usize;
                let v = vel_mps[vz * N + vx];
                let idx = z * self.nx + x;
                clamp_mask[idx] = if v > self.vmin && v < self.vmax {
                    1.0
                } else {
                    0.0
                };
                padded[idx] = v.clamp(self.vmin, self.vmax);
            }
        }
        (padded, clamp_mask)
    }

    fn read_buffer(&self, src: &wgpu::Buffer, staging: &wgpu::Buffer, byte_len: u64) -> Vec<f32> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("read_encoder"),
            });
        encoder.copy_buffer_to_buffer(src, 0, staging, 0, byte_len);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(0..byte_len);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let out: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        out
    }

    /// Update the active source coordinates `[shots, 2]` in meters on the GPU geometry buffer.
    pub fn set_sources(&mut self, sources: &[[f32; 2]]) {
        assert_eq!(sources.len(), self.shots);
        let mut src_words = vec![0u32; 8 * self.shots];
        for (k, &pt) in sources.iter().enumerate() {
            let gx = pt[0] / H + (self.pad as f32);
            let gz = pt[1] / H + (self.pad as f32);
            let x0 = (gx.floor() as usize).min(self.nx - 2);
            let z0 = (gz.floor() as usize).min(self.nz - 2);
            let fx = gx - (x0 as f32);
            let fz = gz - (z0 as f32);
            let corners = [
                ((z0 * self.nx + x0) as u32, (1.0 - fx) * (1.0 - fz)),
                ((z0 * self.nx + x0 + 1) as u32, fx * (1.0 - fz)),
                (((z0 + 1) * self.nx + x0) as u32, (1.0 - fx) * fz),
                (((z0 + 1) * self.nx + x0 + 1) as u32, fx * fz),
            ];
            for (j, (cell, w)) in corners.into_iter().enumerate() {
                src_words[4 * k + j] = cell;
                src_words[4 * self.shots + 4 * k + j] = w.to_bits();
            }
        }
        self.queue
            .write_buffer(&self.geometry_buf, 0, bytemuck::cast_slice(&src_words));
        self.sources = sources.to_vec();
    }

    /// Replace the source wavelet `[nt]` in the GPU statics buffer.
    pub fn set_wavelet(&mut self, wavelet: &[f32]) {
        assert_eq!(wavelet.len(), self.nt);
        let cells = self.nx * self.nz;
        let offset = ((cells + 3 * (self.nx + self.nz)) * 4) as u64;
        self.queue
            .write_buffer(&self.statics_buf, offset, bytemuck::cast_slice(wavelet));
        self.wavelet = wavelet.to_vec();
    }

    /// Simulate all shots in parallel on GPU for a 70x70 velocity model in m/s.
    /// Returns `[shots * nt * nrec]` float32 traces.
    pub fn simulate(&self, vel_mps: &[f32]) -> Vec<f32> {
        let (padded_vel, _) = self.pad_velocity(vel_mps);
        self.queue
            .write_buffer(&self.statics_buf, 0, bytemuck::cast_slice(&padded_vel));

        let wx = (self.nx as u32).div_ceil(16);
        let wz = (self.nz as u32).div_ceil(16);
        let shots_u32 = self.shots as u32;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("forward_clear"),
            });
        encoder.clear_buffer(&self.u0_buf, 0, None);
        encoder.clear_buffer(&self.u1_buf, 0, None);
        encoder.clear_buffer(&self.psi0_buf, 0, None);
        encoder.clear_buffer(&self.psi1_buf, 0, None);
        encoder.clear_buffer(&self.zeta_buf, 0, None);
        self.queue.submit(Some(encoder.finish()));

        for first in (0..self.nt).step_by(STEPS_PER_SUBMIT) {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("forward_chunk"),
                });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("forward_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.forward_pipeline);
                let end = (first + STEPS_PER_SUBMIT).min(self.nt);
                for t in first..end {
                    let offset = (t as u64 * STEP_STRIDE) as u32;
                    pass.set_bind_group(0, &self.forward_bgs_no_store[t & 1], &[offset]);
                    pass.dispatch_workgroups(wx, wz, shots_u32);
                }
            }
            self.queue.submit(Some(enc.finish()));
        }

        let trace_bytes = (self.shots * self.nt * self.nrec * 4) as u64;
        self.read_buffer(&self.outputs_buf, &self.traces_read_buf, trace_bytes)
    }

    /// Run exact forward + discrete adjoint pass across all shots on GPU.
    /// Returns normalized loss `0.5 * sum(p - d)^2 / sum(d^2)` and folded `[70 * 70]` gradient `dJ / d(v in m/s)`.
    pub fn evaluate_adjoint(&self, vel_mps: &[f32], observed: &[f32]) -> AdjointResult {
        let trace_len = self.shots * self.nt * self.nrec;
        assert_eq!(observed.len(), trace_len);

        let energy: f64 = observed.iter().map(|&v| (v as f64) * (v as f64)).sum();
        assert!(energy > 0.0, "Observed traces have zero energy");
        let scale = (1.0 / energy) as f32;

        let (padded_vel, clamp_mask) = self.pad_velocity(vel_mps);
        self.queue
            .write_buffer(&self.statics_buf, 0, bytemuck::cast_slice(&padded_vel));
        self.queue
            .write_buffer(&self.observed_buf, 0, bytemuck::cast_slice(observed));

        let rp = ResidualParams {
            count: trace_len as u32,
            scale,
            pad0: 0,
            pad1: 0,
        };
        self.queue
            .write_buffer(&self.res_params_buf, 0, bytemuck::bytes_of(&rp));

        let wx = (self.nx as u32).div_ceil(16);
        let wz = (self.nz as u32).div_ceil(16);
        let shots_u32 = self.shots as u32;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("adjoint_clear_forward"),
            });
        encoder.clear_buffer(&self.u0_buf, 0, None);
        encoder.clear_buffer(&self.u1_buf, 0, None);
        encoder.clear_buffer(&self.psi0_buf, 0, None);
        encoder.clear_buffer(&self.psi1_buf, 0, None);
        encoder.clear_buffer(&self.zeta_buf, 0, None);
        self.queue.submit(Some(encoder.finish()));

        for first in (0..self.nt).step_by(STEPS_PER_SUBMIT) {
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("forward_store_chunk"),
                });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("forward_store_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.forward_pipeline);
                let end = (first + STEPS_PER_SUBMIT).min(self.nt);
                for t in first..end {
                    let offset = (t as u64 * STEP_STRIDE) as u32;
                    pass.set_bind_group(0, &self.forward_bgs_store[t & 1], &[offset]);
                    pass.dispatch_workgroups(wx, wz, shots_u32);
                }
            }
            self.queue.submit(Some(enc.finish()));
        }

        let mut res_enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("residual_and_clear_adj"),
            });
        {
            let mut pass = res_enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("residual_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.residual_pipeline);
            pass.set_bind_group(0, &self.residual_bg, &[]);
            let gx = self.residual_groups.min(65535);
            let gy = self.residual_groups.div_ceil(65535);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        res_enc.clear_buffer(&self.l0_buf, 0, None);
        res_enc.clear_buffer(&self.l1_buf, 0, None);
        res_enc.clear_buffer(&self.lpsi_buf, 0, None);
        res_enc.clear_buffer(&self.lzeta_buf, 0, None);
        res_enc.clear_buffer(&self.grad_buf, 0, None);
        self.queue.submit(Some(res_enc.finish()));

        let mut last = (self.nt as isize) - 1;
        while last >= 0 {
            let stop = (last - (STEPS_PER_SUBMIT as isize)).max(-1);
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("adjoint_chunk"),
                });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("adjoint_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.adjoint_pipeline);
                let mut t = last;
                while t > stop {
                    let tu = t as usize;
                    let q = (self.nt - 1 - tu) & 1;
                    let offset = (tu as u64 * STEP_STRIDE) as u32;
                    pass.set_bind_group(0, &self.adjoint_bgs[q], &[offset]);
                    pass.dispatch_workgroups(wx, wz, shots_u32);
                    t -= 1;
                }
            }
            self.queue.submit(Some(enc.finish()));
            last = stop;
        }

        let partials = self.read_buffer(
            &self.partials_buf,
            &self.partials_read_buf,
            (self.residual_groups as u64) * 4,
        );
        let loss: f64 = partials.iter().map(|&x| x as f64).sum();

        let cells = self.nx * self.nz;
        let raw_grad = self.read_buffer(
            &self.grad_buf,
            &self.grad_read_buf,
            (self.shots * cells * 4) as u64,
        );

        let mut padded_grad = vec![0.0f32; cells];
        for c in 0..cells {
            let mut sum_s = 0.0f32;
            for s in 0..self.shots {
                sum_s += raw_grad[s * cells + c];
            }
            padded_grad[c] = sum_s * clamp_mask[c];
        }

        let grad_mps = fold_node_gradient(&padded_grad, self.nx, self.nz, self.pad);
        AdjointResult {
            loss: loss as f32,
            grad_mps,
        }
    }
}

/// Fold padded-grid adjoint sensitivities onto the 70x70 visible grid nodes whose values the absorbing frame repeats.
/// Matches `foldNodeGradient` in `src/refinement/objective.ts`.
pub fn fold_node_gradient(padded: &[f32], nx: usize, nz: usize, pad: usize) -> Vec<f32> {
    assert_eq!(nx, N + 2 * pad);
    assert_eq!(nz, N + 2 * pad);
    let mut sum = vec![0.0f64; CELLS];
    for z in 0..nz {
        let vz = (z as isize - pad as isize).clamp(0, (N - 1) as isize) as usize;
        for x in 0..nx {
            let vx = (x as isize - pad as isize).clamp(0, (N - 1) as isize) as usize;
            sum[vz * N + vx] += padded[z * nx + x] as f64;
        }
    }
    sum.into_iter().map(|v| v as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use geology_core::simulate_shot_cpu;

    fn relative_l2_diff_pct(a: &[f32], b: &[f32]) -> f32 {
        let mut diff_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        for (&x, &y) in a.iter().zip(b.iter()) {
            let d = (x as f64) - (y as f64);
            diff_sq += d * d;
            ref_sq += (y as f64) * (y as f64);
        }
        (100.0 * (diff_sq / ref_sq.max(1e-30)).sqrt()) as f32
    }

    #[test]
    fn test_wgpu_forward_matches_cpu_reference() {
        let engine = NativeWaveEngine::new(5, 2001, 0.0005, 15.0, 20, 1000.0, 5000.0)
            .expect("wgpu engine creation");
        let mut vel = vec![2500.0f32; CELLS];
        for z in 35..70 {
            for x in 0..70 {
                vel[z * 70 + x] = 3200.0;
            }
        }
        let gpu_traces = engine.simulate(&vel);
        let mut cpu_shot0 = vec![0.0f32; 2001 * 70];
        simulate_shot_cpu(
            &vel,
            H,
            0.0005,
            2001,
            20,
            1000.0,
            5000.0,
            15.0,
            0.0,
            H,
            &mut cpu_shot0,
        );
        let diff_pct = relative_l2_diff_pct(&gpu_traces[0..2001 * 70], &cpu_shot0);
        assert!(
            diff_pct < 0.05,
            "GPU vs CPU relative L2 diff too large: {}%",
            diff_pct
        );
    }

    #[test]
    fn test_wgpu_adjoint_finite_difference() {
        let engine = NativeWaveEngine::new(2, 501, 0.0005, 15.0, 20, 1000.0, 5000.0)
            .expect("wgpu engine creation");
        let mut true_vel = vec![2500.0f32; CELLS];
        let mut cur_vel = vec![2500.0f32; CELLS];
        let mut dir = vec![0.0f32; CELLS];
        for z in 0..N {
            for x in 0..N {
                let xf = x as f32;
                let zf = z as f32;
                let g1 = (-((xf - 8.0).powi(2) + (zf - 9.0).powi(2)) / 50.0).exp();
                let g2 = (-((xf - 56.0).powi(2) + (zf - 10.0).powi(2)) / 50.0).exp();
                true_vel[z * N + x] += 320.0 * g1 - 230.0 * g2;
                cur_vel[z * N + x] += 180.0 * g1 - 120.0 * g2;
                dir[z * N + x] = (-((xf - 7.0).powi(2) + (zf - 7.0).powi(2)) / 150.0).exp()
                    + 0.3 * (-((xf - 58.0).powi(2) + (zf - 12.0).powi(2)) / 100.0).exp();
            }
        }
        let observed = engine.simulate(&true_vel);
        let adj = engine.evaluate_adjoint(&cur_vel, &observed);

        let eps = 1.0f32;
        let v_plus: Vec<f32> = cur_vel
            .iter()
            .zip(&dir)
            .map(|(v, d)| v + eps * d)
            .collect();
        let v_minus: Vec<f32> = cur_vel
            .iter()
            .zip(&dir)
            .map(|(v, d)| v - eps * d)
            .collect();

        let loss_plus = engine.evaluate_adjoint(&v_plus, &observed).loss as f64;
        let loss_minus = engine.evaluate_adjoint(&v_minus, &observed).loss as f64;
        let fd_slope = (loss_plus - loss_minus) / (2.0 * eps as f64);
        let adj_slope: f64 = adj
            .grad_mps
            .iter()
            .zip(&dir)
            .map(|(&g, &d)| (g as f64) * (d as f64))
            .sum();

        let rel_err = (fd_slope - adj_slope).abs() / fd_slope.abs().max(1e-12);
        assert!(
            rel_err < 0.03,
            "Adjoint directional derivative mismatch: fd={} adj={} rel_err={}",
            fd_slope,
            adj_slope,
            rel_err
        );
    }
}
