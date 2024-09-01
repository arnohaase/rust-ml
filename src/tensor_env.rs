use std::borrow::Cow;
use std::sync::{Mutex, RwLock};
use rustc_hash::FxHashMap;
use triomphe::Arc;
use wgpu::{BindGroup, BindingResource, ComputePipeline, ShaderModule};
use wgpu::util::DeviceExt;
use crate::dimension::{Dimension, DimensionKind, Dimensions};
use crate::tensor::Tensor;


pub trait TensorEnv {
    type Buffer: Clone;

    fn data_from_buffer(&self, buf: &Self::Buffer) -> Vec<f32>;

    fn create_tensor(&self, dimensions: Dimensions, buf: Vec<f32>) -> Tensor<Self>;

    fn scalar(&self, x: f32) -> Tensor<Self> {
        self.create_tensor(vec![].into(), vec![x])
    }

    fn vector(&self, xs: Vec<f32>, kind: DimensionKind) -> Tensor<Self> {
        self.create_tensor(vec![Dimension {
            len: xs.len(),
            kind,
        }].into(), xs)
    }
}


pub struct BlasEnv {
}

impl TensorEnv for BlasEnv {
    type Buffer = Arc<RwLock<Vec<f32>>>;

    fn data_from_buffer(&self, buf: &Arc<RwLock<Vec<f32>>>) -> Vec<f32> {
        buf.read().unwrap().to_vec()
    }

    fn create_tensor(&self, dimensions: Dimensions, buf: Vec<f32>) -> Tensor<Self> {
        Tensor::create_from_raw(&self, dimensions, Arc::new(RwLock::new(buf)))
    }
}


pub struct WgpuEnv {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    cached_shaders: Mutex<FxHashMap<String, Arc<CachedWgpuShader>>>,
}

impl WgpuEnv {
    pub async fn new() -> WgpuEnv {
        let instance = wgpu::Instance::default(); //TODO configurable
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default()) //TODO configurable
            .await
            .unwrap(); //TODO anyhow?

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor { //TODO configurable
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_defaults(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                },
                None,
            )
            .await
            .unwrap() //TODO anyhow
            ;

        println!("{}", device.limits().max_compute_workgroup_size_x);
        println!("{}", device.limits().max_compute_workgroup_size_y);
        println!("{}", device.limits().max_compute_workgroup_size_z);
        println!("{}", device.limits().max_compute_invocations_per_workgroup);
        println!("{}", device.limits().max_compute_workgroup_storage_size);
        println!("{}", device.limits().max_compute_workgroups_per_dimension);

        WgpuEnv {
            device,
            queue,
            cached_shaders: Default::default(),
        }
    }

    pub fn shader(&self, id: &str, wgsl: impl FnOnce() -> String) -> Arc<CachedWgpuShader> {
        //TODO alternatives to f32
        //TODO workgroup size

        let mut cache = self.cached_shaders.lock().unwrap();
        if let Some(cache_hit) = cache.get(id) {
            return cache_hit.clone();
        }

        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(id),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(&wgsl())),
        });
        let compute_pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(id),
            layout: None,
            module: &module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });
        // let bind_group_layout = pipeline.get_bind_group_layout(0);


        let new_shader = Arc::new(CachedWgpuShader {
            id: id.to_string(),
            module,
            compute_pipeline,
        });

        cache.insert(id.to_string(), new_shader.clone());
        new_shader
    }

    pub fn create_storage_buffer(&self, size: wgpu::BufferAddress) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, //TODO
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }
}

impl TensorEnv for WgpuEnv {
    type Buffer = Arc<wgpu::Buffer>;

    fn data_from_buffer(&self, buf: &Self::Buffer) -> Vec<f32> {
        let size = buf.size();

        //TODO pool?
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None, //TODO
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder =
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None }); //TODO label
        encoder.copy_buffer_to_buffer(buf.as_ref(), 0, &staging_buffer, 0, size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        buffer_slice.map_async(wgpu::MapMode::Read,|v| {}); //TODO error handling

        // Getting a buffer's data is for debugging, not for maximizing multithreaded throughput.

        // We wait until *all* work on the GPU is completed to ensure that the data we copy from the
        //  device is up-to-date
        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout(); //TODO timeout?

        let buf_view = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&buf_view).to_vec();

        drop(buf_view); // to make it explicit that the view must be dropped first
        staging_buffer.unmap();

        result
    }

    fn create_tensor(&self, dimensions: Dimensions, buf: Vec<f32>) -> Tensor<Self> {
        //TODO work from a pool?
        let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None, //TODO
            contents: bytemuck::cast_slice(buf.as_ref()),
            usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST
        });

        Tensor::create_from_raw(&self, dimensions, Arc::new(buf))
    }
}

pub struct CachedWgpuShader {
    pub id: String,
    pub module: ShaderModule,
    pub compute_pipeline: ComputePipeline,
}
impl CachedWgpuShader {
    pub fn bind_group(&self, env: &WgpuEnv, buffers: Vec<BindingResource>) -> BindGroup {
        let bind_group_layout = self.compute_pipeline.get_bind_group_layout(0);

        let entries = buffers.into_iter()
            .enumerate()
            .map(|(i, r)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: r,
            })
            .collect::<Vec<_>>();

        env.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&self.id),
            layout: &bind_group_layout,
            entries: &entries,
        })
    }
}







