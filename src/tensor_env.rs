use std::sync::RwLock;
use triomphe::Arc;
use wgpu::util::DeviceExt;
use crate::tensor::{Dimension, DimensionKind, Tensor};


pub trait TensorEnv {
    type Buffer: Clone;

    fn data_from_buffer(&self, buf: &Self::Buffer) -> Vec<f32>;

    fn create_tensor(&self, dimensions: Vec<Dimension>, buf: Vec<f32>) -> Tensor<Self>;

    fn scalar(&self, x: f32) -> Tensor<Self> {
        self.create_tensor(vec![], vec![x])
    }

    fn vector(&self, xs: Vec<f32>, kind: DimensionKind) -> Tensor<Self> {
        self.create_tensor(vec![Dimension {
            len: xs.len(),
            kind,
        }], xs)
    }
}


pub struct BlasEnv {
}

impl TensorEnv for BlasEnv {
    type Buffer = Arc<RwLock<Vec<f32>>>;

    fn data_from_buffer(&self, buf: &Arc<RwLock<Vec<f32>>>) -> Vec<f32> {
        buf.read().unwrap().to_vec()
    }

    fn create_tensor(&self, dimensions: Vec<Dimension>, buf: Vec<f32>) -> Tensor<Self> {
        Tensor::create_from_raw(&self, dimensions, Arc::new(RwLock::new(buf)))
    }


}


pub struct WgpuEnv {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
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


        // let mut cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        //     label: None,
        //     source: wgpu::ShaderSource::Wgsl(Cow::Borrowed("asdf")),
        // });

        // let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        //     label: None,
        //     layout: None,
        //     module: &cs_module,
        //     entry_point: "main",
        //     compilation_options: Default::default(),
        //     cache: None,
        // });

        WgpuEnv {
            device,
            queue,
        }

        // todo!()
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

    fn create_tensor(&self, dimensions: Vec<Dimension>, buf: Vec<f32>) -> Tensor<Self> {
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








