use std::borrow::Cow;
use std::sync::RwLock;
use triomphe::Arc;
use crate::tensor::{Dimension, DimensionKind, new_tensor_id, Tensor};


pub trait TensorEnv {
    type Buffer: Clone;

    fn data_from_buffer(buf: &Self::Buffer) -> Vec<f64>;

    fn create_tensor(&self, dimensions: Vec<Dimension>, buf: Vec<f64>) -> Tensor<Self>;

    fn scalar(&self, x: f64) -> Tensor<Self> {
        self.create_tensor(vec![], vec![x])
    }

    fn vector(&self, xs: Vec<f64>, kind: DimensionKind) -> Tensor<Self> {
        self.create_tensor(vec![Dimension {
            len: xs.len(),
            kind,
        }], xs)
    }
}


pub struct BlasEnv {
}

impl TensorEnv for BlasEnv {
    type Buffer = Arc<RwLock<Vec<f64>>>;

    fn data_from_buffer(buf: &Arc<RwLock<Vec<f64>>>) -> Vec<f64> {
        buf.read().unwrap().to_vec()
    }

    fn create_tensor(&self, dimensions: Vec<Dimension>, buf: Vec<f64>) -> Tensor<Self> {
        Tensor::create_from_raw(&self, dimensions, Arc::new(RwLock::new(buf)))
    }


}


pub struct WgpuEnv {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl WgpuEnv {
    pub async fn new() -> Option<WgpuEnv> {
        let instance = wgpu::Instance::default(); //TODO configurable
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default()) //TODO configurable
            .await?; //TODO anyhow?

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


        let mut cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed("asdf")),
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &cs_module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });




        WgpuEnv {
            device,
            queue,
        };

        todo!()
    }
}

impl TensorEnv for WgpuEnv {
    type Buffer = ();

    fn data_from_buffer(buf: &Self::Buffer) -> Vec<f64> {
        todo!()
    }

    fn create_tensor(&self, dimensions: Vec<Dimension>, buf: Vec<f64>) -> Tensor<Self> {
        todo!()
    }
}








