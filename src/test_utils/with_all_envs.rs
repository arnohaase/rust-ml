


#[macro_export]
macro_rules! with_all_envs {
    ($env:ident => $f: expr) => {{
        let $env = $crate::tensor_env::BlasEnv{};
        $f;

        let $env = ::async_std::task::block_on($crate::tensor_env::WgpuEnv::new());
        $f;
    }};
}
