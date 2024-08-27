


#[macro_export]
macro_rules! with_all_envs {
    ($env:ident => $f: expr) => {{
        let $env = BlasEnv{};
        $f;

        let $env = block_on(WgpuEnv::new());
        $f;
    }};
}
