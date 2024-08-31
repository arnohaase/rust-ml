struct Data {
    values: array<f32>,
};

@group(0) @binding(0) var<storage, read> a: Data;
@group(0) @binding(1) var<storage, read> b: Data;
@group(0) @binding(2) var<storage, read_write> result: Data;

@compute @workgroup_size({chunk_size})
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let simple_index = id.x;
    let chunk_no = id.y;
    let chunk_base = chunk_no * {chunk_size};

    for(var i: u32 = 0; i<{n}; i=i+1) {
        let chunked_index = chunk_base + simple_index * {n} + i;
        result.values[chunked_index] = a.values[chunked_index] + b.values[simple_index];
    }
}
