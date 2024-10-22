// Constants defining the workgroup size
const WORKGROUP_SIZE: u32 = {workgroup_size};

@group(0) @binding(0) var<storage, read> inputArray: array<f32>;
@group(0) @binding(1) var<storage, read_write> outputSum: array<f32>;

// Shared memory for workgroup reduction
var<workgroup> localSum: array<f32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {

    // Step 1: Each thread loads its corresponding element
    let index = global_id.x;
    let value = inputArray[index];

    // Step 2: Perform local reduction
    localSum[local_id.x] = value;
    workgroupBarrier();

    // Perform parallel reduction within the workgroup
    var step = WORKGROUP_SIZE / 2;
    while (step > 0) {
        if (local_id.x < step) {
            localSum[local_id.x] += localSum[local_id.x + step];
        }
        step = step / 2;
        workgroupBarrier();
    }

    // Step 3: The first thread in the workgroup writes the local sum to the output buffer
    if (local_id.x == 0) {
        atomicAdd(&outputSum, localSum[0]);
    }
}
