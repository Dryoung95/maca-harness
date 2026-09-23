#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

// Print, for one warp of 64 threads, which element index of the fragment
// each thread holds after load_matrix_sync. If the MACA implementation
// assumed a 32-thread warp, thread t's elements will map to the wrong
// (row, col) and the layout will be visibly scrambled.
__global__ void dump_layout_kernel(const __half* src, int n) {
    __shared__ __half tile[16][16];
    const int tid = threadIdx.x;
    if (tid < 256) {
        int r = tid / 16, c = tid % 16;
        tile[r][c] = __float2half((float)(r * 16 + c));
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::load_matrix_sync(a, &tile[0][0], 16);

    if (tid < 8) {
        printf("tid %2d:", tid);
        for (int i = 0; i < 8; i++) printf(" %g", (float)a.x[i]);
        printf("\n");
    }
}

int main() {
    dump_layout_kernel<<<1, 64>>>(nullptr, 16);
    cudaDeviceSynchronize();
    return 0;
}
