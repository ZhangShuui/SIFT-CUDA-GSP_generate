#include <iostream>
#include <chrono>
using namespace std;
#define BLOCK_SIZE 32
#define BLOCK_STRIDE 32

const int MAX = 4096;
float* gauss;
float* gauss_h;
//GaussDePyramid[layer][S+2][len][len]

int n;
int gauss_size(int n){
    int layer=0;
    int len=n;
    int length = 0;
    while (n){
        layer++;
        n/=2;
        length += pow(n, 2) * 5;
    }
    return length;
}




__global__ void gauss_de_pyramid_initialize(int length,float* gauss){
    int threadId = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
                   + threadIdx.z * blockDim.y * blockDim.x
                   + threadIdx.y * blockDim.x + threadIdx.x;
    if (threadId < length)
        gauss[threadId] =  1.0;
}
__global__ void gauss_de_pyramid_generate(int length, float* gauss){
    const float sigma = 2.0;
    const float PI = 3.1415926;
    const int S = 2;
    int threadId = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
                   + threadIdx.z * blockDim.y * blockDim.x
                   + threadIdx.y * blockDim.x + threadIdx.x;
    //GaussDePyramid[layer][S+2][len][len]
    int layer = 0;
    int len = length;
    while (len){
        layer++;
        len /= 2;
    }
    int my_id = threadId;
    int my_layer = 0;
    len = length;
    if (threadId >= length){
        return;
    }
    while (my_id > (3 + S) * len * len){
        my_id -= (3 + S) * len * len;
        len /= 2;
        my_layer ++;
    };
    //确定本线程对应的层数
    int my_len = len;

    int my_S = my_id / (len * len);
    //确定本线程对应的S

    int my_r = my_id % (len * len) / len;
    //确定本线程对应的行

    int my_c = my_id % (len * len) % len;
    //确定本线程对应列

    float sig=sigma/(my_S+1);
    gauss[threadId] *= exp(-(my_r-len)*(my_r-len)/(2*sig*sig))/(sig*sqrt(2*PI));
    gauss[threadId] *= exp(-(my_c-len)*(my_c-len)/(2*sig*sig))/(sig*sqrt(2*PI));
    //进行滤波操作

    __syncthreads();
    //同步，以防后续操作出错
    //后续将进行层间差分操作，为了保证一致性进行划分操作

    if(threadId >= layer)
        return;

    len = length;
    while (my_layer){
        my_id += (3 + S) * len * len;
        len /= 2;
        my_layer --;
    }
    //获取当前起始位置

    for(int s = 0; s < (S + 2); s ++)
        for(int i = 0; i < len; i ++)
            for(int j = 0; j < len; j ++){
                gauss[my_id + s*len*len + i*len + j] -= gauss[my_id + (s+1)*len*len + i*len + j];
            }
    __syncthreads();
}
//验证程序正确性
//void output(float* gauss_h_inner,int len){
//    const int S = 2;
//    int layer = 0;
//    int l = len;
//    while (l){
//        l /= 2;
//        layer ++;
//    }
//    int start_id =0;
//    for (int i=0; i<layer; i++){
//        for (int k=0; k < len; k++){
//            for (l=0; l< len; l++){
//                cout<<gauss_h_inner[start_id + k*len + l]<<" ";
//            }
//            cout<<endl;
//        }
//        for (int k=0; k<len; k++){
//            cout<<"==";
//        }
//        cout<<endl;
//        start_id += i*(S+3)*len*len;
//        len /= 2;
//    }
//}
int main() {
    n = 8;
    int deviceId;
    cudaGetDevice(&deviceId);
    cout << "deviceId: "<<deviceId<<endl;
    size_t threads_per_block = BLOCK_SIZE;
    size_t number_of_blocks = (n + threads_per_block -1)/threads_per_block;
    while(n < MAX){
        cudaMallocHost(&gauss_h, gauss_size(n));
        cudaMalloc(&gauss, gauss_size(n));
        std::chrono::duration<double, std::milli> elapsed{};
        auto start= std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        elapsed +=end-start;
        int times =0;
        while (elapsed.count() < 5000){
            gauss_de_pyramid_initialize<<<number_of_blocks*BLOCK_STRIDE,threads_per_block>>>(gauss_size(n),gauss);
            start= std::chrono::high_resolution_clock::now();
            gauss_de_pyramid_generate<<<number_of_blocks*BLOCK_STRIDE,threads_per_block>>>(gauss_size(n),gauss);
            cudaDeviceSynchronize();
            end = std::chrono::high_resolution_clock::now();
            elapsed += end-start;
            cudaMemcpy(gauss_h,gauss, gauss_size(n),cudaMemcpyDeviceToHost);
            cudaFree(&gauss);
            cudaFreeHost(&gauss_h);
            times += 1;
        }
        cout<<n<<","<<float (elapsed.count())/float (times)<<endl;
        n *= 2;
    }
    return 0;
}
