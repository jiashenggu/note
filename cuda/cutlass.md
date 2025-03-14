


## Compile example

```bash
nvcc -arch=sm_80 -lineinfo turing_tensorop_gemm.cu --std=c++17 \
                                                    -I /home/wenming.gjs/cutlass/include/ \
                                                    -I /home/wenming.gjs/cutlass/tools/util/include/ \
                                                    -I /home/wenming.gjs/cutlass/examples/common/ 
```      
