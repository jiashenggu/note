


## Compile example

```bash
nvcc -arch=sm_80 -lineinfo turing_tensorop_gemm.cu -I /home/wenming.gjs/cutlass/include/ \
                                                    -I /home/wenming.gjs/cutlass/tools/util/include/ \
                                                    -I /home/wenming.gjs/cutlass/examples/common/ --std=c++17
```      
