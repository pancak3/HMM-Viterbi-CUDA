#include <stdio.h> 

#define MAX_OBS 2048 
#define MAX_STATES 64 
#define MAX_OUTS 32 

__global__ void viterbi_cuda(int *obs, double *trans_p, double *emit_p, double *path_p, int *back, int nstates, int nobs, int nouts)
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    int i, j, ipmax;
    
    __shared__ double emit_p_s[MAX_OUTS * MAX_STATES];
    __shared__ double trans_p_s[MAX_STATES * MAX_STATES];
    __shared__ double path_p_s[MAX_STATES];
    __shared__ double path_p_s_n[MAX_STATES];

    for(i = 0; i < nouts; i++) {
        emit_p_s[tx + i*nstates] = emit_p[tx + i*nstates + bx * nouts * nstates];
    }

    for(i = 0; i < nstates; i++) {
        trans_p_s[tx + nstates * i] = trans_p[tx + nstates * i + bx * nstates * nstates];
    }
    
    path_p_s_n[tx] = path_p[tx + bx*nstates];
    
    for(j = 1; j < nobs; j++) {
        path_p_s[tx] = path_p_s_n[tx];
        __syncthreads();
 
        double pmax = logf(0);
        double pt = 0; 
        ipmax = 0; // index of p max

        for(i = 0; i < nstates; i++) {
            pt = emit_p_s[obs[nobs*bx+j]*nstates+tx] + trans_p_s[i*nstates+tx] + path_p_s[i];
            if(pt > pmax) {
                ipmax = i;
                pmax = pt;
            }
        }
    
        path_p_s_n[tx] = pmax;
        back[j*nstates+tx+bx*nstates*nobs] = ipmax;
        __syncthreads();
    }
    
    path_p[tx + bx*nstates] = path_p_s_n[tx];
    
}

__global__ void viterbi_cudabacktrace(int nobs, int nstates, double *path_p, int *back, int *route)
{
    const int tx = threadIdx.x;
    int i;

    double max_p = path_p[tx*nstates];
    int imax_p = 0;
    
    for(i=1; i<nstates; i++) {
        if(path_p[tx*nstates+i] > max_p) {
            max_p = path_p[tx*nstates+i];
            imax_p = i;
        }
    }

    route[tx*nobs + nobs-1] = imax_p; 
    
    for(i=nobs-2; i > -1; i--) {
        route[tx*nobs+i] = back[tx*nstates*nobs+(i+1)*nstates+route[tx*nobs + i+1]];
    }
}

