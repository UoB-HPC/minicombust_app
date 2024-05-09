#include "flow/gpu/gpu_kernels.cuh"
#include <curand.h>
#include <curand_kernel.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>

#include <cuco/static_map.cuh>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define NOT_BOUNDARY 0
#define WALL 1 
#define INLET 2
#define OUTLET 3 

#define IDx(w,x,y,z) (w*1000000) + (x*10000) + (y*100) + z

using namespace minicombust::utils;

__device__ vec<double> vec_add(vec<double> lhs, const vec<double> rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
	return lhs;
}

__device__ vec<double> vec_mult(vec<double> lhs, const double rhs)
{
    lhs.x *= rhs;
    lhs.y *= rhs;
    lhs.z *= rhs;
    return lhs;
}

__device__ vec<double> vec_div(vec<double> lhs, const double rhs)
{
    lhs.x /= rhs;
    lhs.y /= rhs;
    lhs.z /= rhs;
    return lhs;
}

__device__ vec<double> vec_minus(vec<double> lhs, const vec<double> rhs)
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    return lhs;
}

__device__ double dot_product(const vec<double> lhs, const vec<double> rhs)
{
	return (lhs.x*rhs.x)+(lhs.y*rhs.y)+(lhs.z*rhs.z);
}

__device__ double dot_product(const double lhs, const vec<double> rhs)
{
	return (lhs*rhs.x)+(lhs*rhs.y)+(lhs*rhs.z);
}




__device__ double magnitude(vec<double> a)
{
	return sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
}

__device__ vec<double> normalise(vec<double> a)
{
	double tmp = magnitude(a);
	a.x = a.x / tmp;
	a.y = a.y / tmp;
	a.z = a.z / tmp; 
	return a;
}

__device__ void solve(double *A, double *b, double *out)
{
    double det = A[0] * (A[4] * A[8] - A[7] * A[5]) -
                 A[1] * (A[3] * A[8] - A[5] * A[6]) +
                 A[2] * (A[3] * A[7] - A[4] * A[6]);

    double invdet = 1 / det;

    out[0] = b[0] * invdet * (A[4] * A[8] - A[7] * A[5]) +
             b[1] * invdet * (A[2] * A[7] - A[1] * A[8]) +
             b[2] * invdet * (A[1] * A[5] - A[2] * A[4]);
    out[1] = b[0] * invdet * (A[5] * A[6] - A[3] * A[8]) +
             b[1] * invdet * (A[0] * A[8] - A[2] * A[6]) +
             b[2] * invdet * (A[3] * A[2] - A[0] * A[5]);
    out[2] = b[0] * invdet * (A[3] * A[7] - A[6] * A[4]) +
             b[1] * invdet * (A[6] * A[1] - A[0] * A[7]) +
             b[2] * invdet * (A[0] * A[4] - A[3] * A[1]);
}

template <typename Map>
__global__ void insert(Map map_ref,
					   uint64_t *keys,
					   uint64_t *values,
					   size_t num_keys)
{
  auto tid = threadIdx.x + blockIdx.x * blockDim.x;

  while (tid < num_keys) 
  {
	// Map::insert returns `true` if it is the first time the given key was
	// inserted and `false` if the key already existed
	map_ref.insert(cuco::pair{keys[tid], values[tid]});

    tid += gridDim.x * blockDim.x;
  }
}

__global__ void kernel_create_map(uint64_t *map, uint64_t *keys, uint64_t *values, uint64_t size)
{
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= size) return;

	// if (keys[tid] >= 56652225)
	// {
	// 	printf("%lu key out of range\n", keys[tid]);
	// }
	map[keys[tid]] = values[tid];


}

__global__ void kernel_process_particle_fields(uint64_t *sent_cell_indexes, particle_aos<double> *sent_particle_fields, particle_aos<double> *particle_fields, uint64_t num_fields, uint64_t local_mesh_disp)
{
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= num_fields) return;
	
	particle_fields[sent_cell_indexes[tid] - local_mesh_disp].momentum.x += sent_particle_fields[tid].momentum.x;
	particle_fields[sent_cell_indexes[tid] - local_mesh_disp].momentum.y += sent_particle_fields[tid].momentum.y;
	particle_fields[sent_cell_indexes[tid] - local_mesh_disp].momentum.z += sent_particle_fields[tid].momentum.z;
	particle_fields[sent_cell_indexes[tid] - local_mesh_disp].energy     += sent_particle_fields[tid].energy;
	particle_fields[sent_cell_indexes[tid] - local_mesh_disp].fuel       += sent_particle_fields[tid].fuel;

}

__global__ void kernel_pack_phi_halo_buffer(phi_vector<double> send_buffer, phi_vector<double> phi, uint64_t *indexes, uint64_t buf_size)
{
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= buf_size) return;

	send_buffer.U[tid]    = phi.U[indexes[tid]];
	send_buffer.V[tid]    = phi.V[indexes[tid]];
	send_buffer.W[tid]    = phi.W[indexes[tid]];
	send_buffer.P[tid]    = phi.P[indexes[tid]];
	send_buffer.TE[tid]   = phi.TE[indexes[tid]];
	send_buffer.ED[tid]   = phi.ED[indexes[tid]];
	send_buffer.TP[tid]   = phi.TP[indexes[tid]];
	send_buffer.TEM[tid]  = phi.TEM[indexes[tid]];
	send_buffer.FUL[tid]  = phi.FUL[indexes[tid]];
	send_buffer.PRO[tid]  = phi.PRO[indexes[tid]];
	send_buffer.VARF[tid] = phi.VARF[indexes[tid]];
	send_buffer.VARP[tid] = phi.VARP[indexes[tid]];
}

__global__ void kernel_pack_phi_grad_halo_buffer(phi_vector<vec<double>> send_buffer, phi_vector<vec<double>> phi_grad, uint64_t *indexes, uint64_t buf_size)
{
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= buf_size) return;

	send_buffer.U[tid]    = phi_grad.U[indexes[tid]];
	send_buffer.V[tid]    = phi_grad.V[indexes[tid]];
	send_buffer.W[tid]    = phi_grad.W[indexes[tid]];
	send_buffer.P[tid]    = phi_grad.P[indexes[tid]];
	send_buffer.TE[tid]   = phi_grad.TE[indexes[tid]];
	send_buffer.ED[tid]   = phi_grad.ED[indexes[tid]];
	send_buffer.TEM[tid]  = phi_grad.TEM[indexes[tid]];
	send_buffer.FUL[tid]  = phi_grad.FUL[indexes[tid]];
	send_buffer.PRO[tid]  = phi_grad.PRO[indexes[tid]];
	send_buffer.VARF[tid] = phi_grad.VARF[indexes[tid]];
	send_buffer.VARP[tid] = phi_grad.VARP[indexes[tid]];
}

__global__ void kernel_pack_PP_halo_buffer(phi_vector<double> send_buffer, phi_vector<double> phi, uint64_t *indexes, uint64_t buf_size)
{
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= buf_size) return;

	send_buffer.PP[tid]   = phi.PP[indexes[tid]];
}

__global__ void kernel_pack_Aphi_halo_buffer(phi_vector<double> send_buffer, phi_vector<double> phi, uint64_t *indexes, uint64_t buf_size)
{
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= buf_size) return;

	send_buffer.U[tid]   = phi.U[indexes[tid]];
}

__global__ void kernel_pack_PP_grad_halo_buffer(phi_vector<vec<double>> send_buffer, phi_vector<vec<double>> phi_grad, uint64_t *indexes, uint64_t buf_size)
{
	const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid >= buf_size) return;

	send_buffer.PP[tid]   = phi_grad.PP[indexes[tid]];
}


// __global__ __launch_bounds__(256, MIN_BLOCKS_PER_MP)

__global__ void kernel_get_phi_gradient(double *phi_component, bool pressure_solve, uint64_t local_mesh_size, uint64_t local_cells_disp, uint64_t faces_per_cell, gpu_Face<uint64_t> *faces, uint64_t *cell_faces, vec<double> *cell_centers, uint64_t mesh_size, uint64_t *boundary_map, uint64_t *boundary_map_values, int64_t map_size, vec<double> *face_centers, uint64_t nhalos, vec<double> *grad_component)
{
    const uint64_t block_cell = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(block_cell >= local_mesh_size) return;
	
	double data_A[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double data_b[] = {0.0, 0.0, 0.0};
	
    const uint64_t cell = block_cell + local_cells_disp;
    //keep this loop since it is very small
    for(uint64_t f = 0; f < faces_per_cell; f++)
    {
		//reset data arrays
		//gpuErrchk(cudaMemset(data_A, 0.0, 9 * sizeof(T)));
		//gpuErrchk(cudaMemset(data_bU, 0.0, 3 * sizeof(T)));

        const uint64_t face = cell_faces[block_cell * faces_per_cell + f];

        const uint64_t block_cell0 = faces[face].cell0 - local_cells_disp;
        const uint64_t block_cell1 = faces[face].cell1 - local_cells_disp;
        double dphi;
        vec<double> dX;
        if(faces[face].cell1 < mesh_size) //inner cell
        {
            uint64_t phi_index0;
            if(block_cell0 >= local_mesh_size)
            {
				phi_index0 = boundary_map[faces[face].cell0];
            }
            else
            {
                phi_index0 = block_cell0;
            }
            uint64_t phi_index1;
            if (block_cell1 >= local_mesh_size)
            {
				phi_index1 = boundary_map[faces[face].cell1];
            }
            else
            {
                phi_index1 = block_cell1;
            }
            const double mask = (faces[face].cell0 == cell) ? 1. : -1.;
			dphi =   mask * ( phi_component[phi_index1]   - phi_component[phi_index0] );

            dX.x = mask*(cell_centers[faces[face].cell1].x - cell_centers[faces[face].cell0].x);
            dX.y = mask*(cell_centers[faces[face].cell1].y - cell_centers[faces[face].cell0].y);
            dX.z = mask*(cell_centers[faces[face].cell1].z - cell_centers[faces[face].cell0].z);
        }
        else  //boundary face
        {
            const uint64_t boundary_cell = faces[face].cell1 - mesh_size;

			if (pressure_solve)
			{
            	dphi = 0.0;
			}
			else
			{
				dphi = phi_component[local_mesh_size + nhalos + boundary_cell] - phi_component[block_cell0];
			}

			//We only ever single compute a pressure grad.

            dX.x = face_centers[face].x - cell_centers[faces[face].cell0].x;
            dX.y = face_centers[face].y - cell_centers[faces[face].cell0].y;
            dX.z = face_centers[face].z - cell_centers[faces[face].cell0].z;
        }
        data_A[0] += (dX.x * dX.x);
        data_A[1] += (dX.x * dX.y);
        data_A[2] += (dX.x * dX.z);

        data_A[3] += (dX.y * dX.x);
        data_A[4] += (dX.y * dX.y);
        data_A[5] += (dX.y * dX.z);

        data_A[6] += (dX.z * dX.x);
        data_A[7] += (dX.z * dX.y);
        data_A[8] += (dX.z * dX.z);

        data_b[0] += (dX.x * dphi);
        data_b[1] += (dX.y * dphi);
        data_b[2] += (dX.z * dphi);
	}
	solve(data_A, data_b, &grad_component[block_cell].x);
}

__global__ void test_solve()
{
	double data_A[9] = {1,0,0,0,1,0,0,0,1};
	double data_b[3] = {1,1,1};
	double out[3];
	solve(data_A, data_b, out);
	printf("out is (%f, %f, %f)\n", out[0], out[1], out[2]);
}

void C_test_solve()
{
	test_solve<<<1,1>>>();
}

__global__ void kernel_get_phi_gradients(phi_vector<double> phi, phi_vector<vec<double>> phi_grad, uint64_t local_mesh_size, uint64_t local_cells_disp, uint64_t faces_per_cell, gpu_Face<uint64_t> *faces, uint64_t *cell_faces, vec<double> *cell_centers, uint64_t mesh_size, uint64_t *boundary_map, uint64_t *boundary_map_values, int64_t map_size, vec<double> *face_centers, uint64_t nhalos)
{
	const uint64_t block_cell = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(block_cell >= local_mesh_size) return;
	double data_A[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; //= (double *)(((char *)full_data_A)+(block_cell*pitch_A));
	double data_bU[] = {0.0, 0.0, 0.0}; //= (double *)(((char *)full_data_bU)+(block_cell*pitch_bU));
    double data_bV[] = {0.0, 0.0, 0.0}; //=/ (double *)(((char *)full_data_bV)+(block_cell*pitch_bV));
    double data_bW[] = {0.0, 0.0, 0.0}; //= (double *)(((char *)full_data_bW)+(block_cell*pitch_bW));
    double data_bP[] = {0.0, 0.0, 0.0}; //= (double *)(((char *)full_data_bP)+(block_cell*pitch_bP));
    double data_bTE[] = {0.0, 0.0, 0.0}; //= (double *)(((char *)full_data_bTE)+(block_cell*pitch_bTE));
    double data_bED[] = {0.0, 0.0, 0.0}; //= (double *)(((char *)full_data_bED)+(block_cell*pitch_bED));
    double data_bT[] = {0.0, 0.0, 0.0}; //= (double *)(((char *)full_data_bT)+(block_cell*pitch_bT));
    double data_bFU[] = {0.0, 0.0, 0.0}; //= (double *)(((char *)full_data_bFU)+(block_cell*pitch_bFU));
    double data_bPR[] = {0.0, 0.0, 0.0}; //= (double *)(((char *)full_data_bPR)+(block_cell*pitch_bPR));
    double data_bVFU[] = {0.0, 0.0, 0.0}; //= (double *)(((char *)full_data_bVFU)+(block_cell*pitch_bVFU));
    double data_bVPR[] = {0.0, 0.0, 0.0}; //= (double *)(((char *)full_data_bVPR)+(block_cell*pitch_bVPR));

	const uint64_t cell = block_cell + local_cells_disp;
	//keep this loop since it is very small
	for(uint64_t f = 0; f < faces_per_cell; f++)
	{
		const uint64_t face = cell_faces[block_cell * faces_per_cell + f];

		const uint64_t block_cell0 = faces[face].cell0 - local_cells_disp;
		const uint64_t block_cell1 = faces[face].cell1 - local_cells_disp;
		double dU, dV, dW, dP, dTE, dED, dT, dFU, dPR, dVFU, dVPR;
		vec<double> dX;
		if(faces[face].cell1 < mesh_size) //inner cell
		{
			uint64_t phi_index0;
			if(block_cell0 >= local_mesh_size)
			{
				phi_index0 = boundary_map[faces[face].cell0];
			}
			else
			{
				phi_index0 = block_cell0;
			}
			uint64_t phi_index1;
            if(block_cell1 >= local_mesh_size)
            {
				phi_index1 = boundary_map[faces[face].cell1];
            }
            else
            {
                phi_index1 = block_cell1;
            }
			const double mask = (faces[face].cell0 == cell) ? 1. : -1.;
			dU =   mask * ( phi.U[phi_index1]   - phi.U[phi_index0] );
			dV =   mask * ( phi.V[phi_index1]   - phi.V[phi_index0] );
            dW =   mask * ( phi.W[phi_index1]   - phi.W[phi_index0] );
            dP =   mask * ( phi.P[phi_index1]   - phi.P[phi_index0] );
			dTE =  mask * ( phi.TE[phi_index1]  - phi.TE[phi_index0] );
            dED =  mask * ( phi.ED[phi_index1]  - phi.ED[phi_index0] );
            dT =   mask * ( phi.TEM[phi_index1] - phi.TEM[phi_index0] );
            dFU =  mask * ( phi.FUL[phi_index1] - phi.FUL[phi_index0] );
            dPR =  mask * ( phi.PRO[phi_index1] - phi.PRO[phi_index0] );
            dVFU = mask * ( phi.VARF[phi_index1] - phi.VARF[phi_index0] );
            dVPR = mask * ( phi.VARP[phi_index1] - phi.VARP[phi_index0] );

			dX.x = mask*(cell_centers[faces[face].cell1].x - cell_centers[faces[face].cell0].x);
			dX.y = mask*(cell_centers[faces[face].cell1].y - cell_centers[faces[face].cell0].y);
			dX.z = mask*(cell_centers[faces[face].cell1].z - cell_centers[faces[face].cell0].z);
		}
		else  //boundary face
		{
			const uint64_t boundary_cell = faces[face].cell1 - mesh_size;

			dU = phi.U[local_mesh_size + nhalos + boundary_cell] - phi.U[block_cell0];
            dV = phi.V[local_mesh_size + nhalos + boundary_cell] - phi.V[block_cell0];
            dW = phi.W[local_mesh_size + nhalos + boundary_cell] - phi.W[block_cell0];
            dP = 0.0;//dolfyn also enforces dp = 0.0 over boundary
            dTE = phi.TE[local_mesh_size + nhalos + boundary_cell] - phi.TE[block_cell0];
            dED = phi.ED[local_mesh_size + nhalos + boundary_cell] - phi.ED[block_cell0];
            dT = phi.TEM[local_mesh_size + nhalos + boundary_cell] - phi.TEM[block_cell0];
            dFU = phi.FUL[local_mesh_size + nhalos + boundary_cell] - phi.FUL[block_cell0];
            dPR = phi.PRO[local_mesh_size + nhalos + boundary_cell] - phi.PRO[block_cell0];
            dVFU = phi.VARF[local_mesh_size + nhalos + boundary_cell] - phi.VARF[block_cell0];
            dVPR = phi.VARP[local_mesh_size + nhalos + boundary_cell] - phi.VARP[block_cell0];

			dX.x = face_centers[face].x - cell_centers[faces[face].cell0].x;
			dX.y = face_centers[face].y - cell_centers[faces[face].cell0].y;
			dX.z = face_centers[face].z - cell_centers[faces[face].cell0].z;
		}
		data_A[0] += (dX.x * dX.x);
        data_A[1] += (dX.x * dX.y);
        data_A[2] += (dX.x * dX.z);

        data_A[3] += (dX.y * dX.x);
        data_A[4] += (dX.y * dX.y);
        data_A[5] += (dX.y * dX.z);

        data_A[6] += (dX.z * dX.x);
        data_A[7] += (dX.z * dX.y);
        data_A[8] += (dX.z * dX.z);
        
		data_bU[0] += (dX.x * dU);
        data_bU[1] += (dX.y * dU);
        data_bU[2] += (dX.z * dU);

		data_bV[0] += (dX.x * dV);
        data_bV[1] += (dX.y * dV);
        data_bV[2] += (dX.z * dV);

        data_bW[0] += (dX.x * dW);
        data_bW[1] += (dX.y * dW);
        data_bW[2] += (dX.z * dW);

        data_bP[0] += (dX.x * dP);
        data_bP[1] += (dX.y * dP);
        data_bP[2] += (dX.z * dP);

        data_bTE[0] += (dX.x * dTE);
        data_bTE[1] += (dX.y * dTE);
        data_bTE[2] += (dX.z * dTE);

		data_bED[0] += (dX.x * dED);
        data_bED[1] += (dX.y * dED);
        data_bED[2] += (dX.z * dED);

        data_bT[0] += (dX.x * dT);
        data_bT[1] += (dX.y * dT);
        data_bT[2] += (dX.z * dT);

        data_bFU[0] += (dX.x * dFU);
        data_bFU[1] += (dX.y * dFU);
        data_bFU[2] += (dX.z * dFU);

        data_bPR[0] += (dX.x * dPR);
        data_bPR[1] += (dX.y * dPR);
        data_bPR[2] += (dX.z * dPR);

        data_bVFU[0] += (dX.x * dVFU);
        data_bVFU[1] += (dX.y * dVFU);
        data_bVFU[2] += (dX.z * dVFU);

        data_bVPR[0] += (dX.x * dVPR);
        data_bVPR[1] += (dX.y * dVPR);
        data_bVPR[2] += (dX.z * dVPR);
	}
	solve(data_A, data_bU, &phi_grad.U[block_cell].x);
	solve(data_A, data_bV, &phi_grad.V[block_cell].x);
	solve(data_A, data_bW, &phi_grad.W[block_cell].x);
	solve(data_A, data_bP, &phi_grad.P[block_cell].x);
	solve(data_A, data_bTE, &phi_grad.TE[block_cell].x);
	solve(data_A, data_bED, &phi_grad.ED[block_cell].x);
	solve(data_A, data_bT, &phi_grad.TEM[block_cell].x);
	solve(data_A, data_bFU, &phi_grad.FUL[block_cell].x);
	solve(data_A, data_bPR, &phi_grad.PRO[block_cell].x);
	solve(data_A, data_bVFU, &phi_grad.VARF[block_cell].x);
	solve(data_A, data_bVPR, &phi_grad.VARP[block_cell].x);
}

__global__ void kernel_precomp_AU(uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t mesh_size, uint64_t *boundary_types, double effective_viscosity, double * face_rlencos, double *face_mass_fluxes, phi_vector<double> A_phi, uint64_t local_mesh_size, double delta, double *cell_densities, double* cell_volumes)
{
	//Big gpu loop
	const uint64_t face = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(face >= faces_size) return;
	const uint64_t block_cell0 = faces[face].cell0 - local_cells_disp;
	if(faces[face].cell1 >= mesh_size){
		//only boundaries
		const uint64_t boundary_cell = faces[face].cell1 - mesh_size;
		const uint64_t boundary_type = boundary_types[boundary_cell];
		if(boundary_type == INLET)
		{
			const double Visac = effective_viscosity;
			const double VisFace  = Visac * face_rlencos[face];
			const double f = -VisFace + min( face_mass_fluxes[face], 0.0 );
			atomicAdd(&A_phi.U[block_cell0], -1*f);
		}
		else if(boundary_type == OUTLET)
		{
			const double Visac = effective_viscosity;
			const double VisFace  = Visac * face_rlencos[face];
			const double f = -VisFace + min( face_mass_fluxes[face], 0.0 );
			atomicAdd(&A_phi.U[block_cell0], -1*f);
		}
	}
	if(face >= local_mesh_size) return;
	const double rdelta = 1.0/delta;
	double f = cell_densities[face] * cell_volumes[face] * rdelta;
	atomicAdd(&A_phi.U[face], f);
}

__global__ void kernel_calculate_mass_flux(uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t mesh_size, uint64_t local_mesh_size, int64_t map_size, uint64_t *boundary_map, uint64_t *boundary_map_values, phi_vector<vec<double>> phi_grad, vec<double> *cell_centers, vec<double> *face_centers, phi_vector<double> phi, double *cell_densities, phi_vector<double> A_phi, double *cell_volumes, double *face_mass_fluxes, double *face_lambdas, vec<double> *face_normals, double *face_areas, gpu_Face<double> *face_fields, phi_vector<double> S_phi, uint64_t nhalos, uint64_t *boundary_types, vec<double> dummy_gas_vel)
{
	//Big gpu loop
    const uint64_t face = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(face >= faces_size) return;
	const uint64_t block_cell0 = faces[face].cell0 - local_cells_disp;
    const uint64_t block_cell1 = faces[face].cell1 - local_cells_disp;

	if(faces[face].cell1 < mesh_size) //internal
	{
		uint64_t phi_index0;
        if(block_cell0 >= local_mesh_size)
        {
			phi_index0 = boundary_map[faces[face].cell0];
        }
        else
        {
            phi_index0 = block_cell0;
        }
		uint64_t phi_index1;
        if(block_cell1 >= local_mesh_size)
        {
			phi_index1 = boundary_map[faces[face].cell1];
        }
        else
        {
            phi_index1 = block_cell1;
        }
		const double lambda0 = face_lambdas[face];
		const double lambda1 = 1.0 - lambda0;
	
		const vec<double> dUdXac = vec_add(vec_mult(phi_grad.U[phi_index0], lambda0), 
											vec_mult(phi_grad.U[phi_index1], lambda1));
		const vec<double> dVdXac = vec_add(vec_mult(phi_grad.V[phi_index0], lambda0), 
											vec_mult(phi_grad.V[phi_index1], lambda1));	
		const vec<double> dWdXac = vec_add(vec_mult(phi_grad.W[phi_index0], lambda0), 
											vec_mult(phi_grad.W[phi_index1], lambda1));
		
		vec<double> Xac = vec_add(vec_mult(cell_centers[faces[face].cell1], lambda1),
								   vec_mult(cell_centers[faces[face].cell0], lambda0));
		
		const vec<double> delta  = vec_minus(face_centers[face], Xac);

		const double UFace = phi.U[phi_index1]*lambda1 + phi.U[phi_index0]*lambda0 + 
						  	 dot_product(dUdXac,delta);
		const double VFace = phi.V[phi_index1]*lambda1 + phi.V[phi_index0]*lambda0 + 
							 dot_product(dVdXac,delta);
		const double WFace = phi.W[phi_index1]*lambda1 + phi.W[phi_index0]*lambda0 + 
							 dot_product(dWdXac,delta);

		const double densityf = cell_densities[phi_index0]*lambda0 + 
								cell_densities[phi_index1]*lambda1;

		face_mass_fluxes[face] = densityf * (UFace * face_normals[face].x +
                                             VFace * face_normals[face].y +
                                             WFace * face_normals[face].z );

		const vec<double> Xpac = vec_minus(face_centers[face], 
								 vec_mult(normalise(face_normals[face]),
										  dot_product(vec_minus(face_centers[face], 
										  cell_centers[faces[face].cell0]), 
										  normalise(face_normals[face]))));
		const vec<double> Xnac = vec_minus(face_centers[face], 
								 vec_mult(normalise(face_normals[face]),
										  dot_product(vec_minus(face_centers[face],
										  cell_centers[faces[face].cell1]), 
										  normalise(face_normals[face]))));

		const vec<double> delp = vec_minus(Xpac, cell_centers[faces[face].cell0]);
		const vec<double> deln = vec_minus(Xnac, cell_centers[faces[face].cell1]);

		const double cell0_P = phi.P[phi_index0] + 
								dot_product( phi_grad.P[phi_index0] , delp );
		const double cell1_P = phi.P[phi_index1] + 
								dot_product( phi_grad.P[phi_index1] , deln );
		
		const vec<double> Xpn  = vec_minus(Xnac, Xpac);
		const vec<double> Xpn2 = vec_minus(cell_centers[faces[face].cell1], 
										   cell_centers[faces[face].cell0]);

		const double ApV0 = (A_phi.U[phi_index0] != 0.0) ? 1.0 / A_phi.U[phi_index0] : 0.0;
		const double ApV1 = (A_phi.U[phi_index1] != 0.0) ? 1.0 / A_phi.U[phi_index1] : 0.0;
		
		double ApV = cell_densities[phi_index0] * ApV0 * lambda0 + cell_densities[phi_index1] * ApV1 * lambda1;

		const double volume_avg = cell_volumes[phi_index0] * lambda0 + cell_volumes[phi_index1] * lambda1;
		
		ApV  = ApV * face_areas[face] * volume_avg/dot_product(Xpn2, normalise(face_normals[face]));

		const double dpx  = ( phi_grad.P[phi_index1].x * lambda1 + phi_grad.P[phi_index0].x * lambda0) * Xpn.x;
		const double dpy  = ( phi_grad.P[phi_index1].y * lambda1 + phi_grad.P[phi_index0].y * lambda0) * Xpn.y;
		const double dpz  = ( phi_grad.P[phi_index1].z * lambda1 + phi_grad.P[phi_index0].z * lambda0) * Xpn.z;
	
		face_fields[face].cell0 = -ApV;
		face_fields[face].cell1 = -ApV;
		
		face_mass_fluxes[face] -= ApV * ((cell1_P - cell0_P) - dpx - dpy - dpz);
	}	
	else //Boundary
	{
		// Boundary faces
		const uint64_t boundary_cell = faces[face].cell1 - mesh_size;
		const uint64_t boundary_type = boundary_types[boundary_cell];
		if ( boundary_type == INLET )
		{
			// Constant inlet values for velocities and densities. Add custom regions laters
            const vec<double> vel_inward = dummy_gas_vel;
            const double Din = 1.2;

            face_mass_fluxes[face] = Din * dot_product( vel_inward, face_normals[face] );
            atomicAdd(&S_phi.U[block_cell0], -1*face_mass_fluxes[face]);
		}
		else if(boundary_type == OUTLET)
		{
			const vec<double> vel_outward = { phi.U[block_cell0],
											  phi.V[block_cell0],
                                              phi.W[block_cell0] };
            const double Din = 1.2;

            face_mass_fluxes[face] = Din * dot_product(vel_outward, face_normals[face]);
            // !
            // ! For an outlet face_mass_fluxes must be 0.0 or positive
            // !
            if( face_mass_fluxes[face] < 0.0 )
            {
                printf("MAIN COMP PRES NEGATIVE OUTFLOW %3.18f\n", face_mass_fluxes[face]);
                face_mass_fluxes[face] = 1e-15;

                phi.TE[local_mesh_size + nhalos + boundary_cell] =
                    phi.TE[block_cell0];
                phi.ED[local_mesh_size + nhalos + boundary_cell] =
                    phi.ED[block_cell0];
                phi.TEM[local_mesh_size + nhalos + boundary_cell] =
                    phi.TEM[block_cell0];
            }
			else if(boundary_type == WALL)
			{
				face_mass_fluxes[face] = 0.0;
			}
		}
	}
}

__global__ void kernel_compute_flow_correction(uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t mesh_size, uint64_t *boundary_types, double *FlowOut, double *FlowIn, double *areaout, int *count_out, double *face_mass_fluxes, double *face_areas)
{
	const uint64_t face = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(face >= faces_size) return;
	if ( faces[face].cell1 < mesh_size )  return;
	//Boundary
	const uint64_t boundary_cell = faces[face].cell1 - mesh_size;
	const uint64_t boundary_type = boundary_types[boundary_cell];
	if(boundary_type == INLET)
	{
		atomicAdd(FlowIn, face_mass_fluxes[face]);
	}
	else if(boundary_type == OUTLET)
	{
		atomicAdd(FlowOut, face_mass_fluxes[face]);
		atomicAdd(count_out, 1);
		atomicAdd(areaout, face_areas[face]);
	}
}

__global__ void kernel_correct_flow(int *count_out, double *FlowOut, double *FlowIn, double *areaout, uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t mesh_size, uint64_t *boundary_types, double *face_mass_fluxes, double *face_areas, double *cell_densities, phi_vector<double> phi, uint64_t local_mesh_size, uint64_t nhalos, vec<double> *face_normals, uint64_t local_cells_disp, phi_vector<double> S_phi, double *FlowFact)
{
	const uint64_t face = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(face >= faces_size) return;
	if(*FlowOut == 0.0)
	{
		*FlowFact = 0.0;
	}
	else
	{
		*FlowFact = -1 * *FlowIn / *FlowOut;
	}
	if(*FlowOut < 0.0000000001)
	{
		double ratearea = - *FlowIn / *areaout;
		*FlowOut = 0.0;
		if(faces[face].cell1 >= mesh_size)
		{
			//Boundary only
			const uint64_t boundary_cell = faces[face].cell1 - mesh_size;
			const uint64_t boundary_type = boundary_types[boundary_cell];
			if(boundary_type == OUTLET)
			{
				//NOTE: assumes constent and uniform density
				//NOTE: assumes one outflow region
				face_mass_fluxes[face] = ratearea*face_areas[face];
				double FaceFlux = face_mass_fluxes[face]/cell_densities[0]/face_areas[face];

				phi.U[local_mesh_size + nhalos + boundary_cell] = FaceFlux*normalise(face_normals[face]).x;
				phi.V[local_mesh_size + nhalos + boundary_cell] = FaceFlux*normalise(face_normals[face]).y;
				phi.W[local_mesh_size + nhalos + boundary_cell] = FaceFlux*normalise(face_normals[face]).z;

				atomicAdd(FlowOut, face_mass_fluxes[face]);
			}
		}
	}
}

__global__ void kernel_correct_flow2(int *count_out, double *FlowOut, double *FlowIn, double *areaout, uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t mesh_size, uint64_t *boundary_types, double *face_mass_fluxes, double *face_areas, double *cell_densities, phi_vector<double> phi, uint64_t local_mesh_size, uint64_t nhalos, vec<double> *face_normals, uint64_t local_cells_disp, phi_vector<double> S_phi, double *FlowFact)
{
	const uint64_t face = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(face >= faces_size) return;
	double fact  = - *FlowIn/(*FlowOut + 0.0000001);
	if(faces[face].cell1 >= mesh_size)
	{
		//Boundary only
		const uint64_t boundary_cell = faces[face].cell1 - mesh_size;
		const uint64_t boundary_type = boundary_types[boundary_cell];
		if(boundary_type == OUTLET)
		{
			face_mass_fluxes[face] *= *FlowFact;
	
			phi.U[local_mesh_size + nhalos + boundary_cell] *= fact;
			phi.V[local_mesh_size + nhalos + boundary_cell] *= fact;
			phi.W[local_mesh_size + nhalos + boundary_cell] *= fact;

			const uint64_t block_cell0 = faces[face].cell0 - local_cells_disp;
			atomicAdd(&S_phi.U[block_cell0], -1 * face_mass_fluxes[face]);
		}
	}
}

__global__ void kernel_calculate_flux_UVW(uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t mesh_size, uint64_t local_mesh_size, int64_t map_size, uint64_t *boundary_map, uint64_t *boundary_map_values, phi_vector<vec<double>> phi_grad, vec<double> *cell_centers, vec<double> *face_centers, phi_vector<double> phi, phi_vector<double> A_phi, double *face_mass_fluxes, double *face_lambdas, vec<double> *face_normals, gpu_Face<double> *face_fields, phi_vector<double> S_phi, uint64_t nhalos, uint64_t *boundary_types, vec<double> dummy_gas_vel, double effective_viscosity, double *face_rlencos, double inlet_effective_viscosity, double *face_areas)
{
	double GammaBlend = 0.0;

	//Big gpu loop
    const uint64_t face = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(face >= faces_size) return;
    const uint64_t block_cell0 = faces[face].cell0 - local_cells_disp;
    const uint64_t block_cell1 = faces[face].cell1 - local_cells_disp;	

	if(faces[face].cell1 < mesh_size) //internal
	{
		uint64_t phi_index0;
        if(block_cell0 >= local_mesh_size)
        {
			phi_index0 = boundary_map[faces[face].cell0];
        }
        else
        {
            phi_index0 = block_cell0;
        }
        uint64_t phi_index1;
        if(block_cell1 >= local_mesh_size)
        {
			phi_index1 = boundary_map[faces[face].cell1];
		}
        else
        {
            phi_index1 = block_cell1;
        }
        const double lambda0 = face_lambdas[face];
        const double lambda1 = 1.0 - lambda0;

		const vec<double> dUdXac = vec_add(vec_mult(phi_grad.U[phi_index0],
										  lambda0), vec_mult(phi_grad.U[phi_index1],
										  lambda1));
		const vec<double> dVdXac = vec_add(vec_mult(phi_grad.V[phi_index0],
										  lambda0), vec_mult(phi_grad.V[phi_index1],
										  lambda1));
		const vec<double> dWdXac = vec_add(vec_mult(phi_grad.W[phi_index0],
										  lambda0), vec_mult(phi_grad.W[phi_index1],
										  lambda1));

		double Visac = effective_viscosity * lambda0 + effective_viscosity * lambda1;
		double VisFace = Visac * face_rlencos[face];
		
		vec<double> Xpn = vec_minus(cell_centers[faces[face].cell1], cell_centers[faces[face].cell0]);

		double UFace, VFace, WFace;
		if(face_mass_fluxes[face] >= 0.0)
		{
			UFace = phi.U[phi_index0];
			VFace = phi.V[phi_index0];
			WFace = phi.W[phi_index0];
		}
		else
		{
			UFace = phi.U[phi_index1];
			VFace = phi.V[phi_index1];
			WFace = phi.W[phi_index1];
		}
		
		// explicit higher order convective flux (see eg. eq. 8.16)
		const double fuce = face_mass_fluxes[face] * UFace;
        const double fvce = face_mass_fluxes[face] * VFace;
        const double fwce = face_mass_fluxes[face] * WFace;

		const double sx = face_normals[face].x;
        const double sy = face_normals[face].y;
        const double sz = face_normals[face].z;

		// explicit higher order diffusive flux based on simple uncorrected
        // interpolated cell centred gradients(see eg. eq. 8.19)
        const double fude = Visac * ((dUdXac.x+dUdXac.x)*sx + (dUdXac.y+dVdXac.x)*sy + (dUdXac.z+dWdXac.x)*sz);
        const double fvde = Visac * ((dUdXac.y+dVdXac.x)*sx + (dVdXac.y+dVdXac.y)*sy + (dVdXac.z+dWdXac.y)*sz);
        const double fwde = Visac * ((dUdXac.z+dWdXac.x)*sx + (dWdXac.y+dVdXac.z)*sy + (dWdXac.z+dWdXac.z)*sz);

		// ! implicit lower order (simple upwind)
        // ! convective and diffusive fluxes
        const double fmin = min( face_mass_fluxes[face], 0.0 );
        const double fmax = max( face_mass_fluxes[face], 0.0 );

		const double fuci = fmin * phi.U[phi_index0] + fmax * phi.U[phi_index1];
		const double fvci = fmin * phi.V[phi_index0] + fmax * phi.V[phi_index1];
		const double fwci = fmin * phi.W[phi_index0] + fmax * phi.W[phi_index1];
	
		const double fudi = VisFace * dot_product( dUdXac , Xpn );
		const double fvdi = VisFace * dot_product( dVdXac , Xpn );
		const double fwdi = VisFace * dot_product( dWdXac , Xpn );

		// !
        // ! convective coefficients with deferred correction with
        // ! gamma as the blending factor (0.0 <= gamma <= 1.0)
        // !
        // !      low            high    low  OLD
        // ! F = F    + gamma ( F     - F    )
        // !     ----   -------------------------
        // !      |                  |
        // !  implicit           explicit (dump into source term)
        // !
        // !            diffusion       convection
        // !                v               v
		face_fields[face].cell0 = -VisFace - max( face_mass_fluxes[face] , 0.0 );  // P (e);
		face_fields[face].cell1 = -VisFace + min( face_mass_fluxes[face] , 0.0 );  // N (w);
		
		const double blend_u = GammaBlend * ( fuce - fuci );
		const double blend_v = GammaBlend * ( fvce - fvci );
		const double blend_w = GammaBlend * ( fwce - fwci );
	
		// ! assemble the two source terms
        atomicAdd(&S_phi.U[phi_index0], fude - blend_u - fudi);
        atomicAdd(&S_phi.V[phi_index0], fvde - blend_v - fvdi);
        atomicAdd(&S_phi.W[phi_index0], fwde - blend_w - fwdi);

		atomicAdd(&S_phi.U[phi_index1], blend_u - fude + fudi);
		atomicAdd(&S_phi.V[phi_index1], blend_v - fvde + fvdi);
		atomicAdd(&S_phi.W[phi_index1], blend_w - fwde + fwdi);
	}
	else //boundary
	{
		const uint64_t boundary_cell = faces[face].cell1 - mesh_size;
		const uint64_t boundary_type = boundary_types[boundary_cell];

		if ( boundary_type == INLET )
		{
			const vec<double> dUdXac = phi_grad.U[block_cell0];
			const vec<double> dVdXac = phi_grad.V[block_cell0];
			const vec<double> dWdXac = phi_grad.W[block_cell0];

			const double UFace = dummy_gas_vel.x;
			const double VFace = dummy_gas_vel.y;
			const double WFace = dummy_gas_vel.z;

			const double Visac = inlet_effective_viscosity;
			
			const vec<double> Xpn = vec_minus(face_centers[face], cell_centers[faces[face].cell0]);
			const double VisFace  = Visac * face_rlencos[face];

			const double sx = face_normals[face].x;
			const double sy = face_normals[face].y;
			const double sz = face_normals[face].z;

			const double fude = Visac * ((dUdXac.x+dUdXac.x)*sx + (dUdXac.y+dVdXac.x)*sy + (dUdXac.z+dWdXac.x)*sz);
			const double fvde = Visac * ((dUdXac.y+dVdXac.x)*sx + (dVdXac.y+dVdXac.y)*sy + (dVdXac.z+dWdXac.y)*sz);
			const double fwde = Visac * ((dUdXac.z+dWdXac.x)*sx + (dWdXac.y+dVdXac.z)*sy + (dWdXac.z+dWdXac.z)*sz);

			const double fudi = VisFace * dot_product( dUdXac , Xpn );
			const double fvdi = VisFace * dot_product( dVdXac , Xpn );
			const double fwdi = VisFace * dot_product( dWdXac , Xpn );

			// ! by definition points a boundary normal outwards
            // ! therefore an inlet results in a mass flux < 0.0
			const double f = -VisFace + min( face_mass_fluxes[face], 0.0 );
			
			atomicAdd(&A_phi.U[block_cell0], -1 * f);
			atomicAdd(&S_phi.U[block_cell0], -1 * f * UFace + fude - fudi);
			phi.U[local_mesh_size + nhalos + boundary_cell] = UFace;

			atomicAdd(&A_phi.V[block_cell0], -1 * f);
			atomicAdd(&S_phi.V[block_cell0], -1 * f * VFace + fvde - fvdi);
			phi.V[local_mesh_size + nhalos + boundary_cell] = VFace;

			atomicAdd(&A_phi.W[block_cell0], -1 * f);
			atomicAdd(&S_phi.W[block_cell0], -1 * f * WFace + fwde - fwdi);
			phi.W[local_mesh_size + nhalos + boundary_cell] = WFace;
		}
		else if( boundary_type == OUTLET )
		{
			const vec<double> dUdXac = phi_grad.U[block_cell0];
            const vec<double> dVdXac = phi_grad.V[block_cell0];
            const vec<double> dWdXac = phi_grad.W[block_cell0];

			const double Visac = effective_viscosity;

			const vec<double> Xpn = vec_minus(face_centers[face], cell_centers[faces[face].cell0]);
			
			const double UFace = phi.U[block_cell0];
            const double VFace = phi.V[block_cell0];
            const double WFace = phi.W[block_cell0];

			const double VisFace  = Visac * face_rlencos[face];

			const double sx = face_normals[face].x;
            const double sy = face_normals[face].y;
            const double sz = face_normals[face].z;

			const double fude = Visac * ((dUdXac.x+dUdXac.x)*sx + (dUdXac.y+dVdXac.x)*sy + (dUdXac.z+dWdXac.x)*sz);
			const double fvde = Visac * ((dUdXac.y+dVdXac.x)*sx + (dVdXac.y+dVdXac.y)*sy + (dVdXac.z+dWdXac.y)*sz);
			const double fwde = Visac * ((dUdXac.z+dWdXac.x)*sx + (dWdXac.y+dVdXac.z)*sy + (dWdXac.z+dWdXac.z)*sz);

			const double fudi = VisFace * dot_product( dUdXac , Xpn );
            const double fvdi = VisFace * dot_product( dVdXac , Xpn );
            const double fwdi = VisFace * dot_product( dWdXac , Xpn );

			// !
            // ! by definition points a boundary normal outwards
            // ! therefore an outlet results in a mass flux >= 0.0
            // !
			if( face_mass_fluxes[face] < 0.0 )
			{
				printf("MAIN COMP UVW NEGATIVE OUTFLOW %3.18f\n", face_mass_fluxes[face]);
				face_mass_fluxes[face] = 1e-15;
			}

			const double f = -VisFace + min( face_mass_fluxes[face], 0.0 );


			atomicAdd(&A_phi.U[block_cell0], -1 * f);
            atomicAdd(&S_phi.U[block_cell0], -1 * f * UFace + fude - fudi);
			phi.U[local_mesh_size + nhalos + boundary_cell] = UFace;

			atomicAdd(&A_phi.V[block_cell0], -1 * f);
            atomicAdd(&S_phi.V[block_cell0], -1 * f * VFace + fvde - fvdi);
			phi.V[local_mesh_size + nhalos + boundary_cell] = VFace;

			atomicAdd(&A_phi.W[block_cell0], -1 * f);
            atomicAdd(&S_phi.W[block_cell0], -1 * f * WFace + fwde - fwdi);
			phi.W[local_mesh_size + nhalos + boundary_cell] = WFace;
		}
		else if( boundary_type == WALL )
		{
			const double UFace = 0.;
			const double VFace = 0.;
			const double WFace = 0.;

			const double Visac = effective_viscosity;

			const vec<double> Xpn = vec_minus(face_centers[face], cell_centers[faces[face].cell0]);

			const double coef = Visac * face_rlencos[face];

			vec<double> Up;
			Up.x = phi.U[block_cell0] - UFace;
			Up.y = phi.V[block_cell0] - VFace;
			Up.z = phi.W[block_cell0] - WFace;

			const double dp = dot_product( Up , normalise(face_normals[face]));
			vec<double> Ut  = vec_minus(Up, vec_mult(normalise(face_normals[face]), dp));

			const double Uvel = abs(Ut.x) + abs(Ut.y) + abs(Ut.z);

			vec<double> force;
			if (Uvel > 0.0)
			{
				const double distance_to_face = magnitude(Xpn);
				force = vec_div(vec_mult(vec_mult(Ut, face_areas[face]), Visac), distance_to_face);
			}
			else
			{
				force.x = 0.0;
				force.y = 0.0;
				force.z = 0.0;
			} 

			// TotalForce = TotalForce + Force

            // !               standard
            // !               implicit
            // !                  V
			atomicAdd(&A_phi.U[block_cell0], coef);
			atomicAdd(&A_phi.V[block_cell0], coef);
			atomicAdd(&A_phi.W[block_cell0], coef);

			// !
            // !                    corr.                     expliciet
            // !                  impliciet
            // !                     V                         V
			atomicAdd(&S_phi.U[block_cell0], coef*phi.U[block_cell0] - force.x);
			atomicAdd(&S_phi.V[block_cell0], coef*phi.V[block_cell0] - force.y);
			atomicAdd(&S_phi.W[block_cell0], coef*phi.W[block_cell0] - force.z);

			phi.U[local_mesh_size + nhalos + boundary_cell] = UFace;
			phi.V[local_mesh_size + nhalos + boundary_cell] = VFace;
			phi.W[local_mesh_size + nhalos + boundary_cell] = WFace;
		}
	}
}

__global__ void kernel_apply_forces(uint64_t local_mesh_size, double *cell_densities, double *cell_volumes, phi_vector<double> phi, phi_vector<double> S_phi, phi_vector<vec<double>> phi_grad, double delta, phi_vector<double> A_phi, particle_aos<double> *particle_terms)
{
	const uint64_t cell = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(cell >= local_mesh_size) return;

	// Gravity force (enthalpy)
	const double BodyForce = -0.001*cell_densities[cell]*cell_volumes[cell]*(phi.TEM[cell] - 273);
	const double gravity[3] = {0.0, -9.81, 0.0};

	S_phi.U[cell] += gravity[0]*BodyForce;
	S_phi.V[cell] += gravity[1]*BodyForce;
	S_phi.W[cell] += gravity[2]*BodyForce;

	// Pressure force
	S_phi.U[cell] -= phi_grad.P[cell].x*cell_volumes[cell];
	S_phi.V[cell] -= phi_grad.P[cell].y*cell_volumes[cell];
	S_phi.W[cell] -= phi_grad.P[cell].z*cell_volumes[cell];

	// If Transient and Euler
	const double rdelta = 1.0 / delta;
	const double f = cell_densities[cell] * cell_volumes[cell] * rdelta;
	
	S_phi.U[cell] += f * phi.U[cell];
	S_phi.V[cell] += f * phi.V[cell];
	S_phi.W[cell] += f * phi.W[cell];

	A_phi.U[cell] += f;
	A_phi.V[cell] += f;
	A_phi.W[cell] += f;

	//RHS from particle code
	S_phi.U[cell] += particle_terms[cell].momentum.x;
	S_phi.V[cell] += particle_terms[cell].momentum.y;
	S_phi.W[cell] += particle_terms[cell].momentum.z;
}

__global__ void kernel_setup_sparse_matrix(double URFactor, uint64_t local_mesh_size, int *rows_ptr, int64_t *col_indices, uint64_t local_cells_disp, gpu_Face<uint64_t> *faces, int64_t map_size, uint64_t *boundary_map, uint64_t *boundary_map_values, double *A_phi_component, gpu_Face<double> *face_fields, double *values, double *S_phi_component, double *phi_component, uint64_t mesh_size, uint64_t faces_per_cell, uint64_t *cell_faces, int *nnz)
{
	const uint64_t cell = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(cell >= local_mesh_size) return;
	int tmp_nnz = cell*(faces_per_cell+1);
	rows_ptr[cell] = tmp_nnz;
	col_indices[tmp_nnz] = cell + local_cells_disp;
	tmp_nnz += 1;
	for(uint64_t face = 0; face < faces_per_cell; face++)
    {
    	const uint64_t cell_face = (cell*faces_per_cell) + face;
        const uint64_t true_face = cell_faces[cell_face];
		if(cell == (faces[true_face].cell0 - local_cells_disp))
		{
			const uint64_t block_cell0 = faces[true_face].cell0 - local_cells_disp;
			uint64_t phi_index0;
			if(block_cell0 >= local_mesh_size)
			{
				phi_index0 = boundary_map[faces[face].cell0];
			}
			else
			{
				phi_index0 = block_cell0;
			}
			if(faces[true_face].cell1 >= mesh_size)
			{
				col_indices[tmp_nnz] = 0 + local_cells_disp;
				values[tmp_nnz] = 0.0;
			}
			else
			{
				atomicAdd(&A_phi_component[phi_index0], -1 * face_fields[true_face].cell1);
				col_indices[tmp_nnz] = faces[true_face].cell1;
				values[tmp_nnz] = face_fields[true_face].cell1;
			}
		}
    	else
    	{
        	const uint64_t block_cell1 = faces[true_face].cell1 - local_cells_disp;
	        uint64_t phi_index1;
    	    if(block_cell1 >= local_mesh_size)
        	{
				phi_index1 = boundary_map[faces[face].cell1];
        	}
        	else
       		{
        		phi_index1 = block_cell1;
        	}
            atomicAdd(&A_phi_component[phi_index1], -1 * face_fields[true_face].cell0);
           	col_indices[tmp_nnz] = faces[true_face].cell0;
            values[tmp_nnz] = face_fields[true_face].cell0;
        }
        tmp_nnz += 1;
    }
	if(cell == local_mesh_size-1)
	{
		rows_ptr[local_mesh_size] = tmp_nnz;
	}
    //A_phi_component[cell] *= RURF;
    //values[rows_ptr[cell]] = A_phi_component[cell];
    //S_phi_component[cell] = S_phi_component[cell] + (1.0 - URFactor) * 
	//						A_phi_component[cell] * phi_component[cell];
}

__global__ void kernel_under_relax(double *A_phi_component, uint64_t local_mesh_size, double URFactor, int *rows_ptr, double *values,double *S_phi_component, double *phi_component)
{
	double RURF = 1. / URFactor;
    const uint64_t cell = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(cell >= local_mesh_size) return;
	A_phi_component[cell] *= RURF;	
	values[rows_ptr[cell]] = A_phi_component[cell];
	S_phi_component[cell] = S_phi_component[cell] + (1.0 - URFactor) *
						  A_phi_component[cell] * phi_component[cell];
} 

/*__global__ void kernel_setup_sparse_matrix(double URFactor, uint64_t local_mesh_size, int *rows_ptr, int64_t *col_indices, uint64_t local_cells_disp, gpu_Face<uint64_t> *faces, int64_t map_size, uint64_t *boundary_map, uint64_t *boundary_map_values, double *A_phi_component, gpu_Face<double> *face_fields, double *values, double *S_phi_component, double *phi_component, uint64_t mesh_size, uint64_t faces_per_cell, uint64_t *cell_faces, int *nnz)
{
	double RURF = 1. / URFactor;
	for(uint64_t cell = 0; cell < local_mesh_size; cell++)
	{
		rows_ptr[cell] = *nnz;
		col_indices[*nnz] = cell + local_cells_disp;
		*nnz += 1;
		for(uint64_t face = 0; face < faces_per_cell; face++)
		{
			const uint64_t cell_face = (cell*faces_per_cell) + face;
			const uint64_t true_face = cell_faces[cell_face];
			if (faces[true_face].cell1 >= mesh_size)  continue;
			if(cell == (faces[true_face].cell0 - local_cells_disp))
			{
				const uint64_t block_cell0 = faces[true_face].cell0 - local_cells_disp;
				uint64_t phi_index0;
				if(block_cell0 >= local_mesh_size)
				{
					for(int i = 0; i < map_size; i++)
					{
						if(boundary_map[i] == faces[face].cell0)
						{
							phi_index0 = boundary_map_values[i];
						}
					}
				}
				else
				{
					phi_index0 = block_cell0;
				}
				A_phi_component[phi_index0] -= face_fields[true_face].cell1;
				col_indices[*nnz] = faces[true_face].cell1;
				values[*nnz] = face_fields[true_face].cell1;
			}
			else
			{
				const uint64_t block_cell1 = faces[true_face].cell1 - local_cells_disp;
				uint64_t phi_index1;
				if(block_cell1 >= local_mesh_size)
				{
					for(int i = 0; i < map_size; i++)
					{
						if(boundary_map[i] == faces[face].cell1)
						{
							phi_index1 = boundary_map_values[i];
						}
					}
				}
				else
				{
					phi_index1 = block_cell1;
				}
				A_phi_component[phi_index1] -= face_fields[true_face].cell0;
				col_indices[*nnz] = faces[true_face].cell0;
				values[*nnz] = face_fields[true_face].cell0;
			}
			*nnz += 1;
		}
	}
	rows_ptr[local_mesh_size] = *nnz;
	for (uint64_t i = 0; i < local_mesh_size; i++)
	{
		A_phi_component[i] *= RURF;	
		values[rows_ptr[i]] = A_phi_component[i];
		S_phi_component[i] = S_phi_component[i] + (1.0 - URFactor) * A_phi_component[i] * phi_component[i];
	}
}*/

__global__ void kernel_update_sparse_matrix(double URFactor, uint64_t local_mesh_size, double *A_phi_component, double *values, int *rows_ptr, double *S_phi_component, double *phi_component, uint64_t faces_size, gpu_Face<uint64_t> *faces, int64_t map_size, uint64_t *boundary_map, uint64_t *boundary_map_values, uint64_t local_cells_disp, gpu_Face<double> *face_fields, uint64_t mesh_size)
{
	const uint64_t face = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(face >= faces_size) return;
	if(faces[face].cell1 < mesh_size)
	{
		const uint64_t block_cell0 = faces[face].cell0 - local_cells_disp;
    	const uint64_t block_cell1 = faces[face].cell1 - local_cells_disp;
    	uint64_t phi_index0;
    	if(block_cell0 >= local_mesh_size)
	    {
			phi_index0 = boundary_map[faces[face].cell0];
   		}
    	else
    	{
    		phi_index0 = block_cell0;
    	}
    	uint64_t phi_index1;
    	if(block_cell1 >= local_mesh_size)
    	{
			phi_index1 = boundary_map[faces[face].cell1];
    	}
    	else
    	{
    		phi_index1 = block_cell1;
    	}
    	atomicAdd(&A_phi_component[phi_index0], -1 * face_fields[face].cell1);
    	atomicAdd(&A_phi_component[phi_index1], -1 * face_fields[face].cell0);
	}
}

__global__ void kernel_cell_update(double URFactor, uint64_t local_mesh_size, double *A_phi_component, double *values, int *rows_ptr, double *S_phi_component, double *phi_component)
{ 
	const uint64_t cell = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(cell >= local_mesh_size) return;
	double RURF = 1. / URFactor;
    A_phi_component[cell] *= RURF;
    values[rows_ptr[cell]] = A_phi_component[cell];
    S_phi_component[cell] = S_phi_component[cell] + (1.0 - URFactor) * 
							A_phi_component[cell] * phi_component[cell];
}

/*__global__ void kernel_update_sparse_matrix(double URFactor, uint64_t local_mesh_size, double *A_phi_component, double *values, int *rows_ptr, double *S_phi_component, double *phi_component, uint64_t faces_size, gpu_Face<uint64_t> *faces, int64_t map_size, uint64_t *boundary_map, uint64_t *boundary_map_values, uint64_t local_cells_disp, gpu_Face<double> *face_fields, uint64_t mesh_size)
{
	for(uint64_t face = 0; face < faces_size; face++)
	{
    	if(faces[face].cell1 >= mesh_size)  continue;
        const uint64_t block_cell0 = faces[face].cell0 - local_cells_disp;
		const uint64_t block_cell1 = faces[face].cell1 - local_cells_disp;        
        uint64_t phi_index0;
        if(block_cell0 >= local_mesh_size)
        {
        	for(int i = 0; i < map_size; i++)
            {
	            if(boundary_map[i] == faces[face].cell0)
                {
    	            phi_index0 = boundary_map_values[i];
                }
            }
        }
        else
        {
        	phi_index0 = block_cell0;
        }
        uint64_t phi_index1;
        if(block_cell1 >= local_mesh_size)
        {
        	for(int i = 0; i < map_size; i++)
            {
	            if(boundary_map[i] == faces[face].cell1)
                {
    	            phi_index1 = boundary_map_values[i];
                }
            }
        }
        else
        {
        	phi_index1 = block_cell1;
        }
		A_phi_component[phi_index0] -= face_fields[face].cell1;
		A_phi_component[phi_index1] -= face_fields[face].cell0;
    }
	double RURF = 1. / URFactor;
	for (uint64_t i = 0; i < local_mesh_size; i++)
	{
		A_phi_component[i] *= RURF;
		values[rows_ptr[i]] = A_phi_component[i];
		S_phi_component[i] = S_phi_component[i] + (1.0 - URFactor) * A_phi_component[i] * phi_component[i];
    }	
}*/

__global__ void kernel_print_grads(phi_vector<vec<double>> phi_grad, uint64_t local_mesh_size)
{
	for(uint64_t block_cell = 0; block_cell < local_mesh_size; block_cell++)
	{
		printf("grad for cell %lu is (%3.6f,%3.6f,%3.6f)\n",block_cell, phi_grad.U[block_cell].x, phi_grad.U[block_cell].y, phi_grad.U[block_cell].z);
	}
}

__global__ void kernel_setup_pressure_matrix(uint64_t local_mesh_size, int *rows_ptr, int64_t *col_indices, uint64_t local_cells_disp, gpu_Face<uint64_t> *faces, int64_t map_size, uint64_t *boundary_map, uint64_t *boundary_map_values, gpu_Face<double> *face_fields, double *values, uint64_t mesh_size, uint64_t faces_per_cell, uint64_t *cell_faces, int *nnz, double *face_mass_fluxes, phi_vector<double> A_phi, phi_vector<double> S_phi)
{
	const uint64_t cell = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(cell >= local_mesh_size) return;
	int tmp_nnz = cell*(faces_per_cell+1);
	rows_ptr[cell] = tmp_nnz;
	col_indices[tmp_nnz] = cell + local_cells_disp;
    tmp_nnz += 1;
    for(uint64_t face = 0; face < faces_per_cell; face++)
    {
        const uint64_t cell_face = (cell*faces_per_cell) + face;
        const uint64_t true_face = cell_faces[cell_face];
        if(cell == (faces[true_face].cell0 - local_cells_disp))
        {
            const uint64_t block_cell0 = faces[true_face].cell0 - local_cells_disp;
            uint64_t phi_index0;
            if(block_cell0 >= local_mesh_size)
            {
				phi_index0 = boundary_map[faces[face].cell0];
            }
            else
            {
                phi_index0 = block_cell0;
            }
            if(faces[true_face].cell1 >= mesh_size)
            {
                col_indices[tmp_nnz] = 0 + local_cells_disp;
                values[tmp_nnz] = 0.0;
            }
            else
            {
				atomicAdd(&S_phi.U[phi_index0], -1 * face_mass_fluxes[true_face]);
                atomicAdd(&A_phi.V[phi_index0], -1 * face_fields[true_face].cell1);
                col_indices[tmp_nnz] = faces[true_face].cell1;
                values[tmp_nnz] = face_fields[true_face].cell1;
			}
		}
		else
		{
			const uint64_t block_cell1 = faces[true_face].cell1 - local_cells_disp;
            uint64_t phi_index1;
            if(block_cell1 >= local_mesh_size)
            {
				phi_index1 = boundary_map[faces[face].cell1];
            }
            else
            {
                phi_index1 = block_cell1;
            }
			atomicAdd(&S_phi.U[phi_index1], face_mass_fluxes[true_face]);
            atomicAdd(&A_phi.V[phi_index1], -1 * face_fields[true_face].cell0);
            col_indices[tmp_nnz] = faces[true_face].cell0;
            values[tmp_nnz] = face_fields[true_face].cell0;
		}
		tmp_nnz += 1;
	}
	if(cell == local_mesh_size-1)
    {
        rows_ptr[local_mesh_size] = tmp_nnz;
    }
}
__global__ void apply_diag(uint64_t local_mesh_size, int *rows_ptr, double *values, phi_vector<double> A_phi)
{
	const uint64_t cell = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(cell >= local_mesh_size) return; 
	values[rows_ptr[cell]] = A_phi.V[cell];
}
/*__global__ void kernel_setup_pressure_matrix(uint64_t local_mesh_size, int *rows_ptr, int64_t *col_indices, uint64_t local_cells_disp, gpu_Face<uint64_t> *faces, int64_t map_size, uint64_t *boundary_map, uint64_t *boundary_map_values, gpu_Face<double> *face_fields, double *values, uint64_t mesh_size, uint64_t faces_per_cell, uint64_t *cell_faces, int *nnz, double *face_mass_fluxes, phi_vector<double> A_phi, phi_vector<double> S_phi)
{
    for(uint64_t cell = 0; cell < local_mesh_size; cell++)
    {
        rows_ptr[cell] = *nnz;
        col_indices[*nnz] = cell + local_cells_disp;
        *nnz += 1;
        for(uint64_t face = 0; face < faces_per_cell; face++)
        {
            const uint64_t cell_face = (cell*faces_per_cell) + face;
            const uint64_t true_face = cell_faces[cell_face];
            if (faces[true_face].cell1 >= mesh_size)  continue;
            if(cell == (faces[true_face].cell0 - local_cells_disp))
            {
                const uint64_t block_cell0 = faces[true_face].cell0 - local_cells_disp;
                uint64_t phi_index0;
                if(block_cell0 >= local_mesh_size)
                {
                    for(int i = 0; i < map_size; i++)
                    {
                        if(boundary_map[i] == faces[true_face].cell0)
                        {
                            phi_index0 = boundary_map_values[i];
                        }
                    }
                }
                else
                {
                    phi_index0 = block_cell0;
                }
				S_phi.U[phi_index0] -= face_mass_fluxes[true_face];
                A_phi.V[phi_index0] -= face_fields[true_face].cell1;
                col_indices[*nnz] = faces[true_face].cell1;
                values[*nnz] = face_fields[true_face].cell1;
            }
            else
            {
                const uint64_t block_cell1 = faces[true_face].cell1 - local_cells_disp;
                uint64_t phi_index1;
                if(block_cell1 >= local_mesh_size)
                {
                    for(int i = 0; i < map_size; i++)
                    {
                        if(boundary_map[i] == faces[true_face].cell1)
                        {
                            phi_index1 = boundary_map_values[i];
                        }
                    }
                }
                else
                {
                    phi_index1 = block_cell1;
                }
				S_phi.U[phi_index1] += face_mass_fluxes[true_face];
                A_phi.V[phi_index1] -= face_fields[true_face].cell0;
                col_indices[*nnz] = faces[true_face].cell0;
                values[*nnz] = face_fields[true_face].cell0;
            }
            *nnz += 1;
        }
    }
    rows_ptr[local_mesh_size] = *nnz;
	for (uint64_t i = 0; i < local_mesh_size; i++)
    {
		//force a dominate diagonal to stablise pressure solve 
        values[rows_ptr[i]] = A_phi.V[i] + 10000;
    }
}*/

__global__ void kernel_find_pressure_correction_max(double *Pressure_correction_max, double *phi_component, uint64_t local_mesh_size)
{
	*Pressure_correction_max = phi_component[0];
	for(uint64_t i = 0; i < local_mesh_size; i++)
	{
		if(abs(phi_component[i]) > *Pressure_correction_max)
		{
			*Pressure_correction_max = abs(phi_component[i]);
		}
	}
}

__global__ void kernel_Update_P_at_boundaries(uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t mesh_size, uint64_t local_mesh_size, uint64_t nhalos, double *phi_component)
{
	const uint64_t face = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(face >= faces_size) return;
	const uint64_t block_cell0 = faces[face].cell0 - local_cells_disp;
	if (faces[face].cell1 >= mesh_size)
	{
		const uint64_t boundary_cell = faces[face].cell1 - mesh_size;
		phi_component[local_mesh_size + nhalos + boundary_cell] = phi_component[block_cell0];
	}
}

__global__ void kernel_update_vel_and_flux(uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t local_mesh_size, uint64_t nhalos, gpu_Face<double> *face_fields, uint64_t mesh_size, int64_t map_size, uint64_t *boundary_map, uint64_t *boundary_map_values, double *face_mass_fluxes, phi_vector<double> A_phi, phi_vector<double> phi, double *cell_volumes, phi_vector<vec<double>> phi_grad, int timestep_count)
{
	const uint64_t face = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(face >= faces_size) return;
	const uint64_t block_cell0 = faces[face].cell0 - local_cells_disp;
	const uint64_t block_cell1 = faces[face].cell1 - local_cells_disp;

	if(faces[face].cell1 < mesh_size)
	{
		uint64_t phi_index0;
        if(block_cell0 >= local_mesh_size)
        {
			phi_index0 = boundary_map[faces[face].cell0];
        }
        else
        {
            phi_index0 = block_cell0;
        }
		uint64_t phi_index1;
        if(block_cell1 >= local_mesh_size)
        {
			phi_index1 = boundary_map[faces[face].cell1];
        }
        else
        {
            phi_index1 = block_cell1;
        }
		face_mass_fluxes[face] += (face_fields[face].cell0*(phi.PP[phi_index1] - phi.PP[phi_index0]));
	}
	if(face >= (local_mesh_size + nhalos)) return;
	double Ar = (A_phi.U[face] != 0.0) ? 1.0 / A_phi.U[face] : 0.0;
	double fact = cell_volumes[face] * Ar;
		
        if (timestep_count == 0)
        {
                phi.P[face] += 0.2*phi.PP[face];
	
		phi.U[face] -= phi_grad.PP[face].x * fact;
		phi.V[face] -= phi_grad.PP[face].y * fact;
		phi.W[face] -= phi_grad.PP[face].z * fact;
        }
}

__global__ void kernel_Update_P(uint64_t faces_size, uint64_t local_mesh_size, uint64_t nhalos, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t mesh_size, vec<double> *cell_centers, vec<double> *face_centers, uint64_t *boundary_types, double *phi_component, vec<double> *phi_grad_component)
{
	const uint64_t face = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(face >= faces_size) return;
	const uint64_t block_cell0 = faces[face].cell0 - local_cells_disp;
	if(faces[face].cell1 >= mesh_size)
	{
		const uint64_t boundary_cell = faces[face].cell1 - mesh_size;
		const uint64_t boundary_type = boundary_types[boundary_cell];
		if(boundary_type == OUTLET)
		{
			phi_component[local_mesh_size + nhalos + boundary_cell] = phi_component[block_cell0];
		}
		else
		{
			vec<double> ds = vec_minus(face_centers[face], cell_centers[faces[face].cell0]);
			phi_component[local_mesh_size + nhalos + boundary_cell] = phi_component[block_cell0] + dot_product(phi_grad_component[block_cell0], ds);
		}
	}
}

__global__ void kernel_set_up_fgm_table(double *fgm_table, uint64_t seed)
{
	const uint64_t id = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(id >= 100*100*100*100) return;
	curandState state;
	curand_init(seed, id, 0, &state);
	fgm_table[id] = curand_uniform(&state);
}

__global__ void kernel_fgm_look_up(double *fgm_table, phi_vector<double> S_phi, phi_vector<double> phi, uint64_t local_mesh_size)
{
	const uint64_t cell = (blockIdx.x * blockDim.x) + threadIdx.x;
	if( cell >= local_mesh_size) return;
	//find location in table based on variables
    //simulate finding the closest two points in the database
    int progress_1 = max(0, min(99, (int) floor(phi.PRO[cell]*100)));
    int progress_2 = max(0, min(99, (int) ceil(phi.PRO[cell]*100)));
    int var_progress_1 = max(0, min(99, (int) floor(phi.VARP[cell]*100)));
    int var_progress_2 = max(0, min(99, (int) ceil(phi.VARP[cell]*100)));
    int fuel_1 = max(0, min(99, (int) floor(phi.FUL[cell]*100)));
    int fuel_2 = max(0, min(99, (int) ceil(phi.FUL[cell]*100)));
    int var_fuel_1 = max(0, min(99, (int) floor(phi.VARF[cell]*100)));
    int var_fuel_2 = max(0, min(99, (int) ceil(phi.VARF[cell]*100)));

	//interpolate table values to find given value
    //simulate using the average of the 16
    double sum = 0;
    sum += fgm_table[IDx(progress_1, var_progress_1, fuel_1, var_fuel_1)];
	sum += fgm_table[IDx(progress_1, var_progress_1, fuel_1, var_fuel_2)];
	sum += fgm_table[IDx(progress_1, var_progress_1, fuel_2, var_fuel_1)];
	sum += fgm_table[IDx(progress_1, var_progress_1, fuel_2, var_fuel_2)];
	sum += fgm_table[IDx(progress_1, var_progress_2, fuel_1, var_fuel_1)];
	sum += fgm_table[IDx(progress_1, var_progress_2, fuel_1, var_fuel_2)];
	sum += fgm_table[IDx(progress_1, var_progress_2, fuel_2, var_fuel_1)];
	sum += fgm_table[IDx(progress_1, var_progress_2, fuel_2, var_fuel_2)];
	sum += fgm_table[IDx(progress_2, var_progress_1, fuel_1, var_fuel_1)];
	sum += fgm_table[IDx(progress_2, var_progress_1, fuel_1, var_fuel_2)];
	sum += fgm_table[IDx(progress_2, var_progress_1, fuel_2, var_fuel_1)];
	sum += fgm_table[IDx(progress_2, var_progress_1, fuel_2, var_fuel_2)];
	sum += fgm_table[IDx(progress_2, var_progress_2, fuel_1, var_fuel_1)];
	sum += fgm_table[IDx(progress_2, var_progress_2, fuel_1, var_fuel_2)];
	sum += fgm_table[IDx(progress_2, var_progress_2, fuel_2, var_fuel_1)];
	sum += fgm_table[IDx(progress_2, var_progress_2, fuel_2, var_fuel_2)];

	//this would give us same values to be used as source terms
    //in our code this will be thrown away.
    sum /= 16;
    S_phi.U[cell] = sum;
}

__global__ void kernel_flux_scalar(int type, uint64_t faces_size, uint64_t local_mesh_size, uint64_t nhalos, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t mesh_size, vec<double> *cell_centers, vec<double> *face_centers, uint64_t *boundary_types, uint64_t *boundary_map, uint64_t *boundary_map_values, int64_t map_size, double *phi_component, phi_vector<double> A_phi, phi_vector<double> S_phi, vec<double> *phi_grad_component, double *face_lambdas, double effective_viscosity, double *face_rlencos, double *face_mass_fluxes, vec<double> *face_normals, double inlet_effective_viscosity, gpu_Face<double> *face_fields, vec<double> dummy_gas_vel, double dummy_gas_tem, double dummy_gas_fuel)
{
	const uint64_t face = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(face >= faces_size) return;

	double GammaBlend = 0.0;

	const uint64_t block_cell0 = faces[face].cell0 - local_cells_disp;
	const uint64_t block_cell1 = faces[face].cell1 - local_cells_disp;

	if(faces[face].cell1 < mesh_size) //INTERNAL
	{
		uint64_t phi_index0;
        if(block_cell0 >= local_mesh_size)
        {
			phi_index0 = boundary_map[faces[face].cell0];
        }
        else
        {
            phi_index0 = block_cell0;
        }
        uint64_t phi_index1;
        if(block_cell1 >= local_mesh_size)
        {
			phi_index1 = boundary_map[faces[face].cell1];
        }
        else
        {
            phi_index1 = block_cell1;
        }
		const double lambda0 = face_lambdas[face];
		const double lambda1 = 1.0 - lambda0;

		double Visac = effective_viscosity * lambda0 + effective_viscosity * lambda1;

		Visac -= effective_viscosity;

		if(type == TEMP)
        {
            Visac = (effective_viscosity + Visac / 0.9) / 0.6905;
        }
        else if(type  == TERBTE)
        {
            Visac = effective_viscosity + Visac;
        }
        else if(type == TERBED)
        {
            Visac = effective_viscosity + Visac / 1.219;
        }
        else
        {
            Visac = (effective_viscosity + Visac / 0.9) / 0.9;
        }

		vec<double> dPhiXac = vec_add(vec_mult(phi_grad_component[phi_index0], lambda0), vec_mult(phi_grad_component[phi_index1], lambda1));
	
		vec<double> Xpn = vec_minus(cell_centers[faces[face].cell1], cell_centers[faces[face].cell0]);

		const double VisFace = Visac * face_rlencos[face];
	
		double PhiFace;

		if ( face_mass_fluxes[face] >= 0.0 )
        {
            PhiFace  = phi_component[phi_index0];
        }
        else
        {
            PhiFace  = phi_component[phi_index1];
        }

		// explicit higher order convective flux (see eg. eq. 8.16)
		const double fce = face_mass_fluxes[face] * PhiFace;
		const double fde1 = Visac * dot_product ( dPhiXac , face_normals[face] );

		//implicit lower order (simple upwind)
        //convective and diffusive fluxes
		const double  fci = min( face_mass_fluxes[face], 0.0 ) * phi_component[phi_index0] + max( face_mass_fluxes[face], 0.0 ) * phi_component[phi_index1];

		const double fdi = VisFace * dot_product( dPhiXac , Xpn );

		// !
        // ! convective coefficients with deferred correction with
        // ! gamma as the blending factor (0.0 <= gamma <= 1.0)
        // !
        // !      low            high    low  OLD
        // ! F = F    + gamma ( F     - F    )
        // !     ----   -------------------------
        // !      |                  |
        // !  implicit           explicit (dump into source term)
        // !
		face_fields[face].cell0 = -VisFace - max( face_mass_fluxes[face] , 0.0 );
		face_fields[face].cell1 = -VisFace + min( face_mass_fluxes[face] , 0.0 );

		const double blend = GammaBlend * ( fce - fci );
		
		atomicAdd(&S_phi.U[phi_index0], fde1 - blend - fdi);
        atomicAdd(&S_phi.U[phi_index1], blend - fde1 + fdi);
	}
	else //BOUNDARY
	{
		// Boundary faces
		const uint64_t boundary_cell = faces[face].cell1 - mesh_size;
        const uint64_t boundary_type = boundary_types[boundary_cell];

		if ( boundary_type == INLET )
		{
			// Option to add more inlet region information and functions here.
			const vec<double> dPhidXac = phi_grad_component[block_cell0];

			double PhiFace;
			if(type == TEMP)
            {
                PhiFace = dummy_gas_tem;
            }
            else if(type  == TERBTE)
            {
                double velmag2 = pow(dummy_gas_vel.x,2) + pow(dummy_gas_vel.y,2) + pow(dummy_gas_vel.z,2);
                PhiFace = 3.0/2.0*((0.1*0.1)*velmag2);;
            }
            else if(type == TERBED)
            {
                double velmag2 = pow(dummy_gas_vel.x,2) + pow(dummy_gas_vel.y,2) + pow(dummy_gas_vel.z,2);
                PhiFace = pow(0.09,0.75) * pow((3.0/2.0*((0.1*0.1)*velmag2)),1.5);
            }
            else if(type == FUEL)
            {
                PhiFace = dummy_gas_fuel;
            }
            else if(type == PROG)
            {
                PhiFace = 0.0;
            }
            else if(type == VARFU)
            {
                PhiFace = dummy_gas_fuel;
            }
            else if(type == VARPR)
            {
                PhiFace = 0.0;
            }

			double Visac = inlet_effective_viscosity;
			
			Visac -= effective_viscosity; //This will always be 0 right?

			if(type == TEMP)
            {
                Visac = (effective_viscosity + Visac / 0.9) / 0.6905;
            }
            else if(type  == TERBTE)
            {
                Visac = effective_viscosity + Visac;
            }
            else if(type == TERBED)
            {
                Visac = effective_viscosity + Visac / 1.219;
            }
            else
            {
                Visac = (effective_viscosity + Visac / 0.9) / 0.9;
            }

			vec<double> Xpn = vec_minus(face_centers[face], cell_centers[faces[face].cell0]);
			const double VisFace  = Visac * face_rlencos[face];
			
			const double fde = Visac * dot_product( dPhidXac , face_normals[face]);
		
			//implicit part
			const double fdi = VisFace * dot_product( dPhidXac, Xpn);

			const double f = -VisFace + min( face_mass_fluxes[face], 0.0 );

			atomicAdd(&A_phi.V[block_cell0], -1 * f);
			atomicAdd(&S_phi.U[block_cell0], -1 * f * PhiFace + fde - fdi);

			phi_component[local_mesh_size + nhalos + boundary_cell] = PhiFace;
		}
		else if( boundary_type == OUTLET )
		{
			const vec<double> dPhidXac = phi_grad_component[block_cell0];
			
			double Visac = effective_viscosity;
		
			Visac -= effective_viscosity; //This will always be 0 right?

			if(type == TEMP)
            {
                Visac = (effective_viscosity + Visac / 0.9) / 0.6905;
            }
            else if(type  == TERBTE)
            {
                Visac = effective_viscosity + Visac;
            }
            else if(type == TERBED)
            {
                Visac = effective_viscosity + Visac / 1.219;
            }
            else
            {
                Visac = (effective_viscosity + Visac / 0.9) / 0.9;
            }

			const vec<double> Xpn = vec_minus(face_centers[face], cell_centers[faces[face].cell0]);

			const double PhiFace = phi_component[block_cell0] + dot_product( dPhidXac , Xpn );
			const double VisFace  = Visac * face_rlencos[face];
			
			const double fde = Visac * dot_product( dPhidXac , face_normals[face] );
		
			const double fdi = VisFace * dot_product( dPhidXac , Xpn );

			atomicAdd(&S_phi.U[block_cell0], fde - fdi);
	
			phi_component[local_mesh_size + nhalos + boundary_cell] = PhiFace;
		}
		else if( boundary_type == WALL )
		{
			if( type != TERBTE or type != TERBED )
			{
				phi_component[local_mesh_size + nhalos + boundary_cell] = phi_component[block_cell0];
			}
		}
	}
}

__global__ void kernel_apply_pres_forces(int type, uint64_t local_mesh_size, double delta, double *cell_densities, double *cell_volumes, double *phi_component, phi_vector<double> A_phi, phi_vector<double> S_phi, particle_aos<double> *particle_terms)
{
	const uint64_t cell = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(cell >= local_mesh_size) return;
	
	const double rdelta = 1.0/delta;

	//unsteady term
	double f = cell_densities[cell] * cell_volumes[cell] * rdelta;
	S_phi.U[cell] += f * phi_component[cell];
	A_phi.V[cell] += f;

	if(type == TEMP)
	{
		S_phi.U[cell] += particle_terms[cell].energy;
	}
}

__global__ void kernel_solve_turb_models_cell(int type, uint64_t local_mesh_size, phi_vector<double> A_phi, phi_vector<double> S_phi, phi_vector<vec<double>> phi_grad, phi_vector<double> phi, double effective_viscosity, double *cell_densities, double *cell_volumes)
{
	const uint64_t cell = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(cell >= local_mesh_size) return;

	if(type == TERBTE)
	{
		const vec<double> dUdXp = phi_grad.U[cell];
		const vec<double> dVdXp = phi_grad.V[cell];
		const vec<double> dWdXp = phi_grad.W[cell];

		const double s1 = (dUdXp.x+dUdXp.x)*dUdXp.x + (dUdXp.y+dVdXp.x)*dUdXp.y + (dUdXp.z+dWdXp.x)*dUdXp.z;
		const double s2 = (dVdXp.x+dUdXp.y)*dVdXp.x + (dVdXp.y+dVdXp.y)*dVdXp.y + (dVdXp.z+dWdXp.y)*dVdXp.z;
		const double s3 = (dWdXp.x+dUdXp.z)*dWdXp.x + (dWdXp.y+dVdXp.z)*dWdXp.y + (dWdXp.z+dWdXp.z)*dWdXp.z;

		double VisT = effective_viscosity - effective_viscosity;
		
		double Pk = VisT * (s1 + s2 + s3);
	
		phi.TP[cell] = Pk;

		double Dis = cell_densities[cell] * phi.ED[cell];

		S_phi.U[cell] = S_phi.U[cell] + phi.TP[cell] * cell_volumes[cell];
		A_phi.V[cell] = A_phi.V[cell] + Dis / (phi.TE[cell] + 0.000000000000000001) * cell_volumes[cell];
	}
	else if(type == TERBED)
	{
		double fact = phi.ED[cell]/(phi.TE[cell]+0.000000000000000001) * cell_volumes[cell];
		S_phi.U[cell] = S_phi.U[cell] + 1.44 * fact * phi.TP[cell];
		A_phi.V[cell] = A_phi.V[cell] + 1.92 * fact * cell_densities[cell];
	}
}

__global__ void kernel_solve_turb_models_face(int type, uint64_t faces_size, uint64_t mesh_size, double *cell_volumes, double effective_viscosity, vec<double> *face_centers, vec<double> *cell_centers, vec<double> *face_normals, double *cell_densities, uint64_t local_mesh_size, uint64_t nhalos, uint64_t local_cells_disp, phi_vector<double> A_phi, phi_vector<double> S_phi, phi_vector<double> phi, gpu_Face<double> *face_fields, gpu_Face<uint64_t> *faces, uint64_t *boundary_types, uint64_t faces_per_cell, uint64_t *cell_faces)
{
	const uint64_t face = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(face >= faces_size) return;
	if(type == TERBTE)
	{
		double Cmu = 0.09;
		double Cmu75 = pow(Cmu, 0.75);
		
		const uint64_t block_cell0 = faces[face].cell0 - local_cells_disp;
		
		if(faces[face].cell1 >= mesh_size)
		{
			//only need the boundary cells
			const uint64_t boundary_cell = faces[face].cell1 - mesh_size;
			const uint64_t boundary_type = boundary_types[boundary_cell];
			
			if(boundary_type == WALL)
			{
				//at walls we need a different source term
                S_phi.U[block_cell0] = S_phi.U[block_cell0] - phi.TP[block_cell0] * cell_volumes[block_cell0];
				const double UFace = 0.; // Customisable (add regions here later)
				const double VFace = 0.; // Customisable (add regions here later)
				const double WFace = 0.; // Customisable (add regions here later)

				const double Visc = effective_viscosity;

				const vec<double> Xpn = vec_minus(face_centers[face], cell_centers[faces[face].cell0]);
				
				vec<double> Up;
				Up.x = phi.U[block_cell0] - UFace;
				Up.y = phi.V[block_cell0] - VFace;
				Up.z = phi.W[block_cell0] - WFace;

				const double dp = dot_product( Up , normalise(face_normals[face]));

				vec<double> Ut = vec_minus(Up, vec_mult(normalise(face_normals[face]), dp));

				const double Uvel = sqrt(dot_product( Ut, Ut));
				const double distance_to_face = magnitude(Xpn);

				//production of in wall region
				const double rkapdn = 1.0/( 0.419 * distance_to_face);

				//if yplus > ylog we only have less than implemented
				const double Tau_w = Visc * Uvel / distance_to_face;
				const double Utau = sqrt( Tau_w / cell_densities[block_cell0]);

				phi.TP[block_cell0] = Tau_w * Utau * rkapdn;
				phi.TE[local_mesh_size + nhalos + boundary_cell] = phi.TE[block_cell0];

				atomicAdd(&S_phi.U[block_cell0], phi.TP[block_cell0] * cell_volumes[block_cell0]);
				
				//dissipation term
				double DisP = Cmu75*sqrt(phi.TE[block_cell0])*rkapdn;

				atomicAdd(&A_phi.V[block_cell0], cell_densities[block_cell0] * DisP * cell_volumes[block_cell0]);
			}
		}
	}
	else if(type == TERBED)
	{
		const uint64_t block_cell0 = faces[face].cell0 - local_cells_disp;
		double Cmu = 0.09;
		double Cmu75 = pow(Cmu, 0.75);
		if(faces[face].cell1 >= mesh_size)
		{
        	//only need the boundary cells
			const uint64_t boundary_cell = faces[face].cell1 - mesh_size;
            const uint64_t boundary_type = boundary_types[boundary_cell];
				
			if(boundary_type == WALL)
			{
				const double turb = phi.TE[block_cell0];
				const double distance = magnitude(vec_minus(face_centers[face], cell_centers[faces[face].cell0]));
				
				const double Dis = Cmu75 * pow(turb,1.5) / ( distance * 0.419 );
					
				for(uint64_t j = 0; j < faces_per_cell; j++)
				{
					uint64_t neigh_face = cell_faces[(block_cell0 * faces_per_cell) + j];
					if((faces[neigh_face].cell0 < mesh_size) and (faces[neigh_face].cell1 < mesh_size))
					{
						if((faces[neigh_face].cell1 - local_cells_disp) >= local_mesh_size)
						{
							face_fields[neigh_face].cell0 = 0.0;
							face_fields[neigh_face].cell1 = 0.0;
						}
						//internal node
						if((faces[neigh_face].cell0 - local_cells_disp) == block_cell0)
						{
							face_fields[neigh_face].cell1 = 0.0;
						}
						else if((faces[neigh_face].cell1 - local_cells_disp) == block_cell0)
						{
							face_fields[neigh_face].cell0 = 0.0;
						}
					}
				}
				phi.ED[block_cell0] = Dis;
				S_phi.U[block_cell0] = Dis;
				A_phi.V[block_cell0] = 1;
				phi.ED[local_mesh_size + nhalos + boundary_cell] = phi.ED[block_cell0];
			}
		}
	}
}

__global__ void kernel_test_values(int *nnz, double *values, int *rows_ptr, int64_t *col_indices, uint64_t local_mesh_size, double *b, double *u)
{
    printf("nnz is %d\n", *nnz);
    printf("rows is: ");
    for(int i = 0; i < local_mesh_size; i++)
    {
        printf("%d, ", rows_ptr[i]);
    }
    printf("%d\n", rows_ptr[local_mesh_size]);

    printf("cols is: ");
    for(int i = 0; i < *nnz; i++)
    {
        printf("%ld, ", col_indices[i]);
    }
    printf("\n");

    printf("values is: ");
    for(int i = 0; i < *nnz; i++)
    {
        printf("%6.18f, ", values[i]);
    }
    printf("\n");
    printf("RHS is: ");
    for(int i = 0; i < local_mesh_size; i++)
    {
        printf("%6.18f, ", b[i]);
    }
    printf("\n");
    printf("SOL is: ");
    for(int i = 0; i < local_mesh_size; i++)
    {
        printf("%6.18f, ", u[i]);
    }
    printf("\n");
}


__global__ void kernel_test_values_less(int *nnz, double *values, int *rows_ptr, int64_t *col_indices, uint64_t local_mesh_size)
{
	printf("nnz is %d\n", *nnz);
	printf("rows is: ");
	for(int i = 0; i < local_mesh_size; i++)
	{
		printf("%d, ", rows_ptr[i]);
	}
	printf("%d\n", rows_ptr[local_mesh_size]);
	
	printf("cols is: ");
	for(int i = 0; i < *nnz; i++)
	{
		printf("%ld, ", col_indices[i]);
	}
	printf("\n");

	printf("values is: ");
	for(int i = 0; i < *nnz; i++)
	{
		printf("%6.18f, ", values[i]);
	}
	printf("\n");
}

__global__ void kernel_print(double *to_print, uint64_t num_print)
{
	for(uint64_t i = 0; i < num_print; i++)
	{
		printf("%3.18f, ",to_print[i]);
	}
	printf("\n");
}

__global__ void kernel_test_particle_terms(particle_aos<double> *particle_terms, uint64_t local_mesh_size)
{
	for(uint64_t i = 0; i < local_mesh_size; i++)
    {
    	printf("gpu terms for cell %lu (%3.18f,%3.18f,%3.18f)\n", i, particle_terms[i].momentum.x, particle_terms[i].momentum.y, particle_terms[i].momentum.z);
    }
}

__global__ void kernel_vec_print(vec<double> *to_print, uint64_t num_print)
{
	for(uint64_t i = 0; i < num_print; i++)
	{
		printf("(%3.18f,%3.18f,%3.18f)\n",to_print[i].x,to_print[i].y,to_print[i].z);
	}
	printf("\n");
}

__global__ void kernel_interpolate_phi_to_nodes(phi_vector<double> phi, phi_vector<vec<double>> phi_grad, phi_vector<double> phi_nodes, vec<double> *points, uint64_t *points_map, uint8_t *cells_per_point, uint64_t *cells, vec<double> *cell_centers, uint64_t local_mesh_size, uint64_t cell_disp, uint64_t local_points_size)
{
  	auto phi_cell = threadIdx.x + blockIdx.x * blockDim.x;
    if (phi_cell >= local_mesh_size) return;

	const uint64_t cell_size = 8;
	const uint64_t node_neighbours = 8;

	if (phi_cell >= local_mesh_size) // Need a mapping from halos to global_nodes
	{
		// phi_cell is a halo index?

		// How do we get the correct node id
	}

	// Every local cell and every halo contributes to the interpolation accumulation.

	const vec<double> cell_center = cell_centers[phi_cell + cell_disp];

	// printf("cell center (%f,%f,%f)\n",cell_center.x,cell_center.y,cell_center.z);

	for (uint64_t n = 0; n < cell_size; n++)
	{
		const uint64_t node_id = cells[phi_cell*cell_size + n];
		// printf("node_id (%d) phi_nodes.U[node_id] %f  points[node_id] %f, %f %f\n", node_id, phi_nodes.U[node_id], points[node_id].x, points[node_id].y, points[node_id].z);

		const vec<double> direction      = vec_minus(points[node_id], cell_center);
	
		phi_nodes.U[points_map[node_id]]   += (phi.U[phi_cell]   + dot_product(phi_grad.U[phi_cell],   direction)) / node_neighbours;
		phi_nodes.V[points_map[node_id]]   += (phi.V[phi_cell]   + dot_product(phi_grad.V[phi_cell],   direction)) / node_neighbours;
		phi_nodes.W[points_map[node_id]]   += (phi.W[phi_cell]   + dot_product(phi_grad.W[phi_cell],   direction)) / node_neighbours;
		phi_nodes.P[points_map[node_id]]   += (phi.P[phi_cell]   + dot_product(phi_grad.P[phi_cell],   direction)) / node_neighbours;
		phi_nodes.TEM[points_map[node_id]] += (phi.TEM[phi_cell] + dot_product(phi_grad.TEM[phi_cell], direction)) / node_neighbours;
	}
}

__global__ void kernel_interpolate_init_boundaries(phi_vector<double> phi_nodes, uint8_t *cells_per_point, uint64_t local_points_size)
{
	auto node_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (node_id >= local_points_size) return;

	const uint64_t node_neighbours = 8;

	// TODO THESE SHOULD BE MESH DUMMY TERMS

	phi_nodes.U[node_id]   = (node_neighbours - cells_per_point[node_id]) * 50.0  / node_neighbours;
	phi_nodes.V[node_id]   = (node_neighbours - cells_per_point[node_id]) * 0.0   / node_neighbours;
	phi_nodes.W[node_id]   = (node_neighbours - cells_per_point[node_id]) * 0.0   / node_neighbours;
	phi_nodes.P[node_id]   = (node_neighbours - cells_per_point[node_id]) * 100.0 / node_neighbours;
	phi_nodes.TEM[node_id] = (node_neighbours - cells_per_point[node_id]) * 273.0 / node_neighbours;
}

void C_kernel_interpolate_init_boundaries(int block_count, int thread_count, phi_vector<double> phi_nodes, uint8_t *cells_per_point, uint64_t local_points_size)
{
	kernel_interpolate_init_boundaries<<<block_count, thread_count>>> (phi_nodes, cells_per_point, local_points_size);
}

void C_kernel_interpolate_phi_to_nodes(int block_count, int thread_count, phi_vector<double> phi, phi_vector<vec<double>> phi_grad, phi_vector<double> phi_nodes, vec<double> *points, uint64_t *points_map, uint8_t *cells_per_point, uint64_t *cells, vec<double> *cell_centers, uint64_t local_mesh_size, uint64_t cell_disp, uint64_t local_points_size)
{
	kernel_interpolate_phi_to_nodes<<<block_count, thread_count>>> (phi, phi_grad, phi_nodes, points, points_map, cells_per_point, cells, cell_centers, local_mesh_size, cell_disp, local_points_size);
}

void C_kernel_vec_print(vec<double> *to_print, uint64_t num_print)
{
	kernel_vec_print<<<1,1>>>(to_print, num_print);
}

void C_kernel_test_particle_terms(particle_aos<double> *particle_terms, uint64_t local_mesh_size)
{
	kernel_test_particle_terms<<<1,1>>>(particle_terms, local_mesh_size);
}

void C_kernel_print(double *to_print, uint64_t num_print)
{
	kernel_print<<<1,1>>>(to_print, num_print);
}

void C_kernel_test_values(int *nnz, double *values, int *rows_ptr, int64_t *col_indices, uint64_t local_mesh_size, double *b, double *u)
{
	kernel_test_values<<<1,1>>>(nnz, values, rows_ptr, col_indices, local_mesh_size, b, u);
}

void C_kernel_test_values(int *nnz, double *values, int *rows_ptr, int64_t *col_indices, uint64_t local_mesh_size)
{
    kernel_test_values_less<<<1,1>>>(nnz, values, rows_ptr, col_indices, local_mesh_size);
}

__global__ void kernel_update_mass_flux(uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t local_mesh_size, uint64_t mesh_size, uint64_t *boundary_map, uint64_t *boundary_map_values, int64_t map_size, vec<double> *face_centers, vec<double> *cell_centers, gpu_Face<double> *face_fields, phi_vector<vec<double>> phi_grad, double *face_mass_fluxes, phi_vector<double> S_phi, vec<double> *face_normals)
{
	const uint64_t face = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(face >= faces_size) return;
	
	const uint64_t block_cell0 = faces[face].cell0 - local_cells_disp;
    const uint64_t block_cell1 = faces[face].cell1 - local_cells_disp;

	if(faces[face].cell1 < mesh_size)
	{
		//internel
		uint64_t phi_index0;
		if(block_cell0 >= local_mesh_size)
		{
			phi_index0 = boundary_map[faces[face].cell0];
		}
		else
		{
			phi_index0 = block_cell0;
		}
		uint64_t phi_index1;
		if(block_cell1 >= local_mesh_size)
		{
			phi_index1 = boundary_map[faces[face].cell1];

		}
		else
		{
			phi_index1 = block_cell1;
		}
		const vec<double> Xpac = vec_minus(face_centers[face],
                                 vec_mult(normalise(face_normals[face]),
                                          dot_product(vec_minus(face_centers[face],
                                          cell_centers[faces[face].cell0]),
                                          normalise(face_normals[face]))));
		const vec<double> Xnac = vec_minus(face_centers[face],
                                 vec_mult(normalise(face_normals[face]),
                                          dot_product(vec_minus(face_centers[face],
                                          cell_centers[faces[face].cell1]),
                                          normalise(face_normals[face]))));

		vec<double> Xn = vec_minus(Xnac, cell_centers[faces[face].cell1]);
		vec<double> Xp = vec_minus(Xpac, cell_centers[faces[face].cell0]);

		double fact = face_fields[face].cell0;		

		const double dpx  = phi_grad.PP[phi_index1].x * Xn.x - phi_grad.PP[phi_index0].x * Xp.x;
		const double dpy  = phi_grad.PP[phi_index1].y * Xn.y - phi_grad.PP[phi_index0].y * Xp.y;
		const double dpz  = phi_grad.PP[phi_index1].z * Xn.z - phi_grad.PP[phi_index0].z * Xp.z;

		const double fc = fact * (dpx + dpy + dpz) * 0.8;
	
		face_mass_fluxes[face] += fc;
		
		atomicAdd(&S_phi.U[phi_index0], -1*fc);
		atomicAdd(&S_phi.U[phi_index1], fc);
	}
}

void C_kernel_update_mass_flux(int block_count, int thread_count, uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t local_mesh_size, uint64_t mesh_size, uint64_t *boundary_map, uint64_t *boundary_map_values, int64_t map_size, vec<double> *face_centers, vec<double> *cell_centers, gpu_Face<double> *face_fields, phi_vector<vec<double>> phi_grad, double *face_mass_fluxes, phi_vector<double> S_phi, vec<double> *face_normals)
{
	kernel_update_mass_flux<<<block_count,thread_count>>>(faces_size, faces, local_cells_disp, local_mesh_size, mesh_size, boundary_map, boundary_map_values, map_size, face_centers, cell_centers, face_fields, phi_grad, face_mass_fluxes, S_phi, face_normals);
}

void C_kernel_precomp_AU(int block_count, int thread_count, uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t mesh_size, uint64_t *boundary_types, double effective_viscosity, double * face_rlencos, double *face_mass_fluxes, phi_vector<double> A_phi, uint64_t local_mesh_size, double delta, double *cell_densities, double* cell_volumes) 
{
	kernel_precomp_AU<<<block_count,thread_count>>>(faces_size, faces, local_cells_disp, mesh_size, boundary_types, effective_viscosity, face_rlencos, face_mass_fluxes, A_phi, local_mesh_size, delta, cell_densities, cell_volumes);
}

void C_kernel_get_phi_gradients(int block_count, int thread_count, phi_vector<double> phi, phi_vector<vec<double>> phi_grad, uint64_t local_mesh_size, uint64_t local_cells_disp, uint64_t faces_per_cell, gpu_Face<uint64_t> *faces, uint64_t *cell_faces, vec<double> *cell_centers, uint64_t mesh_size, uint64_t *boundary_map, uint64_t *boundary_map_values, int64_t map_size, vec<double> *face_centers, uint64_t nhalos)
{ 

	kernel_get_phi_gradients<<<block_count,thread_count>>>(phi, phi_grad, local_mesh_size, local_cells_disp, faces_per_cell, faces, cell_faces, cell_centers, mesh_size, boundary_map, boundary_map_values, map_size, face_centers, nhalos);
	size_t free, total;
	cudaMemGetInfo( &free, &total );

	// if (free < 0 )
	// // if (free < 2'000'000'000 )
	// {
	// 	printf("Warning: Operating on phi gradients to reduce memory. This is a hack and should be removed.\n");
	// 	kernel_get_phi_gradient<<<block_count,thread_count>>>(phi.U, false, local_mesh_size, local_cells_disp, faces_per_cell, faces, cell_faces, cell_centers, mesh_size, boundary_map, boundary_map_values, map_size, face_centers, nhalos, phi_grad.U);
	// 	kernel_get_phi_gradient<<<block_count,thread_count>>>(phi.V, false, local_mesh_size, local_cells_disp, faces_per_cell, faces, cell_faces, cell_centers, mesh_size, boundary_map, boundary_map_values, map_size, face_centers, nhalos, phi_grad.V);
	// 	kernel_get_phi_gradient<<<block_count,thread_count>>>(phi.W, false, local_mesh_size, local_cells_disp, faces_per_cell, faces, cell_faces, cell_centers, mesh_size, boundary_map, boundary_map_values, map_size, face_centers, nhalos, phi_grad.W);
	// 	kernel_get_phi_gradient<<<block_count,thread_count>>>(phi.P, true,  local_mesh_size, local_cells_disp, faces_per_cell, faces, cell_faces, cell_centers, mesh_size, boundary_map, boundary_map_values, map_size, face_centers, nhalos, phi_grad.P);
	// 	kernel_get_phi_gradient<<<block_count,thread_count>>>(phi.TE, false, local_mesh_size, local_cells_disp, faces_per_cell, faces, cell_faces, cell_centers, mesh_size, boundary_map, boundary_map_values, map_size, face_centers, nhalos, phi_grad.TE);
	// 	kernel_get_phi_gradient<<<block_count,thread_count>>>(phi.ED, false, local_mesh_size, local_cells_disp, faces_per_cell, faces, cell_faces, cell_centers, mesh_size, boundary_map, boundary_map_values, map_size, face_centers, nhalos, phi_grad.ED);
	// 	kernel_get_phi_gradient<<<block_count,thread_count>>>(phi.TEM, false, local_mesh_size, local_cells_disp, faces_per_cell, faces, cell_faces, cell_centers, mesh_size, boundary_map, boundary_map_values, map_size, face_centers, nhalos, phi_grad.TEM);
	// 	kernel_get_phi_gradient<<<block_count,thread_count>>>(phi.FUL, false, local_mesh_size, local_cells_disp, faces_per_cell, faces, cell_faces, cell_centers, mesh_size, boundary_map, boundary_map_values, map_size, face_centers, nhalos, phi_grad.FUL);
	// 	kernel_get_phi_gradient<<<block_count,thread_count>>>(phi.PRO, false, local_mesh_size, local_cells_disp, faces_per_cell, faces, cell_faces, cell_centers, mesh_size, boundary_map, boundary_map_values, map_size, face_centers, nhalos, phi_grad.PRO);
	// 	kernel_get_phi_gradient<<<block_count,thread_count>>>(phi.VARF, false, local_mesh_size, local_cells_disp, faces_per_cell, faces, cell_faces, cell_centers, mesh_size, boundary_map, boundary_map_values, map_size, face_centers, nhalos, phi_grad.VARF);
	// 	kernel_get_phi_gradient<<<block_count,thread_count>>>(phi.VARP, false, local_mesh_size, local_cells_disp, faces_per_cell, faces, cell_faces, cell_centers, mesh_size, boundary_map, boundary_map_values, map_size, face_centers, nhalos, phi_grad.VARP);

	// }
	// else
	// {
	// }




}

void C_kernel_calculate_flux_UVW(int block_count, int thread_count, uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t mesh_size, uint64_t local_mesh_size, int64_t map_size, uint64_t *boundary_map, uint64_t *boundary_map_values, phi_vector<vec<double>> phi_grad, vec<double> *cell_centers, vec<double> *face_centers, phi_vector<double> phi, phi_vector<double> A_phi, double *face_mass_fluxes, double *face_lambdas, vec<double> *face_normals, gpu_Face<double> *face_fields, phi_vector<double> S_phi, uint64_t nhalos, uint64_t *boundary_types, vec<double> dummy_gas_vel, double effective_viscosity, double *face_rlencos, double inlet_effective_viscosity, double *face_areas)
{
	kernel_calculate_flux_UVW<<<block_count,thread_count>>>(faces_size, faces, local_cells_disp, mesh_size, local_mesh_size, map_size, boundary_map, boundary_map_values, phi_grad, cell_centers, face_centers, phi, A_phi, face_mass_fluxes, face_lambdas, face_normals, face_fields, S_phi, nhalos, boundary_types, dummy_gas_vel, effective_viscosity, face_rlencos, inlet_effective_viscosity, face_areas);
}

void C_kernel_apply_forces(int block_count, int thread_count, uint64_t local_mesh_size, double *cell_densities, double *cell_volumes, phi_vector<double> phi, phi_vector<double> S_phi, phi_vector<vec<double>> phi_grad, double delta, phi_vector<double> A_phi, particle_aos<double> *particle_terms)
{
	kernel_apply_forces<<<block_count,thread_count>>>(local_mesh_size, cell_densities, cell_volumes, phi, S_phi, phi_grad, delta, A_phi, particle_terms); 
}

void C_kernel_setup_sparse_matrix(int block_count, int thread_count, double URFactor, uint64_t local_mesh_size, int *rows_ptr, int64_t *col_indices, uint64_t local_cells_disp, gpu_Face<uint64_t> *faces, int64_t map_size, uint64_t *boundary_map, uint64_t *boundary_map_values, double *A_phi_component, gpu_Face<double> *face_fields, double *values, double *S_phi_component, double *phi_component, uint64_t mesh_size, uint64_t faces_per_cell, uint64_t *cell_faces, int *nnz)
{
	kernel_setup_sparse_matrix<<<block_count,thread_count>>>(URFactor, local_mesh_size, rows_ptr, col_indices, local_cells_disp, faces, map_size, boundary_map, boundary_map_values, A_phi_component, face_fields, values, S_phi_component, phi_component, mesh_size, faces_per_cell, cell_faces, nnz);


	kernel_under_relax<<<block_count,thread_count>>>(A_phi_component, local_mesh_size, URFactor, rows_ptr, values, S_phi_component, phi_component);
}

void C_kernel_update_sparse_matrix(int block_count, int thread_count, double URFactor, uint64_t local_mesh_size, double *A_phi_component, double *values, int *rows_ptr, double *S_phi_component, double *phi_component, uint64_t faces_size, gpu_Face<uint64_t> *faces, int64_t map_size, uint64_t *boundary_map, uint64_t *boundary_map_values, uint64_t local_cells_disp, gpu_Face<double> *face_fields, uint64_t mesh_size)
{
	kernel_update_sparse_matrix<<<block_count,thread_count>>>(URFactor, local_mesh_size, A_phi_component, values, rows_ptr, S_phi_component, phi_component, faces_size, faces, map_size, boundary_map, boundary_map_values, local_cells_disp, face_fields, mesh_size);


	kernel_cell_update<<<block_count,thread_count>>>(URFactor, local_mesh_size, A_phi_component, values, rows_ptr, S_phi_component, phi_component);
}

void C_kernel_setup_pressure_matrix(int block_count, int thread_count, uint64_t local_mesh_size, int *rows_ptr, int64_t *col_indices, uint64_t local_cells_disp, gpu_Face<uint64_t> *faces, int64_t map_size, uint64_t *boundary_map, uint64_t *boundary_map_values, gpu_Face<double> *face_fields, double *values, uint64_t mesh_size, uint64_t faces_per_cell, uint64_t *cell_faces, int *nnz, double *face_mass_fluxes, phi_vector<double> A_phi, phi_vector<double> S_phi)
{
	kernel_setup_pressure_matrix<<<block_count,thread_count>>>(local_mesh_size, rows_ptr, col_indices, local_cells_disp, faces, map_size, boundary_map, boundary_map_values, face_fields, values, mesh_size, faces_per_cell, cell_faces, nnz, face_mass_fluxes, A_phi, S_phi);

	apply_diag<<<block_count,thread_count>>>(local_mesh_size, rows_ptr, values, A_phi);
}

void C_kernel_find_pressure_correction_max(int block_count, int thread_count, double *Pressure_correction_max, double *phi_component, uint64_t local_mesh_size)
{
	thrust::device_ptr<double> dev_ptr = thrust::device_pointer_cast(phi_component);
	*Pressure_correction_max = thrust::transform_reduce(dev_ptr, dev_ptr+local_mesh_size,
                                          []__device__(double x) -> double {
                                          	  return abs(x);
                                          },
                                          double{0.0},
										  [] __host__ __device__ (const double first, 
														const double second)
    													{
        												if(first >= second)
														{
															return first;
														}	
														else
														{
															return second;
														}
														}        
										  );
	//kernel_find_pressure_correction_max<<<block_count,thread_count>>>(Pressure_correction_max, phi_component, local_mesh_size);
}

void C_kernel_pack_phi_halo_buffer(int block_count, int thread_count, phi_vector<double> send_buffer, phi_vector<double> phi, uint64_t *indexes, uint64_t buf_size)
{
	// printf("Kernel launch <<%d, %d>>\n", block_count, thread_count);
	kernel_pack_phi_halo_buffer<<<block_count,thread_count>>>(send_buffer, phi, indexes, buf_size);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk(cudaDeviceSynchronize());
}

void C_kernel_pack_phi_grad_halo_buffer(int block_count, int thread_count, phi_vector<vec<double>> send_buffer, phi_vector<vec<double>> phi_grad, uint64_t *indexes, uint64_t buf_size)
{
	// printf("Kernel grad launch <<%d, %d>>\n", block_count, thread_count);
	kernel_pack_phi_grad_halo_buffer<<<block_count, thread_count>>>(send_buffer, phi_grad, indexes, buf_size);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk(cudaDeviceSynchronize());
}

void C_kernel_pack_PP_halo_buffer(int block_count, int thread_count, phi_vector<double> send_buffer, phi_vector<double> phi, uint64_t *indexes, uint64_t buf_size)
{
	// printf("Kernel launch <<%d, %d>>\n", block_count, thread_count);
	kernel_pack_PP_halo_buffer<<<block_count,thread_count>>>(send_buffer, phi, indexes, buf_size);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk(cudaDeviceSynchronize());
}

void C_kernel_pack_Aphi_halo_buffer(int block_count, int thread_count, phi_vector<double> send_buffer, phi_vector<double> phi, uint64_t *indexes, uint64_t buf_size)
{
	// printf("Kernel launch <<%d, %d>>\n", block_count, thread_count);
	kernel_pack_Aphi_halo_buffer<<<block_count,thread_count>>>(send_buffer, phi, indexes, buf_size);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk(cudaDeviceSynchronize());
}

void C_kernel_pack_PP_grad_halo_buffer(int block_count, int thread_count, phi_vector<vec<double>> send_buffer, phi_vector<vec<double>> phi_grad, uint64_t *indexes, uint64_t buf_size)
{
	// printf("Kernel grad launch <<%d, %d>>\n", block_count, thread_count);
	kernel_pack_PP_grad_halo_buffer<<<block_count, thread_count>>>(send_buffer, phi_grad, indexes, buf_size);
	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk(cudaDeviceSynchronize());
}

void C_kernel_Update_P_at_boundaries(int block_count, int thread_count, uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t mesh_size, uint64_t local_mesh_size, uint64_t nhalos, double *phi_component)
{
	kernel_Update_P_at_boundaries<<<block_count,thread_count>>>(faces_size, faces, local_cells_disp, mesh_size, local_mesh_size, nhalos, phi_component);
}

void C_kernel_get_phi_gradient(int block_count, int thread_count, double *phi_component, uint64_t local_mesh_size, uint64_t local_cells_disp, uint64_t faces_per_cell, gpu_Face<uint64_t> *faces, uint64_t *cell_faces, vec<double> *cell_centers, uint64_t mesh_size, uint64_t *boundary_map, uint64_t *boundary_map_values, int64_t map_size, vec<double> *face_centers, uint64_t nhalos, vec<double> *grad_component)
{
	kernel_get_phi_gradient<<<block_count,thread_count>>>(phi_component, true, local_mesh_size, local_cells_disp, faces_per_cell, faces, cell_faces, cell_centers, mesh_size, boundary_map, boundary_map_values, map_size, face_centers, nhalos, grad_component);
}

void C_kernel_update_vel_and_flux(int block_count, int thread_count, uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t local_mesh_size, uint64_t nhalos, gpu_Face<double> *face_fields, uint64_t mesh_size, int64_t map_size, uint64_t *boundary_map, uint64_t *boundary_map_values, double *face_mass_fluxes, phi_vector<double> A_phi, phi_vector<double> phi, double *cell_volumes, phi_vector<vec<double>> phi_grad, int timestep_count) 
{
	kernel_update_vel_and_flux<<<block_count,thread_count>>>(faces_size, faces, local_cells_disp, local_mesh_size, nhalos, face_fields, mesh_size, map_size, boundary_map, boundary_map_values, face_mass_fluxes, A_phi, phi, cell_volumes, phi_grad, timestep_count);
}

void C_kernel_Update_P(int block_count, int thread_count, uint64_t faces_size, uint64_t local_mesh_size, uint64_t nhalos, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t mesh_size, vec<double> *cell_centers, vec<double> *face_centers, uint64_t *boundary_types, double *phi_component, vec<double> *phi_grad_component)
{
	kernel_Update_P<<<block_count,thread_count>>>(faces_size, local_mesh_size, nhalos, faces, local_cells_disp, mesh_size, cell_centers, face_centers, boundary_types, phi_component, phi_grad_component);
}

void C_kernel_calculate_mass_flux(int block_count, int thread_count, uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t mesh_size, uint64_t local_mesh_size, int64_t map_size, uint64_t *boundary_map, uint64_t *boundary_map_values, phi_vector<vec<double>> phi_grad, vec<double> *cell_centers, vec<double> *face_centers, phi_vector<double> phi, double *cell_densities, phi_vector<double> A_phi, double *cell_volumes, double *face_mass_fluxes, double *face_lambdas, vec<double> *face_normals, double *face_areas, gpu_Face<double> *face_fields, phi_vector<double> S_phi, uint64_t nhalos, uint64_t *boundary_types, vec<double> dummy_gas_vel)
{
	kernel_calculate_mass_flux<<<block_count,thread_count>>>(faces_size, faces, local_cells_disp, mesh_size, local_mesh_size, map_size, boundary_map, boundary_map_values, phi_grad, cell_centers, face_centers, phi, cell_densities, A_phi, cell_volumes, face_mass_fluxes, face_lambdas, face_normals, face_areas, face_fields, S_phi, nhalos, boundary_types, dummy_gas_vel);
}

void C_kernel_compute_flow_correction(int block_count, int thread_count, uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t mesh_size, uint64_t *boundary_types, double *FlowOut, double *FlowIn, double *areaout, int *count_out, double *face_mass_fluxes, double *face_areas) 
{
	kernel_compute_flow_correction<<<block_count,thread_count>>>(faces_size, faces, mesh_size, boundary_types, FlowOut, FlowIn, areaout, count_out, face_mass_fluxes, face_areas);
}

void C_kernel_correct_flow(int block_count, int thread_count, int *count_out, double *FlowOut, double *FlowIn, double *areaout, uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t mesh_size, uint64_t *boundary_types, double *face_mass_fluxes, double *face_areas, double *cell_densities, phi_vector<double> phi, uint64_t local_mesh_size, uint64_t nhalos, vec<double> *face_normals, uint64_t local_cells_disp, phi_vector<double> S_phi, double *FlowFact)
{
	kernel_correct_flow<<<block_count,thread_count>>>(count_out, FlowOut, FlowIn, areaout, faces_size, faces, mesh_size, boundary_types, face_mass_fluxes, face_areas, cell_densities, phi, local_mesh_size, nhalos, face_normals, local_cells_disp, S_phi, FlowFact);
}

void C_kernel_correct_flow2(int block_count, int thread_count, int *count_out, double *FlowOut, double *FlowIn, double *areaout, uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t mesh_size, uint64_t *boundary_types, double *face_mass_fluxes, double *face_areas, double *cell_densities, phi_vector<double> phi, uint64_t local_mesh_size, uint64_t nhalos, vec<double> *face_normals, uint64_t local_cells_disp, phi_vector<double> S_phi, double *FlowFact)
{
	kernel_correct_flow2<<<block_count,thread_count>>>(count_out, FlowOut, FlowIn, areaout, faces_size, faces, mesh_size, boundary_types, face_mass_fluxes, face_areas, cell_densities, phi, local_mesh_size, nhalos, face_normals, local_cells_disp, S_phi, FlowFact);
}

void C_kernel_flux_scalar(int block_count, int thread_count, int type, uint64_t faces_size, uint64_t local_mesh_size, uint64_t nhalos, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t mesh_size, vec<double> *cell_centers, vec<double> *face_centers, uint64_t *boundary_types, uint64_t *boundary_map, uint64_t *boundary_map_values, int64_t map_size, double *phi_component, phi_vector<double> A_phi, phi_vector<double> S_phi, vec<double> *phi_grad_component, double *face_lambdas, double effective_viscosity, double *face_rlencos, double *face_mass_fluxes, vec<double> *face_normals, double inlet_effective_viscosity, gpu_Face<double> *face_fields, vec<double> dummy_gas_vel, double dummy_gas_tem, double dummy_gas_fuel)
{
	kernel_flux_scalar<<<block_count,thread_count>>>(type, faces_size, local_mesh_size, nhalos, faces, local_cells_disp, mesh_size, cell_centers, face_centers, boundary_types, boundary_map, boundary_map_values, map_size, phi_component, A_phi, S_phi, phi_grad_component, face_lambdas, effective_viscosity, face_rlencos, face_mass_fluxes, face_normals, inlet_effective_viscosity, face_fields, dummy_gas_vel, dummy_gas_tem, dummy_gas_fuel);
}

void C_kernel_apply_pres_forces(int block_count, int thread_count, int type, uint64_t local_mesh_size, double delta, double *cell_densities, double *cell_volumes, double *phi_component, phi_vector<double> A_phi, phi_vector<double> S_phi, particle_aos<double> *particle_terms)
{
	kernel_apply_pres_forces<<<block_count,thread_count>>>(type, local_mesh_size, delta, cell_densities, cell_volumes, phi_component, A_phi, S_phi, particle_terms);
}

void C_kernel_solve_turb_models_cell(int block_count, int thread_count, int type, uint64_t local_mesh_size, phi_vector<double> A_phi, phi_vector<double> S_phi, phi_vector<vec<double>> phi_grad, phi_vector<double> phi, double effective_viscosity, double *cell_densities, double *cell_volumes) 
{
	kernel_solve_turb_models_cell<<<block_count,thread_count>>>(type, local_mesh_size, A_phi, S_phi, phi_grad, phi, effective_viscosity, cell_densities, cell_volumes);
}

void C_kernel_solve_turb_models_face(int block_count, int thread_count, int type, uint64_t faces_size, uint64_t mesh_size, double *cell_volumes, double effective_viscosity, vec<double> *face_centers, vec<double> *cell_centers, vec<double> *face_normals, double *cell_densities, uint64_t local_mesh_size, uint64_t nhalos, uint64_t local_cells_disp, phi_vector<double> A_phi, phi_vector<double> S_phi, phi_vector<double> phi, gpu_Face<double> *face_fields, gpu_Face<uint64_t> *faces, uint64_t *boundary_types, uint64_t faces_per_cell, uint64_t *cell_faces) 
{
	kernel_solve_turb_models_face<<<block_count,thread_count>>>(type, faces_size, mesh_size, cell_volumes, effective_viscosity, face_centers, cell_centers, face_normals, cell_densities, local_mesh_size, nhalos, local_cells_disp, A_phi, S_phi, phi, face_fields, faces, boundary_types, faces_per_cell, cell_faces);
}

void C_kernel_set_up_fgm_table(int block_count, int thread_count, double *fgm_table, uint64_t seed)
{
	kernel_set_up_fgm_table<<<block_count,thread_count>>>(fgm_table, time(NULL));
}

void C_kernel_fgm_look_up(int block_count, int thread_count, double *fgm_table, phi_vector<double> S_phi, phi_vector<double> phi, uint64_t local_mesh_size)
{
	kernel_fgm_look_up<<<block_count,thread_count>>>(fgm_table, S_phi, phi, local_mesh_size);
}


void C_create_map(int block_count, int thread_count, uint64_t *map,  uint64_t *gpu_keys, uint64_t *gpu_values, uint64_t size)
{
	kernel_create_map<<<block_count, thread_count>>>(map, gpu_keys, gpu_values, size);
}

void C_kernel_process_particle_fields(uint64_t block_count, int thread_count, uint64_t *sent_cell_indexes, particle_aos<double> *sent_particle_fields, particle_aos<double> *particle_fields, uint64_t num_fields, uint64_t local_mesh_disp)
{

	kernel_process_particle_fields<<<block_count, thread_count>>>(sent_cell_indexes, sent_particle_fields, particle_fields, num_fields, local_mesh_disp);
}

// void C_create_cuco_map(int block_count, int thread_count, uint64_t *gpu_keys, uint64_t *gpu_values, uint64_t size)
// {
// 	printf("Creating a map with size %lu\n", size);
// 	using Key   = int;
// 	using Value = int;

// 	// Empty slots are represented by reserved "sentinel" values. These values should be selected such
// 	// that they never occur in your input data.
// 	Key constexpr empty_key_sentinel     = -1;
// 	Value constexpr empty_value_sentinel = -1;

//   	// Number of key/value pairs to be inserted
//   	std::size_t num_keys = size;

//   	// Compute capacity based on a 50% load factor
// 	auto constexpr load_factor = 0.5;
// 	std::size_t const capacity = std::ceil(num_keys / load_factor);

//     auto boundary_map = cuco::static_map{capacity,
// 								cuco::empty_key{empty_key_sentinel},
// 								cuco::empty_value{empty_value_sentinel},
// 								thrust::equal_to<Key>{},
// 								cuco::linear_probing<1, cuco::default_hash_function<Key>>{}};

// 	// Get a non-owning, mutable reference of the map that allows inserts to pass by value into the
// 	// kernel
// 	auto insert_ref = boundary_map.ref(cuco::insert);

// 	auto constexpr block_size = 256;
// 	auto const grid_size      = (num_keys + block_size - 1) / block_size;
// 	insert<<<grid_size, block_size>>>(insert_ref, gpu_keys, gpu_values, num_keys);

// 	std::cout << "Number of keys inserted: " << num_keys << std::endl;

// 	// Get a non-owning reference of the map that allows find operations to pass by value into the
// 	// kernel
// 	auto find_ref = boundary_map.ref(cuco::find);
// }
