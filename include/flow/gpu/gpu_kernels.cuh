#include "flow/gpu/gpu_type_def.h"

#include "flow/gpu/gpu_hash_map.inl"


using namespace minicombust::utils;

void C_kernel_print(double *to_print, uint64_t num_print);

void C_kernel_vec_print(vec<double> *to_print, uint64_t num_print);

void C_test_solve();

void C_kernel_interpolate_init_boundaries(int block_count, int thread_count, phi_vector<double> phi_nodes, uint8_t *cells_per_point, uint64_t local_points_size);

void C_kernel_interpolate_phi_to_nodes(int block_count, int thread_count, phi_vector<double> phi, phi_vector<vec<double>> phi_grad, phi_vector<double> phi_nodes, vec<double> *points, Hash_map *node_map, Hash_map *boundary_map, uint8_t *cells_per_point, uint64_t *cells, vec<double> *cell_centers, uint64_t local_mesh_size, uint64_t cell_disp, uint64_t local_points_size, uint64_t nhalos);

void C_kernel_process_particle_fields(uint64_t block_count, int thread_count, uint64_t *sent_cell_indexes, particle_aos<double> *sent_particle_fields, particle_aos<double> *particle_fields, uint64_t num_fields, uint64_t local_mesh_disp);

void C_kernel_test_particle_terms(particle_aos<double> *particle_terms, uint64_t local_mesh_size);

void C_kernel_test_values(int *nnz, double *values, int *rows_ptr, int64_t *col_indices, uint64_t local_mesh_size, double *b, double *u);

void C_kernel_test_values(int *nnz, double *values, int *rows_ptr, int64_t *col_indices, uint64_t local_mesh_size);

void C_kernel_update_mass_flux(int block_count, int thread_count, uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t local_mesh_size, uint64_t mesh_size, Hash_map *boundary_map, uint64_t *boundary_map_values, int64_t map_size, vec<double> *face_centers, vec<double> *cell_centers, gpu_Face<double> *face_fields, phi_vector<vec<double>> phi_grad, double *face_mass_fluxes, phi_vector<double> S_phi, vec<double> *face_normals);

void C_kernel_correct_flow2(int block_count, int thread_count, int *count_out, double *FlowOut, double *FlowIn, double *areaout, uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t mesh_size, uint64_t *boundary_types, double *face_mass_fluxes, double *face_areas, double *cell_densities, phi_vector<double> phi, uint64_t local_mesh_size, uint64_t nhalos, vec<double> *face_normals, uint64_t local_cells_disp, phi_vector<double> S_phi, double *FlowFact);

void C_kernel_get_phi_gradients(int block_count, int thread_count, phi_vector<double> phi, phi_vector<vec<double>> phi_grad, uint64_t local_mesh_size, uint64_t local_cells_disp, uint64_t faces_per_cell, gpu_Face<uint64_t> *faces, uint64_t *cell_faces, vec<double> *cell_centers, uint64_t mesh_size, Hash_map *boundary_map, uint64_t *boundary_map_values, int64_t map_size, vec<double> *face_centers, uint64_t nhalos);

void C_kernel_calculate_flux_UVW(int block_count, int thread_count, uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t mesh_size, uint64_t local_mesh_size, int64_t map_size, Hash_map *boundary_map, uint64_t *boundary_map_values, phi_vector<vec<double>> phi_grad, vec<double> *cell_centers, vec<double> *face_centers, phi_vector<double> phi, phi_vector<double> A_phi, double *face_mass_fluxes, double *face_lambdas, vec<double> *face_normals, gpu_Face<double> *face_fields, phi_vector<double> S_phi, uint64_t nhalos, uint64_t *boundary_types, vec<double> dummy_gas_vel, double effective_viscosity, double *face_rlencos, double inlet_effective_viscosity, double *face_areas);

void C_kernel_apply_forces(int block_count, int thread_count, uint64_t local_mesh_size, double *cell_densities, double *cell_volumes, phi_vector<double> phi, phi_vector<double> S_phi, phi_vector<vec<double>> phi_grad, double delta, phi_vector<double> A_phi, particle_aos<double> *particle_terms);

void C_kernel_setup_sparse_matrix(int block_count, int thread_count, double URFactor, uint64_t local_mesh_size, int *rows_ptr, int64_t *col_indices, uint64_t local_cells_disp, gpu_Face<uint64_t> *faces, int64_t map_size, Hash_map *boundary_map, uint64_t *boundary_map_values, double *A_phi_component, gpu_Face<double> *face_fields, double *values, double *S_phi_component, double *phi_component, uint64_t mesh_size, uint64_t faces_per_cell, uint64_t *cell_faces, int *nnz);

void C_kernel_precomp_AU(int block_count, int thread_count, uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t mesh_size, uint64_t *boundary_types, double effective_viscosity, double * face_rlencos, double *face_mass_fluxes, phi_vector<double> A_phi, uint64_t local_mesh_size, double delta, double *cell_densities, double* cell_volumes);

void C_kernel_update_sparse_matrix(int block_count, int thread_count, double URFactor, uint64_t local_mesh_size, double *A_phi_component, double *values, int *rows_ptr, double *S_phi_component, double *phi_component, uint64_t faces_size, gpu_Face<uint64_t> *faces, int64_t map_size, Hash_map *boundary_map, uint64_t *boundary_map_values, uint64_t local_cells_disp, gpu_Face<double> *face_fields, uint64_t mesh_size);

void C_kernel_setup_pressure_matrix(int block_count, int thread_count, uint64_t local_mesh_size, int *rows_ptr, int64_t *col_indices, uint64_t local_cells_disp, gpu_Face<uint64_t> *faces, int64_t map_size, Hash_map *boundary_map, uint64_t *boundary_map_values, gpu_Face<double> *face_fields, double *values, uint64_t mesh_size, uint64_t faces_per_cell, uint64_t *cell_faces, int *nnz, double *face_mass_fluxes, phi_vector<double> A_phi, phi_vector<double> S_phi);

void C_kernel_find_pressure_correction_max(int block_count, int thread_count, double *Pressure_correction_max, double *phi_component, uint64_t local_mesh_size);

void C_kernel_Update_P_at_boundaries(int block_count, int thread_count, uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t mesh_size, uint64_t local_mesh_size, uint64_t nhalos, double *phi_component);

void C_kernel_get_phi_gradient(int block_count, int thread_count, double *phi_component, uint64_t local_mesh_size, uint64_t local_cells_disp, uint64_t faces_per_cell, gpu_Face<uint64_t> *faces, uint64_t *cell_faces, vec<double> *cell_centers, uint64_t mesh_size, Hash_map *boundary_map, uint64_t *boundary_map_values, int64_t map_size, vec<double> *face_centers, uint64_t nhalos, vec<double> *grad_component);

void C_kernel_update_vel_and_flux(int block_count, int thread_count, uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t local_mesh_size, uint64_t nhalos, gpu_Face<double> *face_fields, uint64_t mesh_size, int64_t map_size, Hash_map *boundary_map, uint64_t *boundary_map_values, double *face_mass_fluxes, phi_vector<double> A_phi, phi_vector<double> phi, double *cell_volumes, phi_vector<vec<double>> phi_grad, int timestep_count);

void C_kernel_Update_P(int block_count, int thread_count, uint64_t faces_size, uint64_t local_mesh_size, uint64_t nhalos, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t mesh_size, vec<double> *cell_centers, vec<double> *face_centers, uint64_t *boundary_types, double *phi_component, vec<double> *phi_grad_component);

void C_kernel_calculate_mass_flux(int block_count, int thread_count, uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t mesh_size, uint64_t local_mesh_size, int64_t map_size, Hash_map *boundary_map, uint64_t *boundary_map_values, phi_vector<vec<double>> phi_grad, vec<double> *cell_centers, vec<double> *face_centers, phi_vector<double> phi, double *cell_densities, phi_vector<double> A_phi, double *cell_volumes, double *face_mass_fluxes, double *face_lambdas, vec<double> *face_normals, double *face_areas, gpu_Face<double> *face_fields, phi_vector<double> S_phi, uint64_t nhalos, uint64_t *boundary_types, vec<double> dummy_gas_vel);

void C_kernel_compute_flow_correction(int block_count, int thread_count, uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t mesh_size, uint64_t *boundary_types, double *FlowOut, double *FlowIn, double *areaout, int *count_out, double *face_mass_fluxes, double *face_areas);

void C_kernel_correct_flow(int block_count, int thread_count, int *count_out, double *FlowOut, double *FlowIn, double *areaout, uint64_t faces_size, gpu_Face<uint64_t> *faces, uint64_t mesh_size, uint64_t *boundary_types, double *face_mass_fluxes, double *face_areas, double *cell_densities, phi_vector<double> phi, uint64_t local_mesh_size, uint64_t nhalos, vec<double> *face_normals, uint64_t local_cells_disp, phi_vector<double> S_phi, double *FlowFact);

void C_kernel_flux_scalar(int block_count, int thread_count, int type, uint64_t faces_size, uint64_t local_mesh_size, uint64_t nhalos, gpu_Face<uint64_t> *faces, uint64_t local_cells_disp, uint64_t mesh_size, vec<double> *cell_centers, vec<double> *face_centers, uint64_t *boundary_types, Hash_map *boundary_map, uint64_t *boundary_map_values, int64_t map_size, double *phi_component, phi_vector<double> A_phi, phi_vector<double> S_phi, vec<double> *phi_grad_component, double *face_lambdas, double effective_viscosity, double *face_rlencos, double *face_mass_fluxes, vec<double> *face_normals, double inlet_effective_viscosity, gpu_Face<double> *face_fields, vec<double> dummy_gas_vel, double dummy_gas_tem, double dummy_gas_fuel);

void C_kernel_apply_pres_forces(int block_count, int thread_count, int type, uint64_t local_mesh_size, double delta, double *cell_densities, double *cell_volumes, double *phi_component, phi_vector<double> A_phi, phi_vector<double> S_phi, particle_aos<double> *particle_terms);

void C_kernel_solve_turb_models_cell(int block_count, int thread_count, int type, uint64_t local_mesh_size, phi_vector<double> A_phi, phi_vector<double> S_phi, phi_vector<vec<double>> phi_grad, phi_vector<double> phi, double effective_viscosity, double *cell_densities, double *cell_volumes);

void C_kernel_solve_turb_models_face(int block_count, int thread_count, int type, uint64_t faces_size, uint64_t mesh_size, double *cell_volumes, double effective_viscosity, vec<double> *face_centers, vec<double> *cell_centers, vec<double> *face_normals, double *cell_densities, uint64_t local_mesh_size, uint64_t nhalos, uint64_t local_cells_disp, phi_vector<double> A_phi, phi_vector<double> S_phi, phi_vector<double> phi, gpu_Face<double> *face_fields, gpu_Face<uint64_t> *faces, uint64_t *boundary_types, uint64_t faces_per_cell, uint64_t *cell_faces);

void C_kernel_set_up_fgm_table(int block_count, int thread_count, double *fgm_table, uint64_t seed);

void C_kernel_fgm_look_up(int block_count, int thread_count, double *fgm_table, phi_vector<double> S_phi, phi_vector<double> phi, uint64_t local_mesh_size);

void C_kernel_pack_phi_halo_buffer(int block_count, int thread_count, phi_vector<double> send_buffer, phi_vector<double> phi, uint64_t *indexes, uint64_t buf_size);

void C_kernel_pack_phi_grad_halo_buffer(int block_count, int thread_count, phi_vector<vec<double>> send_buffer, phi_vector<vec<double>> phi_grad, uint64_t *indexes, uint64_t buf_size);

void C_kernel_pack_PP_halo_buffer(int block_count, int thread_count, phi_vector<double> send_buffer, phi_vector<double> phi, uint64_t *indexes, uint64_t buf_size);
void C_kernel_pack_Aphi_halo_buffer(int block_count, int thread_count, phi_vector<double> send_buffer, phi_vector<double> phi, uint64_t *indexes, uint64_t buf_size);
void C_kernel_pack_flow_field_buffer(int block_count, int thread_count, uint64_t *index_buffer, phi_vector<double> phi_nodes, flow_aos<double> *flow_buffer, Hash_map *node_map, uint64_t buf_size, uint64_t node_map_size, uint64_t rank_slot);
void C_kernel_pack_PP_grad_halo_buffer(int block_count, int thread_count, phi_vector<vec<double>> send_buffer, phi_vector<vec<double>> phi_grad, uint64_t *indexes, uint64_t buf_size);

void C_create_cuco_map(int block_count, int thread_count, uint64_t *gpu_keys, uint64_t *gpu_values, uint64_t size);
void C_create_map(int block_count, int thread_count, Hash_map *node_map,  uint64_t *gpu_keys, uint64_t *gpu_values, uint64_t size);

