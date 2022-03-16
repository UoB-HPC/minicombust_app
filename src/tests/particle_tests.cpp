#include "tests/particle_tests.hpp"
#include "examples/mesh_examples.hpp"

using namespace std;

using namespace minicombust::particles;


bool check_particle_posistion(Mesh<double> *mesh, uint64_t correct_cell, vec<double> start, vec<double> velocity)
{
    if (PARTICLE_DEBUG)  cout << endl;
    
    Particle<double> *p = new Particle<double>(mesh, start, velocity, vec<double>{0, 0, 0});


    p->timestep(mesh);


    return p->cell == correct_cell;
}


void run_particle_tests()
{
    printf("\tRunning particle tests\n");
    const double box_dim                  = 100;
    const uint64_t elements_per_dim       = 10;

    vec<double> FRONT_UNIT_VEC = { 0.,  0., -1.};
    vec<double> BACK_UNIT_VEC  = { 0.,  0.,  1.};
    vec<double> LEFT_UNIT_VEC  = {-1.,  0.,  0.};
    vec<double> RIGHT_UNIT_VEC = { 1.,  0.,  0.};
    vec<double> DOWN_UNIT_VEC  = { 0., -1.,  0.};
    vec<double> UP_UNIT_VEC    = { 0.,  1.,  0.};
    
    Mesh<double> *mesh    = load_mesh(box_dim, elements_per_dim);

    vec<double> start     = {15.0, 15.0, 15.0};


    // Tests for moving to immediate neighbour
    cout << "\t\tRunning immediate neighbour tests..." << endl;
    assert(((void)"Particle is not moving to FRONT cell\n", check_particle_posistion(mesh, 11,   start, 10.*FRONT_UNIT_VEC)));
    assert(((void)"Particle is not moving to BACK  cell\n", check_particle_posistion(mesh, 211,  start, 10.*BACK_UNIT_VEC)));

    assert(((void)"Particle is not moving to LEFT  cell\n", check_particle_posistion(mesh, 110,  start, 10.*LEFT_UNIT_VEC)));
    assert(((void)"Particle is not moving to RIGHT cell\n", check_particle_posistion(mesh, 112,  start, 10.*RIGHT_UNIT_VEC)));

    assert(((void)"Particle is not moving to DOWN  cell\n", check_particle_posistion(mesh, 101,  start, 10.*DOWN_UNIT_VEC)));
    assert(((void)"Particle is not moving to UP    cell\n", check_particle_posistion(mesh, 121,  start, 10.*UP_UNIT_VEC)));
    cout << "\t\tPassed immediate neighbour tests." << endl << endl;;

    // // Tests for moving diagonally through edges
    cout << "\t\tRunning edge neighbour tests..." << endl;
    start     = {15.0, 15.0, 5.0};
    assert(((void)"Particle is not moving to FRONT+LEFT cell\n",  check_particle_posistion(mesh, MESH_BOUNDARY,   start, 10.*FRONT_UNIT_VEC + 10.*LEFT_UNIT_VEC)));
    assert(((void)"Particle is not moving to FRONT+RIGHT cell\n", check_particle_posistion(mesh, MESH_BOUNDARY,   start, 10.*FRONT_UNIT_VEC + 10.*RIGHT_UNIT_VEC)));
    assert(((void)"Particle is not moving to FRONT+DOWN cell\n",  check_particle_posistion(mesh, MESH_BOUNDARY,   start, 10.*FRONT_UNIT_VEC + 10.*DOWN_UNIT_VEC)));
    assert(((void)"Particle is not moving to FRONT+UP cell\n",    check_particle_posistion(mesh, MESH_BOUNDARY,   start, 10.*FRONT_UNIT_VEC + 10.*UP_UNIT_VEC)));

    assert(((void)"Particle is not moving to BACK+LEFT cell\n",   check_particle_posistion(mesh,           110,   start, 10.*BACK_UNIT_VEC + 10.*LEFT_UNIT_VEC)));
    assert(((void)"Particle is not moving to BACK+RIGHT cell\n",  check_particle_posistion(mesh,           112,   start, 10.*BACK_UNIT_VEC + 10.*RIGHT_UNIT_VEC)));
    assert(((void)"Particle is not moving to BACK+DOWN cell\n",   check_particle_posistion(mesh,           101,   start, 10.*BACK_UNIT_VEC + 10.*DOWN_UNIT_VEC)));
    assert(((void)"Particle is not moving to BACK+UP cell\n",     check_particle_posistion(mesh,           121,   start, 10.*BACK_UNIT_VEC + 10.*UP_UNIT_VEC)));

    assert(((void)"Particle is not moving to DOWN+LEFT cell\n",   check_particle_posistion(mesh,             0,   start, 10.*DOWN_UNIT_VEC + 10.*LEFT_UNIT_VEC)));
    assert(((void)"Particle is not moving to UP+LEFT cell\n",     check_particle_posistion(mesh,            20,   start, 10.*UP_UNIT_VEC   + 10.*LEFT_UNIT_VEC)));
    assert(((void)"Particle is not moving to DOWN+RIGHT cell\n",  check_particle_posistion(mesh,             2,   start, 10.*DOWN_UNIT_VEC + 10.*RIGHT_UNIT_VEC)));
    assert(((void)"Particle is not moving to UP+RIGHT cell\n",    check_particle_posistion(mesh,            22,   start, 10.*UP_UNIT_VEC   + 10.*RIGHT_UNIT_VEC)));

    cout << "\t\tPassed edge neighbour tests." << endl << endl;

    // Tests for moving diagonally through vertexes
    cout << "\t\tRunning vertex neighbour tests..." << endl;
    start     = {15.0, 15.0, 15.0};
    assert(((void)"Particle is not moving to FRONT+LEFT+DOWN cell\n",   check_particle_posistion(mesh,         0,   start, 10.*FRONT_UNIT_VEC + 10.*LEFT_UNIT_VEC + 10.*DOWN_UNIT_VEC)));
    assert(((void)"Particle is not moving to FRONT+LEFT+UP cell\n",     check_particle_posistion(mesh,        20,   start, 10.*FRONT_UNIT_VEC + 10.*LEFT_UNIT_VEC + 10.*UP_UNIT_VEC)));
    assert(((void)"Particle is not moving to FRONT+RIGHT+DOWN cell\n",  check_particle_posistion(mesh,         2,   start, 10.*FRONT_UNIT_VEC + 10.*RIGHT_UNIT_VEC + 10.*DOWN_UNIT_VEC)));
    assert(((void)"Particle is not moving to FRONT+RIGHT+UP cell\n",    check_particle_posistion(mesh,        22,   start, 10.*FRONT_UNIT_VEC + 10.*RIGHT_UNIT_VEC + 10.*UP_UNIT_VEC)));

    assert(((void)"Particle is not moving to BACK+LEFT+DOWN cell\n",    check_particle_posistion(mesh,       200,   start, 10.*BACK_UNIT_VEC + 10.*LEFT_UNIT_VEC + 10.*DOWN_UNIT_VEC)));
    assert(((void)"Particle is not moving to BACK+LEFT+UP cell\n",      check_particle_posistion(mesh,       220,   start, 10.*BACK_UNIT_VEC + 10.*LEFT_UNIT_VEC + 10.*UP_UNIT_VEC)));
    assert(((void)"Particle is not moving to BACK+RIGHT+DOWN cell\n",   check_particle_posistion(mesh,       202,   start, 10.*BACK_UNIT_VEC + 10.*RIGHT_UNIT_VEC + 10.*DOWN_UNIT_VEC)));
    assert(((void)"Particle is not moving to BACK+RIGHT+UP cell\n",     check_particle_posistion(mesh,       222,   start, 10.*BACK_UNIT_VEC + 10.*RIGHT_UNIT_VEC + 10.*UP_UNIT_VEC)));
    cout << "\t\tPassed vertex neighbour tests." << endl << endl;

    // Tests for moving diagonally through vertexes
    cout << "\t\tRunning multiple cell tests..." << endl;
    start     = {15.0, 15.0, 15.0};
    assert(((void)"Particle is not moving to 999 cell\n",             check_particle_posistion(mesh,            999,   start, 80.*BACK_UNIT_VEC + 80.*RIGHT_UNIT_VEC + 80.*UP_UNIT_VEC)));
    assert(((void)"Particle is not moving to MESH_BOUNDARY cell\n",   check_particle_posistion(mesh,  MESH_BOUNDARY,   start, 90.*BACK_UNIT_VEC + 90.*RIGHT_UNIT_VEC + 90.*UP_UNIT_VEC)));
    start     = {95.0, 95.0, 95.0};
    assert(((void)"Particle is not moving to 999 cell\n",             check_particle_posistion(mesh,              0,   start, 90.*FRONT_UNIT_VEC + 90.*LEFT_UNIT_VEC + 90.*DOWN_UNIT_VEC)));
    cout << "\t\tPassed multiple cell tests." << endl << endl;

}   