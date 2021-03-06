#include "tests/particle_tests.hpp"
#include "examples/mesh_examples.hpp"

#define CATCH_CONFIG_MAIN
#include "tests/catch.hpp"

using namespace std;

using namespace minicombust::particles;


bool check_particle_position(Mesh<double> *mesh, uint64_t correct_cell, vec<double> start, vec<double> velocity)
{
    particle_logger logger;
    Particle<double> *p = new Particle<double>(mesh, start, velocity, vec<double>{0, 0, 0}, 300., 0, &logger);

    memset(&logger, 0, sizeof(particle_logger));
    p->x1 += p->v1 * 1.0;
    p->update_cell(mesh, &logger);

    return p->cell == correct_cell;
}

bool check_particle_position(Mesh<double> *mesh, uint64_t correct_cell, vec<double> x1, uint64_t starting_cell)
{
    particle_logger logger;
    Particle<double> *p = new Particle<double>(mesh, x1, vec<double>{0, 0, 0}, vec<double>{0, 0, 0}, 300., starting_cell, &logger);

    memset(&logger, 0, sizeof(particle_logger));
    p->update_cell(mesh, &logger);

    return p->cell == correct_cell;
}

const double box_dim                  = 100;
const uint64_t elements_per_dim       = 10;

Mesh<double> *mesh    = load_mesh(box_dim, elements_per_dim, 0);

TEST_CASE( "Particles can move from cell to cell correctly. (Cube Mesh)", "[particle]" ) {

    const vec<double> FRONT_UNIT_VEC = { 0.,  0., -1.};
    const vec<double> BACK_UNIT_VEC  = { 0.,  0.,  1.};
    const vec<double> LEFT_UNIT_VEC  = {-1.,  0.,  0.};
    const vec<double> RIGHT_UNIT_VEC = { 1.,  0.,  0.};
    const vec<double> DOWN_UNIT_VEC  = { 0., -1.,  0.};
    const vec<double> UP_UNIT_VEC    = { 0.,  1.,  0.};


    SECTION( "Particle can move to immediate neighbours." ) {
        const vec<double> start     = {15.0, 15.0, 15.0};

        REQUIRE( check_particle_position(mesh, 11,   start, 10.*FRONT_UNIT_VEC) );
        REQUIRE( check_particle_position(mesh, 211,  start, 10.*BACK_UNIT_VEC)  );

        REQUIRE( check_particle_position(mesh, 110,  start, 10.*LEFT_UNIT_VEC)  );
        REQUIRE( check_particle_position(mesh, 112,  start, 10.*RIGHT_UNIT_VEC) );

        REQUIRE( check_particle_position(mesh, 101,  start, 10.*DOWN_UNIT_VEC)  );
        REQUIRE( check_particle_position(mesh, 121,  start, 10.*UP_UNIT_VEC  )  );
    }

    SECTION( "Particle can move to neighbours through edges." ) {
        const vec<double> start     = {15.0, 15.0, 5.0};

        REQUIRE( check_particle_position(mesh, MESH_BOUNDARY,   start, 10.*FRONT_UNIT_VEC + 10.*LEFT_UNIT_VEC) );
        REQUIRE( check_particle_position(mesh, MESH_BOUNDARY,   start, 10.*FRONT_UNIT_VEC + 10.*RIGHT_UNIT_VEC));
        REQUIRE( check_particle_position(mesh, MESH_BOUNDARY,   start, 10.*FRONT_UNIT_VEC + 10.*DOWN_UNIT_VEC) );
        REQUIRE( check_particle_position(mesh, MESH_BOUNDARY,   start, 10.*FRONT_UNIT_VEC + 10.*UP_UNIT_VEC)   );
        
        REQUIRE( check_particle_position(mesh,           110,   start, 10.*BACK_UNIT_VEC + 10.*LEFT_UNIT_VEC)  );
        REQUIRE( check_particle_position(mesh,           112,   start, 10.*BACK_UNIT_VEC + 10.*RIGHT_UNIT_VEC) );
        REQUIRE( check_particle_position(mesh,           101,   start, 10.*BACK_UNIT_VEC + 10.*DOWN_UNIT_VEC)  );
        REQUIRE( check_particle_position(mesh,           121,   start, 10.*BACK_UNIT_VEC + 10.*UP_UNIT_VEC)    );
        
        REQUIRE( check_particle_position(mesh,             0,   start, 10.*DOWN_UNIT_VEC + 10.*LEFT_UNIT_VEC)  );
        REQUIRE( check_particle_position(mesh,            20,   start, 10.*UP_UNIT_VEC   + 10.*LEFT_UNIT_VEC)  );
        REQUIRE( check_particle_position(mesh,             2,   start, 10.*DOWN_UNIT_VEC + 10.*RIGHT_UNIT_VEC) );
        REQUIRE( check_particle_position(mesh,            22,   start, 10.*UP_UNIT_VEC   + 10.*RIGHT_UNIT_VEC) );
    }

    SECTION( "Particle can move to neighbours through vertexes." ) {
        const vec<double> start     = {15.0, 15.0, 15.0};
        REQUIRE( check_particle_position(mesh,         0,   start, 10.*FRONT_UNIT_VEC + 10.*LEFT_UNIT_VEC + 10.*DOWN_UNIT_VEC));
        REQUIRE( check_particle_position(mesh,        20,   start, 10.*FRONT_UNIT_VEC + 10.*LEFT_UNIT_VEC + 10.*UP_UNIT_VEC));
        REQUIRE( check_particle_position(mesh,         2,   start, 10.*FRONT_UNIT_VEC + 10.*RIGHT_UNIT_VEC + 10.*DOWN_UNIT_VEC));
        REQUIRE( check_particle_position(mesh,        22,   start, 10.*FRONT_UNIT_VEC + 10.*RIGHT_UNIT_VEC + 10.*UP_UNIT_VEC));

        REQUIRE( check_particle_position(mesh,       200,   start, 10.*BACK_UNIT_VEC + 10.*LEFT_UNIT_VEC + 10.*DOWN_UNIT_VEC));
        REQUIRE( check_particle_position(mesh,       220,   start, 10.*BACK_UNIT_VEC + 10.*LEFT_UNIT_VEC + 10.*UP_UNIT_VEC));
        REQUIRE( check_particle_position(mesh,       202,   start, 10.*BACK_UNIT_VEC + 10.*RIGHT_UNIT_VEC + 10.*DOWN_UNIT_VEC));
        REQUIRE( check_particle_position(mesh,       222,   start, 10.*BACK_UNIT_VEC + 10.*RIGHT_UNIT_VEC + 10.*UP_UNIT_VEC));
    }

    SECTION( "Particle can move through multiple cells." ) {
        vec<double> start     = {15.0, 15.0, 15.0};
        REQUIRE( check_particle_position(mesh,            999,   start, 80.*BACK_UNIT_VEC + 80.*RIGHT_UNIT_VEC + 80.*UP_UNIT_VEC));
        REQUIRE( check_particle_position(mesh,  MESH_BOUNDARY,   start, 90.*BACK_UNIT_VEC + 90.*RIGHT_UNIT_VEC + 90.*UP_UNIT_VEC));
        start     = {95.0, 95.0, 95.0};
        REQUIRE( check_particle_position(mesh,              0,   start, 90.*FRONT_UNIT_VEC + 90.*LEFT_UNIT_VEC + 90.*DOWN_UNIT_VEC));
    }

    SECTION( "Particle can start and land on cell boundaries." ) {
        const vec<double> start     = {10.0, 10.0, 10.0};
        REQUIRE( check_particle_position(mesh,            100,   start, 5.*BACK_UNIT_VEC));
        REQUIRE( check_particle_position(mesh,            101,   start, 10.*BACK_UNIT_VEC + 5.*RIGHT_UNIT_VEC));
        REQUIRE( check_particle_position(mesh,            111,   start, 10.*BACK_UNIT_VEC + 10.*RIGHT_UNIT_VEC + 10.*UP_UNIT_VEC));
    }

    SECTION( "Particle can start and land on cell boundaries." ) {
        const vec<double> start     = {.012004, .012004, .0122856};

        Mesh<double> *mesh2         = load_mesh(0.3, 50, 0);

        REQUIRE( check_particle_position(mesh2, 5102,   start, 5051));
    }
}
