import os
import numpy
import matplotlib.pyplot as plt
from amuse.units import units
from amuse.ext.star_to_sph import (pickle_stellar_model, convert_stellar_model_to_SPH,)
from amuse.test.amusetest import get_path_to_results
from amuse.community.mesa.interface import MESA
from amuse.datamodel import Particles
from amuse.community.fi.interface import Fi
from amuse.lab import nbody_system
from amuse.datamodel import Grid, new_regular_grid

# def setup_stellar_evolution_model():
#     out_pickle_file = os.path.join(get_path_to_results(), "super_giant_stellar_structure.pkl")

#     stellar_evolution = MESA(redirection = "none")
#     stars = Particles(1)
#     stars.mass = 15.0 | units.MSun
#     stellar_evolution.particles.add_particles(stars)
#     stellar_evolution.commit_particles()

#     print("Evolving a MESA star with mass: ", stellar_evolution.particles[0].mass)

#     try:
#         while stellar_evolution.model_time < 0.12 | units.Myr:
#             stellar_evolution.evolve_model()
#             print("star: ", stellar_evolution.particles[0].stellar_type, stellar_evolution.model_time.in_(units.Myr))
#     except AmuseException as ex:
#         print("Evolved star to", stellar_evolution.particles[0].age)
#         print("Radius: ", stellar_evolution.particles[0].radius)

#     pickle_stellar_model(stellar_evolution.particles[0], out_pickle_file)
#     stellar_evolution.stop()
#     return out_pickle_file

# pickle_file = setup_stellar_evolution_model()
pickle_file = './super_giant_stellar_structure.pkl'
print("Star generated. ")

number_of_sph_particles = 100
print(pickle_file)
print("Creating initial conditions from a MESA stellar evolution model...")
model = convert_stellar_model_to_SPH(
        None,
        number_of_sph_particles,
        seed = 12345,
        pickle_file=pickle_file,
        #        base_grid_options = dict(type = "glass", target_rms = 0.01),
        with_core_particle = True,
        target_core_mass = 1.4|units.MSun
    )
print("model = ", model)
core, gas_without_core, core_radius = \
        model.core_particle, model.gas_particles, model.core_radius
print("Created", len(gas_without_core),
       "SPH particles and one 'core-particle':\n", core)
print("Setting gravitational smoothing to:", core_radius.in_(units.km))

def plot_star(core, gas):
    plt.scatter(gas.x.value_in(units.au), gas.y.value_in(units.au), c = 'b')
    plt.scatter(core.x.value_in(units.au), core.y.value_in(units.au), c = 'r')
    plt.axis("equal")
    plt.show()

plot_star(core, gas_without_core)

def inject_supernova_energy(gas_particles, 
                            explosion_energy=1.0e+51|units.erg,
                            exploding_region=10|units.RSun):
    inner = gas_particles.select(
        lambda pos: pos.length_squared() < exploding_region**2,
        ["position"])
    print(len(inner), "innermost particles selected.")
    print("Adding", explosion_energy / inner.total_mass(), "of supernova " \
        "(specific internal) energy to each of the n=", len(inner), "SPH particles.")
    inner.u += explosion_energy / inner.total_mass()
    
inject_supernova_energy(gas_without_core, exploding_region=1|units.RSun)

converter = nbody_system.nbody_to_si(10|units.MSun, core_radius)

hydro_code = Fi(converter)
hydro_code.parameters.epsilon_squared = core_radius**2
hydro_code.parameters.n_smooth_tol = 0.01
hydro_code.gas_particles.add_particles(gas_without_core)
hydro_code.dm_particles.add_particle(core)

def hydro_plot(hydro_code, view_size, npixels):
    view = [-1, 1, -1, 1] * view_size
    shape = (npixels, npixels, 1)
    size = npixels**2
    axis_lengths = [0.0, 0.0, 0.0] | units.m
    axis_lengths[0] = view[1] - view[0]
    axis_lengths[1] = view[3] - view[2]
    grid = new_regular_grid(shape, axis_lengths)
    grid.x += view[0]
    grid.y += view[2]
    speed = grid.z.reshape(size) * (0 | 1/units.s)
    rho, rhovx, rhovy, rhovz, rhoe = hydro_code.get_hydro_state_at_point(
            grid.x.reshape(size),
            grid.y.reshape(size),
            grid.z.reshape(size), speed, speed, speed)

    # we have to make some cuts in the parameter space.
    min_v = 800.0 | units.km / units.s
    max_v = 3000.0 | units.km / units.s
    min_rho = 3.0e-9 | units.g / units.cm**3
    max_rho = 1.0e-5 | units.g / units.cm**3
    min_E = 1.0e11 | units.J / units.kg
    max_E = 1.0e13 | units.J / units.kg

    v_sqr = (rhovx**2 + rhovy**2 + rhovz**2) / rho**2
    E = rhoe / rho
    log_v = numpy.log((v_sqr / min_v**2)) / numpy.log((max_v**2 / min_v**2))
    log_rho = numpy.log((rho / min_rho)) / numpy.log((max_rho / min_rho))
    log_E = numpy.log((E / min_E)) / numpy.log((max_E / min_E))

    red = numpy.minimum(numpy.ones_like(rho.number), numpy.maximum(
        numpy.zeros_like(rho.number), log_rho)).reshape(shape)
    green = numpy.minimum(numpy.ones_like(rho.number), numpy.maximum(
        numpy.zeros_like(rho.number), log_v)).reshape(shape)
    blue = numpy.minimum(numpy.ones_like(rho.number), numpy.maximum(
        numpy.zeros_like(rho.number), log_E)).reshape(shape)
    alpha = numpy.minimum(
            numpy.ones_like(log_v),
            numpy.maximum(
                numpy.zeros_like(log_v),
                numpy.log((rho / (10*min_rho)))
                )
            ).reshape(shape)

    rgba = numpy.concatenate((red, green, blue, alpha), axis=2)
    
    
    plt.figure(figsize=(npixels/100.0, npixels/100.0), dpi=100)
    plt.imshow(rgba)

hydro_code.evolve_model(10.0|units.s)
print("Done running to time=", hydro_code.model_time.in_(units.s))
view_size = 20 | units.RSun 
npixels = 200
hydro_plot(hydro_code, view_size, npixels)
print("done plotting")

hydro_code.evolve_model(1.0|units.minute)
print("Done running to time=", hydro_code.model_time.in_(units.minute))
hydro_plot(hydro_code, view_size, npixels)

print("v-core=", hydro_code.dm_particles.velocity.length().in_(units.kms))
print("d-core=", hydro_code.dm_particles.position.length().in_(units.RSun))

hydro_code.evolve_model(1.0 | units.hour)
print("Done running:", hydro_code.model_time.in_(units.hour))
hydro_plot(hydro_code, view_size, npixels)

view_size = 100 | units.RSun
while hydro_code.model_time<0.3|units.day:
    hydro_code.evolve_model(hydro_code.model_time+(1|units.hour))
    print("Done running, until:", hydro_code.model_time.in_(units.hour))
    hydro_plot(hydro_code, view_size, npixels)

print("core=", hydro_code.dm_particles.velocity.length().in_(units.kms))
print("core=", hydro_code.dm_particles.position.length().in_(units.RSun))

hydro_code.stop()

