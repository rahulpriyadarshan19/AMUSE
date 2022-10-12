from pydoc import resolve
import numpy as np
np.random.seed(98429435)
import matplotlib.pyplot as plt
from amuse.units import units, nbody_system
from amuse.lab import new_powerlaw_mass_distribution
from amuse.ic.plummer import new_plummer_model
from amuse.community.ph4.interface import ph4
from amuse.community.seba.interface import SeBa
from amuse.ext.LagrangianRadii import LagrangianRadii
from amuse.lab import Particles

#initializing N-body code
n_stars = 100
alpha_IMF = -2.35
m_stars = new_powerlaw_mass_distribution(n_stars,
                                          10.0 | units.MSun,
                                          100.0 | units.MSun,
                                          alpha_IMF)
r_cluster = 1.0 | units.parsec
converter = nbody_system.nbody_to_si(m_stars.sum(), r_cluster)
stars = new_plummer_model(n_stars, convert_nbody = converter)
stars.mass = m_stars
setattr(stars, "collision radius", 0 | units.RSun)
stars.scale_to_standard(converter)

stellar = SeBa()
stellar.particles.add_particles(stars)

gravity = ph4(converter, number_of_workers = 6)
gravity.particles.add_particles(stars)

#declaring channels
stellar_attributes = ["mass", "radius", "age", "temperature", "luminosity"]
channel = {"from_stellar": stellar.particles.new_channel_to(stars, 
                                                            attributes = stellar_attributes,
                                                            target_names = stellar_attributes),
           "from_gravity": gravity.particles.new_channel_to(stars,
                                                            attributes = ["x", "y", "z", "vx", "vy", "vz", "mass"],
                                                            target_names = ["x", "y", "z", "vx", "vy", "vz", "mass"]),
           "to_gravity": stars.new_channel_to(gravity.particles,
                                              attributes = ["mass", "collision_radius"],
                                              target_names = ["mass", "radius"])}

channel["from_stellar"].copy()

def plot_snapshot(bodies):
    v = (bodies.vx**2 + bodies.vy**2 + bodies.vz**2).sqrt()
    s = bodies.mass.value_in(units.MSun)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (11, 3))
    ax1.scatter(stars.temperature.value_in(units.K),
                stars.luminosity.value_in(units.LSun),
                c = v.value_in(units.kms),
                s = s)
    ax1.set_xlim(6.e+4, 20000)
    ax1.set_ylim(1.e+3, 1.e+7)
    ax1.loglog()
    ax1.set_xlabel("T [K]")
    ax1.set_ylabel("L [$L_{\odot}$]")

    ax2.scatter(bodies.x.value_in(units.pc),
                bodies.y.value_in(units.pc),
                c = v.value_in(units.kms),
                s = s)
    plt.gca().set_aspect('equal', adjustable = 'box')
    ax2.set_xlabel("x [pc]")
    ax2.set_ylabel("y [pc]")
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    plt.show()

plot_snapshot(stars)

#stopping conditions
stopping_condition = gravity.stopping_conditions.collision_detection
stopping_condition.enable()
collision_radius_multiplication_factor = 1000   #for many collisions

def merge_two_stars(bodies, particles_in_encounter):
    com_pos = particles_in_encounter.center_of_mass()
    com_vel = particles_in_encounter.center_of_mass_velocity()
    d = (particles_in_encounter[0].position - particles_in_encounter[1].position)
    v = (particles_in_encounter[0].velocity - particles_in_encounter[1].velocity)
    print("Actual merger occurred: ")
    print("Two stars (M = ", particles_in_encounter.mass.in_(units.MSun),
          ") collided with d = ", d.length().in_(units.au))

    new_particle = Particles(1)
    new_particle.mass = particles_in_encounter.total_mass()
    new_particle.age = min(particles_in_encounter.age)*max(particles_in_encounter.mass)/new_particle.mass   
    new_particle.position = com_pos
    new_particle.velocity = com_vel
    new_particle.radius = particles_in_encounter.radius.sum()
    bodies.add_particles(new_particle)
    bodies.remove_particles(particles_in_encounter)

def resolve_collision(collision_detection, gravity, stellar, bodies):
    f = 0
    if collision_detection.is_set():
        f = 1
        print("Well, we have an actual collision betweee two or more stars.")
        print("This happened at time = ", stellar.model_time.in_(units.Myr))  
        for ci in range(len(collision_detection.particles(0))):
            encountering_particles = Particles(particles = [collision_detection.particles(0)[ci],
                                                            collision_detection.particles(1)[ci]])
            colliding_stars = encountering_particles.get_intersecting_subset_in(bodies)
            merge_two_stars(bodies, colliding_stars)
            bodies.synchronize_to(gravity.particles)
            bodies.synchronize_to(stellar.particles)
    return f

end_time = 10.0 | units.Myr
model_time = 0 | units.Myr
rvir = [] | units.pc
time = [] | units.Myr
t_diag = 1 | units.Myr
number_of_collisions = 0
while (model_time < end_time):
    dt = stellar.particles.time_step.min()
    model_time += dt
    stellar.evolve_model(model_time)
    channel["from_stellar"].copy()
    stars.collision_radius = stars.radius * collision_radius_multiplication_factor
    channel["to_gravity"].copy()
    gravity.evolve_model(model_time)
    f = resolve_collision(stopping_condition, gravity, stellar, stars)
    number_of_collisions += f
    channel["from_gravity"].copy()

    if model_time >= t_diag:
        t_diag += 1 | units.Myr

        print("Evolved to t = ", stellar.model_time.in_(units.Myr),
              gravity.model_time.in_(units.Myr),
              "N = ", len(stars),
              "mass = ", stars.mass.sum().in_(units.MSun),
              "rvir = ", stars.virial_radius().in_(units.pc))
    rvir.append(stars.virial_radius())
    time.append(model_time)

stellar.stop()
gravity.stop()

plot_snapshot(stars)
plt.plot(time.value_in(units.Myr), rvir.value_in(units.pc))
plt.xlabel("t [Myr]")
plt.ylabel("$R_{vir}$ [pc]")
plt.show()

# question 1:
# run with a random seed so that the same run can be repeated again with the initial mass distribution
# number of collisions is 7
print("Number of collisions: ", number_of_collisions)
# stars not on the main sequence are giants
# locations of collisions wrt density centre, the core radius, and the core density
help(stars.densitycentre_coreradius_coredens)

# assignment 1:

