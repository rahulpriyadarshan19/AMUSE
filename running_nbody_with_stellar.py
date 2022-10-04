#simulating a star cluster
from re import T
import numpy as np
import matplotlib.pyplot as plt
from amuse.units import units
from amuse.lab import new_powerlaw_mass_distribution
from amuse.units import nbody_system
from amuse.ic.plummer import new_plummer_model
from amuse.community.ph4.interface import ph4
from amuse.community.seba.interface import SeBa
from amuse.ext.LagrangianRadii import LagrangianRadii

n_stars = 100
alpha_IMF = -2.35
np.random.seed(0)
m_stars = new_powerlaw_mass_distribution(n_stars,
                                         10.0 | units.MSun,
                                         100.0 | units.MSun,
                                         alpha_IMF)
r_cluster = 1.0 | units.parsec
converter = nbody_system.nbody_to_si(m_stars.sum(), r_cluster)
stars = new_plummer_model(n_stars, convert_nbody = converter)
stars.mass = m_stars
stars.scale_to_standard(converter)

def plot_snapshot(bodies):
    v = (bodies.vx**2 + bodies.vy**2 + bodies.vz**2).sqrt()
    s = bodies.mass.value_in(units.MSun)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (11,3))
    ax1.scatter(stars.temperature.value_in(units.K),
                stars.luminosity.value_in(units.LSun),
                c = v.value_in(units.kms),
                s = s)
    ax1.set_xlim(6.e+4, 20000)
    ax1.set_ylim(1.e+3, 1.e+7)
    ax1.loglog()
    ax1.set_xlabel("T [K]")
    ax1.set_ylabel("L [$L_{\odot}$]")
    ax2.scatter(bodies.x.value_in(units.parsec),
                bodies.y.value_in(units.parsec),
                c = v.value_in(units.kms),
                s = s)
    ax2.set_xlabel("x [pc]")
    ax2.set_ylabel("y [pc]")
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    plt.show()

#defining the solver
stellar = SeBa()
stellar.particles.add_particles(stars)

gravity = ph4(converter)
gravity.particles.add_particles(stars)
ch_s2g = stellar.particles.new_channel_to(gravity.particles)
ch_g2l = gravity.particles.new_channel_to(stars)
ch_s2l = stellar.particles.new_channel_to(stars)
ch_s2l.copy()

plot_snapshot(stars)

end_time = 10.0 | units.Myr
model_time = 0 | units.Myr
model_times = []
virial_radius = []
kinetic_energy = []
potential_energy = []
total_energy = []
while(model_time < end_time):
    dt = stellar.particles.time_step.min()
    model_time += dt
    stellar.evolve_model(model_time)
    ch_s2g.copy()
    ch_s2l.copy()
    gravity.evolve_model(model_time)
    ch_g2l.copy()
    model_times.append(model_time.value_in(units.Myr))
    virial_radius.append(stars.virial_radius().value_in(units.parsec))
    KE = stars.kinetic_energy()
    kinetic_energy.append(KE.value_in(units.J))
    PE = stars.potential_energy()
    potential_energy.append(PE.value_in(units.J))
    TE = KE + PE
    total_energy.append(TE.value_in(units.J))
    print("Evolved to t = ", stellar.model_time.in_(units.Myr),
          gravity.model_time.in_(units.Myr),
          "mass = ", stars.mass.sum().in_(units.MSun),
          "rvir = ", stars.virial_radius().in_(units.parsec),
          "Total energy = ", TE)

    # b = stars.get_binaries()
    # if (len(b) > 0):
    #     print("Number of binaries found: ", len(b))
    #     print("First binary:", b[0])
    #     break

plot_snapshot(stars)
stellar.stop()
gravity.stop()

# Question 1: differences between first and last HR diagram and positions 
# In the first HR diagram, all stars were in the main sequence whereas in the last one, 
# some stars have crossed the main sequence. 
# The cluster has expanded after the evolution

# Assignment 1: virial radius as a function of time
plt.plot(model_times, virial_radius)
plt.xlabel("Time (Myr)")
plt.ylabel("Virial radius (pc)")
plt.show()

# Assignment 2: running without stellar evolution
# ws = without stellar
virial radius does not change much if there is no stellar evolution
m_stars_ws = new_powerlaw_mass_distribution(n_stars,
                                         10.0 | units.MSun,
                                         100.0 | units.MSun,
                                         alpha_IMF)
r_cluster_ws = 1.0 | units.parsec
converter_ws = nbody_system.nbody_to_si(m_stars_ws.sum(), r_cluster_ws)
stars_ws = new_plummer_model(n_stars, convert_nbody = converter_ws)
stars_ws.mass = m_stars_ws
stars_ws.scale_to_standard(converter_ws)

# #defining the solver
gravity_ws = ph4(converter_ws)
gravity_ws.particles.add_particles(stars_ws)
channel = gravity_ws.particles.new_channel_to(stars_ws)
times = np.linspace(0, 10, 10000) | units.Myr
virial_radius_ws = []
print("Without stellar evolution")
print(times)
for time in times:
    gravity_ws.evolve_model(time)
    channel.copy()
    print("Evolved to t = ", time.in_(units.Myr),
          "mass = ", stars_ws.mass.sum().in_(units.MSun),
          "rvir = ", stars_ws.virial_radius().in_(units.parsec))
    virial_radius_ws.append(stars_ws.virial_radius().value_in(units.parsec))

print(virial_radius_ws)
plt.plot(model_times, virial_radius, label = "With stellar evolution")
plt.plot(times.value_in(units.Myr), virial_radius_ws, label = "Without stellar evolution")
plt.xlabel("Time (Myr)")
plt.ylabel("Virial radius (pc)")
plt.legend()
plt.show()

# Question 5: running till a binary forms
# By the time a binary forms, the virial radius increases significantly and the
# cluster is not gravitationally bound anymore

# Question 6: plots of KE, PE and TE with time
# Total energy increases in the middle by a small amount but largely remains the same
plt.plot(model_times, kinetic_energy, label = "Kinetic energy")
plt.plot(model_times, potential_energy, label = "Potential energy")
plt.plot(model_times, total_energy, label = "Total energy")
plt.xlabel("Time (Myr)")
plt.ylabel("Energy (J)")
plt.legend()
plt.show()
