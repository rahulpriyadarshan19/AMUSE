from tracemalloc import start
from urllib.parse import ParseResult
import numpy as np
import matplotlib.pyplot as plt
from amuse.units import units
from amuse.datamodel import Particles
from amuse.lab import new_kroupa_mass_distribution, new_salpeter_mass_distribution
from amuse.community.seba.interface import SeBa
from amuse.plot import plot, scatter

#setting up a simulation by specifying IMFs
n_stars = 1024
m_min = 0.1 | units.MSun
m_max = 100 | units.MSun

#exponent changes between the Salpeter and Kroupa IMFs
m_kroupa = new_kroupa_mass_distribution(n_stars, mass_min = m_min, mass_max = m_max)
k_stars = Particles(mass = m_kroupa)
m_salpeter = new_salpeter_mass_distribution(n_stars, mass_min = m_min, mass_max = m_max)
s_stars = Particles(mass = m_salpeter)
print("Mean mass for Kroupa = ", np.mean(k_stars.mass))
print("Mean mass for Salpeter = ", np.mean(s_stars.mass))
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.hist(np.log10(k_stars.mass.value_in(units.MSun)))
plt.title("Kroupa IMF")
plt.yscale('log')
#plt.xlim(-1,2)
fig.add_subplot(1,2,2)
plt.hist(np.log10(s_stars.mass.value_in(units.MSun)))
plt.title("Salpeter IMF")
plt.yscale("log")
#plt.xlim(-1,2)
plt.show()

#setting up the simulation
def start_stellar_code(stars, metallicity = 0.02):
    stellar = SeBa()
    stellar.particles.metallicity = metallicity
    stellar.particles.add_particles(stars)
    channels = {"to_stars": stellar.particles.new_channel_to(stars),
                "to_stellar": stars.new_channel_to(stellar.particles)}
    return stellar, channels

k_stellar, k_channels = start_stellar_code(k_stars)
s_stellar, s_channels = start_stellar_code(s_stars)

times = 10**np.arange(0.0, 4.0, 0.1) | units.Myr
m_mean = []
for time in times:
    k_stellar.evolve_model(time)
    k_channels["to_stars"].copy()
    s_stellar.evolve_model(time)
    s_channels["to_stars"].copy()
    m_mean.append(np.mean(k_stars.mass)/np.mean(s_stars.mass))
k_stellar.stop()
s_stellar.stop()

print("Mean mass for Kroupa = ", np.mean(k_stars.mass))
print("Mean mass for Salpeter = ", np.mean(s_stars.mass))

plot(times, m_mean)
plt.ylabel("Relative mean mass")
plt.semilogx()
plt.show()

scatter(s_stars.temperature, s_stars.luminosity, c = "r", label = "Salpeter")
scatter(k_stars.temperature, k_stars.luminosity, c = "b", s = 3, label = "Kroupa")
plt.xlim(2.e+4, 2000)
plt.ylim(1.e-5, 1000)
plt.loglog()
plt.legend()
plt.show()

#Assignment 1: why does relative mean mass reduce with time? 
#Kroupa IMF has more stars in solar mass bins, which means they are heavier stars which 
#run out of hydrogen faster

#Assignment 2: mean mass of star populations with different metallicities (0.02 and 0.002)
k_stars_002 = Particles(mass = m_kroupa)
k_stars_0002 = Particles(mass = m_kroupa)
s_stars_002 = Particles(mass = m_salpeter)
s_stars_0002 = Particles(mass = m_salpeter)   
#same k_stars initialized but will get updated differently because of different metallicities,
#hence two different quantities
k_stellar_002, k_channels_002 = start_stellar_code(k_stars_002, metallicity = 0.02)
k_stellar_0002, k_channels_0002 = start_stellar_code(k_stars_0002, metallicity = 0.002)
s_stellar_002, s_channels_002 = start_stellar_code(s_stars_002, metallicity = 0.02)
s_stellar_0002, s_channels_0002 = start_stellar_code(s_stars_0002, metallicity = 0.002) 

times = 10**np.arange(0.0, 4.0, 0.1) | units.Myr
L_total_k_002 = []
L_total_k_0002 = []
L_total_s_002 = []
L_total_s_0002 = []
T_LW_k_002 = []
T_LW_k_0002 = []
T_LW_s_002 = []
T_LW_s_0002 = []
for time in times:
    k_stellar_002.evolve_model(time)
    k_channels_002["to_stars"].copy()
    L_total_k_002.append(np.sum(k_stars_002.luminosity).value_in(units.LSun))
    T_LW_k_002.append((np.sum(k_stars_002.luminosity*k_stars_002.temperature)/np.sum(k_stars_002.luminosity)).value_in(units.K))

    k_stellar_0002.evolve_model(time)
    k_channels_0002["to_stars"].copy()
    L_total_k_0002.append(np.sum(k_stars_0002.luminosity).value_in(units.LSun))
    T_LW_k_0002.append((np.sum(k_stars_0002.luminosity*k_stars_0002.temperature)/np.sum(k_stars_0002.luminosity)).value_in(units.K))

    s_stellar_002.evolve_model(time)
    s_channels_002["to_stars"].copy()
    L_total_s_002.append(np.sum(s_stars_002.luminosity).value_in(units.LSun))
    T_LW_s_002.append((np.sum(s_stars_002.luminosity*s_stars_002.temperature)/np.sum(s_stars_002.luminosity)).value_in(units.K))

    s_stellar_0002.evolve_model(time)
    s_channels_0002["to_stars"].copy()
    L_total_s_0002.append(np.sum(s_stars_0002.luminosity).value_in(units.LSun))
    T_LW_s_0002.append((np.sum(s_stars_0002.luminosity*s_stars_0002.temperature)/np.sum(s_stars_0002.luminosity)).value_in(units.K))

print("Mean mass of z = 0.02 stars: ", np.mean(k_stars_002.mass))
print("Mean mass of z = 0.002 stars: ", np.mean(k_stars_0002.mass))
#exactly the same?

#Question 1: comparing differences in compact objects and main sequence stars 
#for different metallicities and IMFs
numbers_k_002, counts_k_002 = np.unique(k_stars_002.stellar_type, return_counts = True)
numbers_k_0002, counts_k_0002 = np.unique(k_stars_0002.stellar_type, return_counts = True)
numbers_s_002, counts_s_002 = np.unique(s_stars_002.stellar_type, return_counts = True)
numbers_s_0002, counts_s_0002 = np.unique(s_stars_0002.stellar_type, return_counts = True)
print("Kroupa IMF:")
print("z = 0.02: Numbers: ", numbers_k_002, "Counts: ", counts_k_002)
print("z = 0.002: Numbers: ", numbers_k_0002, "Counts: ", counts_k_0002)
print("Salpeter IMF")
print("z = 0.02: Numbers: ", numbers_s_002, "Counts: ", counts_s_002)
print("z = 0.002: Numbers: ", numbers_s_0002, "Counts: ", counts_s_0002)
# Kroupa:
# z = 0.02: Main sequence - 948, Compact objects - 71
# z = 0.002: Main sequence - 948, Compact objects - 71
# Salpeter:
# z = 0.02: Main sequence - 983, Compact objects - 38
# z = 0.002: Main sequence - 983, Compact objects- 38
# Metallicity doesn't seem to change number of compact objects or main sequence stars
# Salpeter IMF produces more main sequence stars and less compact objects and vice versa.
# Because Kroupa IMF has more stars with initial mass much greater than the solar mass.

# Assignment 3: total luminosity and luminosity-weighted temperature
plt.plot(times.value_in(units.Myr), L_total_k_002, label = "Kroupa IMF, z = 0.02")
plt.plot(times.value_in(units.Myr), L_total_k_0002, label = "Kroupa IMF, z = 0.002")
plt.plot(times.value_in(units.Myr), L_total_s_002, label = "Salpeter IMF, z = 0.02")
plt.plot(times.value_in(units.Myr), L_total_s_0002, label = "Salpeter IMF, z = 0.002")
plt.ylabel("Luminosity $(L_{\odot})$")
plt.xlabel("Times (Myr)")
plt.semilogx()
plt.semilogy()
plt.legend()
plt.show()

plt.plot(times.value_in(units.Myr), T_LW_k_002, label = "Kroupa IMF, z = 0.02")
plt.plot(times.value_in(units.Myr), T_LW_k_0002, label = "Kroupa IMF, z = 0.002")
plt.plot(times.value_in(units.Myr), T_LW_s_002, label = "Salpeter IMF, z = 0.02")
plt.plot(times.value_in(units.Myr), T_LW_s_0002, label = "Salpeter IMF, z = 0.002")
plt.ylabel("Luminosity-weighted temperature $T_{LW}$ (K)")
plt.xlabel("Times (Myr)")
plt.semilogx()
plt.semilogy()
plt.legend()
plt.show()

# Assignment 4: Luminosity and luminosity-weighted temperature are higher at most times
# for Kroupa IMF compared to Salpeter IMF because it creates stars with larger initial mass