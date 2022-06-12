from beamngpy import BeamNGpy

client = BeamNGpy("localhost", 64256, home="D:\Downloads\BeamNG tech v0 24 0 2\BeamNG.drive-0.24.0.2.13392")
client.open(launch=False, deploy=True)
print(client)