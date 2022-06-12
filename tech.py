from beamngpy import BeamNGpy, Scenario, Vehicle

bng = BeamNGpy("localhost", 64256, home="D:\Downloads\BeamNG tech v0 24 0 2\BeamNG.drive-0.24.0.2.13392")
bng.open(launch=False, deploy=True)

scenario = Scenario("west_coast_usa", "example")

vehicle = Vehicle("ego_vehicle", model="etk800", license="PYTHON")

scenario.add_vehicle(vehicle, pos=(-17, 101, 118), rot=None, rot_quat=(0, 0, 0.3826834, 0.9238795))

scenario.make(bng)

bng.load_scenario(scenario)
bng.start_scenario()

vehicle.ai_set_mode("span")
print(bng)
input("hit enter when done")