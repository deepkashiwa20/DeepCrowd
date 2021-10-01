Our .npy data is a 4D tensor as follows:

* timestep, height, width, channel = data.shape[0], data.shape[1], data.shape[2], data.shape[3]
* tokyo_density.shape, tokyo_flowio.shape = (4800, 80, 80, 1), (4800, 80, 80, 2)
* osaka_density.shape, osaka_flowio.shape = (4800, 60, 60, 1), (4800, 60, 60, 2)

* For timestep index, please check temporal_index.csv for the detail.
* For height&width index, please check tokyo_spatial_index.csv and osaka_spatial_index.csv for the detail.
* For density tensor, channel = 1, for flowio tensor, channel = 2, first channel is inflow, and second channel is outflow.

The spatiotemporal specification is summarized as follows:
* start_date = '2017-04-01 00:00:00'
* end_date = '2017-07-09 23:30:30'
* time_interval = '30min'
* tokyo_mesh: MINLat = 35.5, MAXLat = 35.82, MINLon = 139.5, MAXLon = 139.9, DLat = 0.004, DLon = 0.005 
* osaka_mesh: MINLat = 34.58, MAXLat = 34.82, MINLon = 135.35, MAXLon = 135.65, DLat = 0.004, DLon = 0.005
