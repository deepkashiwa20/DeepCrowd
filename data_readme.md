
Our .npy data is a 4D tensor as follows:

timestep, height, width, channel = data.shape[0], data.shape[1], data.shape[2], data.shape[3]
tokyo_density.shape, tokyo_flowio.shape = (4800, 80, 80, 1), (4800, 80, 80, 2)
osaka_density.shape, osaka_flowio.shape = (4800, 60, 60, 1), (4800, 60, 60, 2)

For timestep index, please check temporal_index.csv for the detail.
For height&width index, please check tokyo_spatial_index.csv and osaka_spatial_index.csv for the detail.
For density tensor, channel = 1, for flowio tensor, channel = 2, first channel is inflow, and second channel is outflow.

The spatiotemporal specification is summarized as follows:
