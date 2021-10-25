YahooBousaiCrowd data is now available at Yahoo! Japan Research website.
* https://randd.yahoo.co.jp/en/softwaredata
* Please fill in Application PDF for Use of Yahoo! Bousai Crowd Data. (See the sample PDF in this directory.)
* https://s.yimg.jp/dl/research_lab/randd/software_data/application_form_YJ21_10042184_en.pdf
* The fields below the underline on the application form do not need to be filled in.
* Send it to yjresearch-data “at” mail.yahoo.co.jp.

YahooBousaiCrowd data contrains the following files:
* density_tokyo_20170401_20170709_30min.npy
* flowio_tokyo_20170401_20170709_30min.npy
* density_osaka_20170401_20170709_30min.npy
* flowio_osaka_20170401_20170709_30min.npy
* temporal_index.csv
* tokyo_spatial_index.csv
* osaka_spatial_index.csv
* readme.md

Each .npy data is a 4D tensor described as follows:
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

The maximum value for the current 0-1 normalized 4D tensor is as follows:
* tokyo_density, tokyo_flowio, osaka_density, osaka_flowio = 2300, 1200, 1800, 770
* The original maximum values used in the paper is 23xx, 12xx, 18xx, 77x. 
* However, according to the company policy of Yahoo Japan Corporation, only the most significant two digits can be published. 
* Thus, the reproduced results might slightly differ from the paper.


Please refer the following paper information when you write some document with this data:

* Plain text format
R. Jiang et al., "DeepCrowd: A Deep Model for Large-Scale Citywide Crowd Density and Flow Prediction," in IEEE Transactions on Knowledge and Data Engineering, doi: 10.1109/TKDE.2021.3077056.

*BibTex format
@ARTICLE{9422199,
  author={Jiang, Renhe and Cai, Zekun and Wang, Zhaonan and Yang, Chuang and Fan, Zipei and Chen, Quanjun and Tsubouchi, Kota and Song, Xuan and Shibasaki, Ryosuke},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={DeepCrowd: A Deep Model for Large-Scale Citywide Crowd Density and Flow Prediction}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TKDE.2021.3077056}}


If you have any question, please contact to the following e-mail.
* ktsubouc@yahoo-corp.jp


Thank you!
