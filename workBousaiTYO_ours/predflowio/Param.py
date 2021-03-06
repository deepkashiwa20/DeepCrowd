INTERVAL = 30
TIMESTEP = 6
DAYTIMESTEP = int(24 * 60 / INTERVAL)
STARTDATE = '20170401'
ENDDATE = '20170709'
CITY = 'tokyo'
SIZE = '500m'
HEIGHT = 80
WIDTH = 80
CHANNEL = 2
BATCHSIZE = 4
SPLIT = 0.2
LEARN = 0.0001
EPOCH = 200
LOSS = 'mse'
OPTIMIZER = 'adam'
MAX_FLOWIO = 1188.0
dataPath = '../../bousai_{}_jiang_new/'.format(CITY)
dataFile = dataPath + 'flowioK_{}_{}_{}_30min.npy'.format(CITY, STARTDATE, ENDDATE)
trainRatio = 0.8 # 80 days for training and validation, 20 days for testing.