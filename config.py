import numpy as np
import tensorflow as tf
import os
import datetime
import pandas as pd

config = {}

#========================= Make sure main folders exist. ==========================

Root = os. getcwd()
config['root'] = Root
config['data'] = os.path.join( Root, 'data' )
config['history'] = os.path.join( Root, 'history' )
config['dictionary'] = os.path.join( Root, 'dictionary' )
config['models'] = os.path.join( Root, 'models' )
config['reports'] = os.path.join( Root, 'reports' )
config['summary'] = os.path.join( Root, 'summary' )
config['checkPoints'] = os.path.join( Root, 'checkPoints' )

for folder, path in config.items() :
    try:
        if not os.path.exists( path ) or not os.path.isdir( path ) :
            access_rights = 0o755 # =======================
            os.mkdir( path, access_rights ) if not os.path.exists( path ) else None
    except OSError :
        raise Exception( "Couldn't create a folder: {}".format( path ) )

config['minAttends'] = 10 #=================
config['minClusterSize'] = 30

config['unknown_token'] = '[UNK]'
config['bookie_profit_percent'] = 5.0
config['baseDate'] = datetime.datetime(2000, 1, 1)
config['baseGameId'] = int(1E6) # gameId: 1012345, 1012346, ...

# config['np_int'] = np.int32
# config['np_flaot'] = np.float32
# config['tf_int'] = tf.int32
# config['tf_float'] = tf.float32