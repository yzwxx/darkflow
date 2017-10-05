from .defaults import argHandler #Import the default arguments, relative import
import os
from .net.build import TFNet

def cliHandler(args):
    FLAGS = argHandler()
    FLAGS.setDefaults()
    FLAGS.parseArgs(args)

    # make sure all necessary dirs(including ./imgdir,./bin,./ckpt,./imgdir/out and ./summary/) exist
    def _get_dir(dirs):
        for d in dirs:
            this = os.path.abspath(os.path.join(os.path.curdir, d))
            if not os.path.exists(this): os.makedirs(this)
    _get_dir([FLAGS.imgdir, FLAGS.binary, FLAGS.backup, 
             os.path.join(FLAGS.imgdir,'out'), FLAGS.summary])

    # fix FLAGS.load to appropriate type
    try: FLAGS.load = int(FLAGS.load)
    except: pass

    # build our detection model
    tfnet = TFNet(FLAGS)
    
    if FLAGS.demo: # using webcam
        tfnet.camera()
        exit('Demo stopped, exit.')

    if FLAGS.train: # train the whole net
        print('Enter training ...'); tfnet.train()
        if not FLAGS.savepb: 
            exit('Training finished, exit.')

    if FLAGS.savepb: # save net and weight to a .pb file
        print('Rebuild a constant version ...')
        tfnet.savepb(); exit('Done')

    tfnet.predict()
