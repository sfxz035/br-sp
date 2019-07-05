from network.ops import *
import tensorflow as tf
from utils.compute import *
mean_x = 127  # tf.reduce_mean(self.input)
mean_y = 127  # tf.reduce_mean(self.input)

def EDSR(input,reuse = False,args=None,name='EDSR'):
    with tf.variable_scope(name,reuse=reuse):
        L1 = conv_relu(input,args.EDFILTER_DIM,name='Conv2d_1')
        x = L1
        for i in range(args.nubBlocks_ED):
            x = resBlock_ED(x,args.EDFILTER_DIM,scale=args.resScale,name='Block_'+str(i))
        L2 = conv_relu(x,args.EDFILTER_DIM,name='conv2d_2')
        L_res = L2+L1
        L_U = upsample(L_res,args.EDFILTER_DIM,scale=args.scale)
        output = tf.clip_by_value(L_U+mean_x,0.0,255.0)
    return output

def net(input,reuse=False,is_training=True,args=None,name='RDN'):
    with tf.variable_scope(name,reuse=reuse):
        F_1 = conv_b(input,64,name='conv2d_1')
        F_0 = conv_b(F_1,64,name='conv2d_2')
        rdb_list = []
        rdb_in = F_0
        for i in range(1,17):
            x = rdb_in
            for j in range(1,9):
                tmp = conv_relu(x,64,name='RDB_'+str(i)+'_'+str(j))
                x = tf.concat([x,tmp],axis=3)
            x = conv_b(x,64,k_h=1,k_w=1,name='RDB_'+str(i))
            rdb_in = x+rdb_in
            rdb_list.append(rdb_in)
        FD = tf.concat(rdb_list,axis=3)
        FGF1 = conv_b(FD,64,k_h=1,k_w=1,name='conv2d_F1')
        FGF2 = conv_b(FGF1,64,name='conv2d_F2')
        FDF = tf.add(F_1,FGF2)

        ### upsamlple
        upsamp1 = conv_relu(FDF,64,name='conv_up1')
        upsamp2 = conv_relu(upsamp1,32,name='conv_up2')
        up = conv_b(upsamp2,48,name='conv_up3')
        up = pixelShuffler(up,4)

        conv_out = conv_b(up,3,name='conv_out')
        return conv_out