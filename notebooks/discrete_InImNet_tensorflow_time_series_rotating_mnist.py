import numpy as np
import os
import tensorflow as tf
#tf.get_logger().setLevel('ERROR')
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

GPU_ID=0
ELL_MAX = 4
NTOTAL = 50
INFLATION_FACTOR = 2
ACTIVATION = tf.keras.activations.relu
NUM_INTERNAL_LAYERS = 2
BATCH_SIZE = 25
I_EVAL = 4
N_MASK = 3
NUM_IMAGES_TOTAL = 16 
NUM_IMAGES_CONTEXT = 1
DIM_LATENT = 16 *  64
INNER_DIM_LATENT=20
DROPOUT_VALUE=0.3
LR_INIM = 0.001
NUM_EPOCHS = 500
REPORT_INTERVAL_EPOCHS = 5
t = np.linspace(0, 1, NUM_IMAGES_TOTAL).astype(np.float32)
t_eval = np.linspace(0, 1, NUM_IMAGES_TOTAL).astype(np.float32)
print ('t: ', t)
print ('t_eval: ', t_eval)


mpl.rcParams['figure.figsize'] = (8, 6)

class InputAutoencoder (tf.keras.Model):
    def __init__(self, represent_dim_in=50, represent_dim_out=50):
        super(InputAutoencoder, self).__init__()
        
        self.encoder_input = []
        self.decoder_input = []


        self.encoder_input.append(tf.keras.layers.Conv2D(16, (5, 5), strides=2, padding='same'))
        self.encoder_input.append(tf.keras.layers.BatchNormalization())
        self.encoder_input.append(tf.keras.layers.ReLU())
        self.encoder_input.append(tf.keras.layers.Conv2D(32, (5, 5), strides=2, padding='same'))
        self.encoder_input.append(tf.keras.layers.BatchNormalization())
        self.encoder_input.append(tf.keras.layers.ReLU())
        self.encoder_input.append(tf.keras.layers.Conv2D(64, (5, 5), strides=2, padding='same'))
        self.encoder_input.append(tf.keras.layers.ReLU())
        #self.encoder_input.append(tf.keras.layers.BatchNormalization())
        #self.encoder_input.append(tf.keras.layers.Conv2D(4, (5, 5), strides=1, activation='relu', padding='same'))
        #self.encoder_input.append(tf.keras.layers.Flatten())


        self.dim_reducer = tf.keras.layers.Dense (represent_dim_in)
        self.dim_reconstructor = tf.keras.layers.Dense (represent_dim_out)
        
        #self.max_pool_output_shape = [6, 6, 8]
        #self.flattened_max_pool_output_shape = 288

        #self.decoder_input.append(tf.keras.layers.Dense(self.flattened_max_pool_output_shape, activation='relu')) 
        #self.decoder_input.append(tf.keras.layers.Reshape(self.max_pool_output_shape))
        self.decoder_input.append(tf.keras.layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same'))
        self.decoder_input.append(tf.keras.layers.BatchNormalization())
        self.decoder_input.append(tf.keras.layers.ReLU())
        self.decoder_input.append(tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
        self.decoder_input.append(tf.keras.layers.BatchNormalization())
        self.decoder_input.append(tf.keras.layers.ReLU())
        self.decoder_input.append(tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=2, padding='same'))
        self.decoder_input.append(tf.keras.layers.BatchNormalization())
        self.decoder_input.append(tf.keras.layers.ReLU())
        self.decoder_input.append(tf.keras.layers.Conv2DTranspose(1, (5, 5), padding='same', activation='sigmoid'))


    def call_encoder(self, x, training):
        z = x               
        for i in range (len(self.encoder_input)):       
            z = self.encoder_input[i](z, training=training)   
            #print ('encoder['+str(i)+'].shape: ', z.shape)
        return z

    def call_dim_reducer(self, x, training):
        return self.dim_reducer (x, training=training) 

    def call_dim_reconstructor(self, x, training):
        return self.dim_reconstructor (x, training=training) 

    def call_decoder (self, x, training):
        z = x               
        for i in range (len(self.decoder_input)):       
            z = self.decoder_input[i](z, training=training)   
            #print ('decoder['+str(i)+'].shape: ', z.shape)
        z = z[:, 2:-2, 2:-2, :]
        return z
    


class AutoDiffInImNet(tf.keras.Model):
    """Choose output method for model."""

    def __init__(self, dim: int, represent_dim_out: int,
                 num_resnet_layers: int,
                 activation,
                 num_internal_layers: int,
                 bias_on: bool = True,
                 mult: int = 1,               
                 use_batch_norm: bool = False,
                 dropout=0.1,
                 lr_inim = 0.001,
                 approx_jacobian = True,
                 weight_regularisation_alpha = 0.0, 
                 n_mask = 3,
                 i_mask = 4):
        super(AutoDiffInImNet, self).__init__()
        self.num_internal_layers = num_internal_layers
        self.num_resnet_layers = num_resnet_layers
        self.activation = activation
        self.dim = dim
        self.n_mask = n_mask
        self.i_mask = i_mask
        self.lr_inim = lr_inim#tf.keras.optimizers.schedules.PiecewiseConstantDecay (boundaries=[128], values=[lr_inim/50, lr_inim])#tf.keras.optimizers.schedules.ExponentialDecay(lr_inim, decay_steps=15625, decay_rate=0.5, staircase=True)
        self.represent_dim_out = represent_dim_out
        self.weight_regularisation_alpha = weight_regularisation_alpha
        self.t_reshaped_mask = None
        self.row_val = None

        self.fc_network = []
        for i in range(num_resnet_layers):
            self.fc_network.append([])
            for j in range (num_internal_layers-1):
                self.fc_network[i].append(tf.keras.layers.Dense(dim*mult, activation=activation))
                #self.fc_network[i].append(tf.keras.layers.BatchNormalization())
                #self.fc_network[i].append(tf.keras.layers.Activation(activation))
                self.fc_network[i].append(tf.keras.layers.Dropout(dropout))
                
            self.fc_network[i].append(tf.keras.layers.Dense(dim+1, activation=activation))
        self.optimiser = []
        for i in range (self.num_resnet_layers):
            self.optimiser.append(tf.keras.optimizers.Adam(learning_rate=self.lr_inim))
        self.approx_jacobian = approx_jacobian
        self.autoencoder =  InputAutoencoder (represent_dim_in=dim, represent_dim_out=represent_dim_out)

        
    def call_phi (self, ell, x, training):
        z = x               
        for i in range (len(self.fc_network[ell])):       
            z = self.fc_network[ell][i](z, training=training)    
        return z

    #@tf.function
    def optimise (self, x, y_in, t, loss): 
        self.output_dim = y_in.shape[2]
        #print ('optimise')

        with tf.GradientTape(persistent=True) as g: 
            z, out_mask = self (x, t, training=True, use_mask = True)
            y_in_shape = self._infer_shape (y_in)
            y = tf.reshape (y_in, [y_in_shape[0] * y_in_shape[1], y_in_shape[2], y_in_shape[3]])
            print ('y.shape: ', y.shape)
            print ('z[0].shape: ', z[0].shape)
            print ('out_mask.shape: ', out_mask.shape)
            y = tf.boolean_mask(y, out_mask)
            losses = []
            total_loss = 0
            for i in range (self.num_resnet_layers):
                #print (y[:, i, :])
                curr_loss = tf.math.reduce_mean(loss(z[i+1], y))
                losses.append(curr_loss)
                total_loss = total_loss + curr_loss
            print ('losses: ', losses)
            #for i in range (self.num_resnet_layers):
            #    for fc_ij in self.fc_network[i]:
            #        print ('fc_ij.trainable_weights: ', fc_ij.trainable_weights)

            #   print ('[fc_ij.trainable_weights for fc_ij in self.encoder[i]+[self.decoder[i]]]: ', [fc_ij.trainable_weights for fc_ij in self.fc_network[i]])
            #    #if i < self.num_resnet_layers-1:
            #    self.optimiser[i].minimize (losses[i],[fc_ij.trainable_weights for fc_ij in self.fc_network[i]], tape=g)
                #else:
                #self.optimiser[i].minimize (losses[i],[fc_ij.trainable_weights for fc_ij in self.fc_network[i]+self.autoencoder.encoder_input+self.autoencoder.decoder_input+[self.autoencoder.dim_reducer]+[self.autoencoder.dim_reconstructor]], tape=g)
            #    print ('Minimisation finished...')

            norm_w = []
            for i in range (self.num_resnet_layers):
                for fc_ij in self.fc_network[i]:
                    norm_w.extend (fc_ij.trainable_weights)
            print ('norm_w: ', norm_w)
            norms = [0.5 * tf.reduce_sum(tf.square(fc_ij_weights)) for fc_ij_weights in norm_w]
            print ('norms: ', norms)
            w_regularisation = tf.reduce_sum(norms)
            print ('w_regularisation: ', w_regularisation)
            final_loss = losses[-1]+self.weight_regularisation_alpha * w_regularisation

        w = []
        for i in range (self.num_resnet_layers):
            w.extend ([fc_ij.trainable_weights for fc_ij in self.fc_network[i]])  
        w.extend ([fc_ij.trainable_weights for fc_ij in self.autoencoder.encoder_input+self.autoencoder.decoder_input+[self.autoencoder.dim_reducer]+[self.autoencoder.dim_reconstructor]])
        self.optimiser[0].minimize (final_loss, w, tape=g)

        #for i in range (len(losses)):
        #   self.autoencoder.optimise_autoencoder (losses[i], g)
        
        
        return final_loss
                  
    def jacobian (self, z, x, g):
        return g.batch_jacobian(z, x) 

    def _infer_shape(self, x):
        x = tf.convert_to_tensor(x)

        # If unknown rank, return dynamic shape
        if x.shape.dims is None:
            return tf.shape(x)

        static_shape = x.shape.as_list()
        dynamic_shape = tf.shape(x)

        ret = []
        for i in range(len(static_shape)):
            dim = static_shape[i]
            if dim is None:
                dim = dynamic_shape[i]
            ret.append(dim)

        return ret

    #@tf.function
    def call(self, x, t, training=False, use_mask = False):
        x_old_shape =  self._infer_shape(x)

        #inferred_shape = x_old_shape
        #x = tf.reshape (x, [inferred_shape[0] * x.shape[1], x.shape[2], x.shape[3]])
        #x = tf.expand_dims (x, -1)
        x = tf.transpose (x, [0, 2, 3, 1])
        encoded_images = self.autoencoder.call_encoder (x, training=training)
    
        shape_encoded = self._infer_shape(encoded_images)
        dim = tf.reduce_prod(shape_encoded[1:])
        #encoded_images = tf.reshape(encoded_images, [-1, dim])
        #encoded_images_shape = self._infer_shape(encoded_images)
        #encoded_images = tf.reshape (encoded_images, [x_old_shape[0], x_old_shape[1], encoded_images_shape[1]])
        #encoded_images_shape = self._infer_shape(encoded_images)
        encoded_images = tf.reshape (encoded_images, [shape_encoded[0], dim])
        #print (encoded_images.shape)
        encoded_images = self.autoencoder.call_dim_reducer (encoded_images, training=training)

        t_shape = self._infer_shape(t)
        print ('t_shape: ', t_shape)
  
        t_reshaped = tf.tile(tf.expand_dims(t, axis=0), [x_old_shape[0], 1])  
        
        if use_mask:
            #if self.t_reshaped_mask is None:
            #    self.t_reshaped_mask = tf.Variable(tf.zeros_like (t_reshaped))
            #else:
            #    self.t_reshaped_mask.assign(tf.zeros_like (t_reshaped))
            #t_reshaped_mask = self.t_reshaped_mask
            t_reshaped_mask = []
            for ind_val in range(x_old_shape[0]): 
                i_val = tf.range(t_shape[0]-1)
                i_val = tf.where(i_val >=self.i_mask, i_val+1, i_val)
                i_val = tf.random.shuffle(i_val)
                ind_rm=i_val[:self.n_mask]
                if self.row_val is None:
                    self.row_val = tf.Variable(1-tf.one_hot([self.i_mask], t_shape[0]))
                else:              
                    self.row_val.assign (1-tf.one_hot([self.i_mask], t_shape[0]))
                #print ('self.row_val: ', self.row_val)
                row_val = self.row_val
                #row_val [self.i_mask].assign(0) 
                for v in range(self.n_mask):
                    row_val.assign_add (1-tf.one_hot([ind_rm[v]], t_shape[0]))# [ind_rm[v]].assign(0)
                
                t_reshaped_mask.append (row_val)
            t_reshaped_mask = tf.stack (t_reshaped_mask)
            t_reshaped_mask = tf.reshape(t_reshaped_mask, [t_shape[0]*x_old_shape[0]])
            t_reshaped = tf.reshape(t_reshaped, [t_shape[0]*x_old_shape[0], 1])
            t_reshaped = t_reshaped[t_reshaped_mask > 0]
            x_reshaped = tf.repeat (encoded_images, t_shape[0], axis=0)
            x_reshaped = tf.boolean_mask(x_reshaped, t_reshaped_mask > 0)
        else:
            t_reshaped = tf.reshape(t_reshaped, [t_shape[0]*x_old_shape[0], 1])
            x_reshaped = tf.repeat (encoded_images, t_shape[0], axis=0)
        #x_reshaped = tf.reshape(x_reshaped, [x_reshaped.shape[0], x_reshaped.shape[1]*x_reshaped.shape[2]])
        x = tf.concat([x_reshaped, t_reshaped], axis=1)
        
        z = [x]
        print ('x.shape: ', x.shape)
        print (z)
        with tf.GradientTape(persistent=True) as g:
          g.watch (x)
          for ell in range (self.num_resnet_layers):
              print ('ell: ', ell)
              #dim = tf.reduce_prod(tf.shape(x)[1:])
              #x_flattened = tf.reshape(x, [-1, dim])
              #z_flattened = tf.reshape(z[-1], [-1, dim])
              with g.stop_recording():
                  if not self.approx_jacobian:
                      jacobian_z_x = self.jacobian(z[-1], x, g) 
                  else:
                      if ell == 0:
                          jacobian_z_x = self.jacobian(x, x, g) 
                      else:
                          jacobian_z_x +=  self.jacobian(phi_curr, x, g) 
                  #print (jacobian_z_x.shape)
              phi_curr = self.call_phi (ell, x, training=training)
              delta = jacobian_z_x @  tf.expand_dims(phi_curr, -1)
              delta = tf.squeeze(delta, -1)
              #print(delta.shape)
              z.append(z[-1] +  delta)

        for ell in range (self.num_resnet_layers+1): 
            print (ell)  
            z[ell] = z[ell][:, :-1]
            z[ell] = self.autoencoder.call_dim_reconstructor (z[ell], training=training)
            z_ell_shape = self._infer_shape(z[ell])
            z[ell] = tf.reshape (z[ell], [z_ell_shape[0], shape_encoded[1], shape_encoded[2], shape_encoded[3]])
            z[ell] = self.autoencoder.call_decoder (z[ell],training=training)
            z[ell] = z[ell] [:, :, :, 0]
        
        print ('call finished')
        if use_mask:
            return z, t_reshaped_mask > 0
        else: 
            return z

import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
import shutil
def load_data(file_name = 'data/rotated_mnist/rot-mnist-3s.mat', N=500, T=16):
    X = loadmat(file_name)['X'].squeeze() # (N, 16, 784)
    print (X.shape)
    X_train = X[:N, :, :].astype(np.float32)
    X_test = X[N:, :, :].astype(np.float32)
    X_train   = X_train.reshape([N,T,28,28])
    X_test = X_test.reshape([-1,T,28,28])
    return X_train, X_test


def do_training (task_id):
    LOG_DIR = 'logs_rotating_mnist'+str(task_id)    
    if os.path.isdir(LOG_DIR):
       shutil.rmtree(LOG_DIR, ignore_errors=True)
    os.mkdir (LOG_DIR)
    trajectories_train, trajectories_test = load_data ()

    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
        tf.config.experimental_connect_to_cluster(resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All devices: ", tf.config.list_logical_devices('TPU'))
        strategy = tf.distribute.TPUStrategy(resolver)
    except:
        strategy = None
    autodiff_inimnet =  AutoDiffInImNet(dim=INNER_DIM_LATENT, represent_dim_out=DIM_LATENT, 
                num_resnet_layers=ELL_MAX,
                activation=ACTIVATION,
                num_internal_layers=NUM_INTERNAL_LAYERS,
                bias_on=True,
                mult=INFLATION_FACTOR,
                dropout=DROPOUT_VALUE,
                lr_inim = tf.keras.optimizers.schedules.ExponentialDecay (LR_INIM, 30 * trajectories_train.shape[0] // BATCH_SIZE, 0.5, staircase=True),
                weight_regularisation_alpha = 0.0,#0.00001
                n_mask=N_MASK,
                i_mask=I_EVAL
                )
    losses = []

    #images = trajectories_train[indices, :NUM_IMAGES_TOTAL, :, :]
    #x = images [:, :NUM_IMAGES_CONTEXT, :, :]
    dataset = tf.data.Dataset.from_tensor_slices((trajectories_train))
    dataset = dataset.shuffle(512).batch(BATCH_SIZE)
    iter_dataset = iter(dataset)
    if strategy is not None:
        dataset = strategy.experimental_distribute_dataset(dataset)

    dataset_test = tf.data.Dataset.from_tensor_slices((trajectories_test))
    dataset_test = dataset_test.batch(BATCH_SIZE)
    iter_dataset_test = iter(dataset_test)
    if strategy is not None:
        dataset_test = strategy.experimental_distribute_dataset(dataset_test)

    loss_train_x = []
    loss_train_y = []
    loss_valid_x = []
    loss_valid_y = []
    loss_test_x = []
    loss_test_y = []
    losses_per_epoch = []

    @tf.function
    def distribute_train_step(data, async_exec=True):
        def replica_fn (d):
            d_im = d[:, :NUM_IMAGES_TOTAL, :, :]
            d_x = d[:, :NUM_IMAGES_CONTEXT, :, :]
            print ('d_x.shape: ', d_x.shape)
            print ('d_im.shape: ', d_im.shape)
            return autodiff_inimnet.optimise (d_x, d_im, t, tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE))
              
        if strategy is not None:         
            if async_exec:
                per_replica_result = [strategy.run(replica_fn, args=(data,))]
                print ('per_replica_result: ', per_replica_result)
                return strategy.gather(per_replica_result, axis=0)
            else:
                results = replica_fn (strategy.gather(data, axis=0))
                print (results)
                return results
        else:
            results = replica_fn (data)
            return [results]

    @tf.function
    def distribute_test_step (data):
        def replica_fn_test (d):
            d_im = d[:, :, :, :]
            d_x = d_im[:, :NUM_IMAGES_CONTEXT, :, :]
            print ('d_x.shape: ', d_x.shape)
            print ('d_im.shape: ', d_im.shape)
            return d_im, autodiff_inimnet (d_x, t_eval, training=False)
        if strategy is not None: 
            return strategy.gather(strategy.run(replica_fn_test, args=(data,)), axis=0)
        else:
            return replica_fn_test(data)
          
    @tf.function
    def distribute_test_step_loss (data, async_exec=True):
        def replica_fn_test_loss (d):
            d_im = d[:, I_EVAL:I_EVAL+1, :, :]
            d_x = d[:, :NUM_IMAGES_CONTEXT, :, :]
            print ('d_x.shape: ', d_x.shape)
            print ('d_im.shape: ', d_im.shape)
            result = autodiff_inimnet (d_x, t_eval[I_EVAL:I_EVAL+1], training=False)
            #d_im = tf.reshape (d_im, [d_im.shape[0] * d_im.shape[1], d_im.shape[2], d_im.shape[3]])
            result = [tf.reshape (r, [ tf.shape(d_im)[0], d_im.shape[1], r.shape[1], r.shape[2]]) for r in result]
                
            return [tf.math.reduce_mean(tf.square(r - d_im), axis=[2, 3]) for r in result]
        if strategy is not None: 
            if async_exec:
                per_replica_result = strategy.run(replica_fn_test_loss, args=(data,))
                results = [strategy.gather(single_result, axis=0) for single_result in per_replica_result]
                return [tf.reduce_sum(r, axis=0) for r in results]
            else:
                results = replica_fn_test_loss (strategy.gather(data, axis=0))
                return [tf.reduce_sum(r, axis=0) for r in results]
        else:
            results = replica_fn_test_loss (data)
            return [tf.reduce_sum(r, axis=0) for r in results]

    best_value = 1e9
    for epoch in range(NUM_EPOCHS):
        running_losses = []
        for x in dataset:
            curr_loss = distribute_train_step(x)
            losses.extend(curr_loss)
            running_losses.extend(curr_loss)
            x_last = x
        print ('training finished')
        losses_per_epoch.append(np.array(tf.reduce_mean(running_losses)))
           
        if epoch % REPORT_INTERVAL_EPOCHS == 0:
            print (epoch)
            curr_loss_train = None
            curr_loss_test = None
            curr_loss_valid = None
             
            for x_train in dataset:
                if curr_loss_train is None:
                    curr_loss_train = np.array(distribute_test_step_loss (x_train))
                else:
                    curr_loss_train += np.array(distribute_test_step_loss (x_train))
            curr_loss_train /= trajectories_train.shape[0]
           
            for x_test in dataset_test:          
                curr_loss_value = distribute_test_step_loss (x_test)
                curr_loss_value = np.array(curr_loss_value)
                if curr_loss_test is None:
                    curr_loss_test = curr_loss_value
                    x_last_test = x_test
                else:
                    if not len(curr_loss_value.shape)  == len(curr_loss_test.shape):
                        curr_loss_value = np.array(distribute_test_step_loss (x_test, False))
                
                    curr_loss_test += curr_loss_value
                
            curr_loss_test /= trajectories_test.shape[0]

            loss_train_x.append(epoch)
            loss_train_y.append(curr_loss_train)
            loss_test_x.append(epoch)
            loss_test_y.append(curr_loss_test)

            for layer_id in range (1, curr_loss_train.shape[0]):
                plt.plot(loss_train_x, np.mean(np.array(loss_train_y)[:, layer_id, :], axis=1), label='train_loss (l' + str(layer_id)+')')
                plt.plot(loss_test_x, np.mean(np.array(loss_test_y)[:, layer_id, :], axis=1), label='test_loss (l' + str(layer_id)+')')        
                #plt.plot(loss_valid_x, np.mean(np.array(loss_valid_y)[:, layer_id, :], axis=1), label='valid_loss(l' + str(layer_id)+')')
                plt.legend()

            plt.savefig(LOG_DIR+'/'+str('losses_history_'+str(epoch)+'.png'))

            for layer_id in range (1, curr_loss_train.shape[0]):
                plt.plot(np.array(loss_train_y)[-1, layer_id, :], label='train_loss (l' + str(layer_id)+', '+str(np.mean(np.array(loss_train_y)[-1, layer_id, :]))+')')
                plt.plot(np.array(loss_test_y)[-1, layer_id, :], label='test_loss (l' + str(layer_id)+', '+str(np.mean(np.array(loss_test_y)[-1, layer_id, :]))+')')        
                #plt.plot(np.array(loss_valid_y)[-1, layer_id, :], label='valid_loss(l' + str(layer_id)+', '+str(np.mean(np.array(loss_valid_y)[-1, layer_id, :]))+')')
                plt.legend() 

            plt.savefig(LOG_DIR+'/'+str('losses_'+str(epoch)+'.png'))
            
            print ('Loss (frame_id), training:')
            for layer_id in range (1, curr_loss_train.shape[0]):
                print  (np.array(loss_train_y)[-1, layer_id, :])
                print ('Loss (frame_id), testing:')
            for layer_id in range (1, curr_loss_train.shape[0]):
                print  (np.array(loss_test_y)[-1, layer_id, :])
            if np. mean (np.array(loss_train_y)[-1, layer_id, :]) < best_value:
                best_value = np. mean (np.array(loss_train_y)[-1, layer_id, :])           
            print('current best value: ', best_value)
            print ('test_')        
            #plt.plot (losses_per_epoch)
            #plt.show()
            with open(LOG_DIR+'/log_train'+'.txt', 'a+') as f:
                 print('epoch: ', epoch, '; ', np.array(loss_train_y)[-1, 1:, :].flatten(), file=f)
            with open(LOG_DIR+'/log_test'+'.txt', 'a+') as f:
                 print('epoch: ', epoch, '; ', np.array(loss_test_y)[-1, 1:, :].flatten(), file=f)
            
            d_im, pred_train  =  distribute_test_step (x_last)
            print ('Data sample (training)')
            print ('GT:')
            fig, axes = plt.subplots(1, d_im.shape[1])
            
            print ('axes.shape[0]: ', axes.shape[0])
            for i in  range(1):
                for j in  range(d_im.shape[1]):
                    axes[j].get_xaxis().set_visible(False)
                    axes[j].get_yaxis().set_visible(False)
                    axes[j].imshow(d_im[0, i*d_im.shape[1]+j, :, :])
            plt.savefig(LOG_DIR+'/data_sample_gt_training_'+str(epoch)+'.png')

            print ('Prediction: ')
            for layer_id in range(len(pred_train)): 
                print ('Layer ' + (str(layer_id)))
                fig, axes = plt.subplots(1, d_im.shape[1])
                for i in  range(1):
                     for j in  range(d_im.shape[1]):
                         axes[j].get_xaxis().set_visible(False)
                         axes[j].get_yaxis().set_visible(False)
                         axes[j].imshow(pred_train[layer_id][i*d_im.shape[1]+j, :, :])
                plt.savefig(LOG_DIR+'/data_sample_pred_training_'+str(epoch)+'_'+str(layer_id)+'.png')

            d_im_test, pred_test  =  distribute_test_step (x_last_test)
            #print ('Data sample (testing)')
            #print ('GT:')
            fig, axes = plt.subplots(1, d_im_test.shape[1])
            print ('axes.shape[0]: ', d_im_test.shape[1])
            for i in  range(1):
                for j in  range(d_im_test.shape[1]):
                    axes[j].get_xaxis().set_visible(False)
                    axes[j].get_yaxis().set_visible(False)
                    axes[j].imshow(d_im_test[0, i*d_im_test.shape[1]+j, :, :])
            plt.savefig(LOG_DIR+'/data_sample_gt_testing_'+str(epoch)+'.png')

            #print ('Prediction: ')
            for layer_id in range(len(pred_test)): 
                print ('Layer ' + (str(layer_id)))
                fig, axes = plt.subplots(1, d_im_test.shape[1])
                for i in  range(1):
                    for j in  range(d_im_test.shape[1]):
                       axes[j].get_xaxis().set_visible(False)
                       axes[j].get_yaxis().set_visible(False)
                       axes[j].imshow(pred_test[layer_id][i*d_im_test.shape[1]+j, :, :])
                plt.savefig(LOG_DIR+'/data_sample_pred_testing_'+str(epoch)+'_'+str(layer_id)+'.png')
      
      
    #plt.plot(np.array(losses))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Rotated MNIST')
    parser.add_argument('--task_id', 
        type=int, 
        default=0, 
        help='GPU ID.') 
    parser.add_argument('--seed',
        type=int,
        default=-1,
        help='GPU ID.')   
    args = parser.parse_args()
    
    if args.seed==-1:
        print('WARNING: No random seed was specified.')
    else:
        print('Setting the random seed to {:}'.format(args.seed))
        random.seed(args.seed)
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
    
    do_training (args.task_id)
