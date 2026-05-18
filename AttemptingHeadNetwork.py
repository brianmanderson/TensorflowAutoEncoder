import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print("GPUs found:", gpus)
# tf.keras.backend.set_floatx('float32') 
# tf.keras.backend.set_epsilon(1e-4) # Adjust epsilon for float16
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)
# tf.config.optimizer.set_jit(False)
from tensorflow.keras import layers, models
import time
import random
import gc
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import SimpleITK as sitk
import pandas as pd

save_data_base_path = return_file_path()
tb_path = os.path.join(save_data_base_path, 'Data', 'Test', 'Tensorboard')

print(f"Saving to {save_data_base_path}")
HP_REDUCTION_FACTOR = hp.HParam('reduction_factor')
HP_RESNET = hp.HParam('ResNet')
HP_ATTENTIONNET = hp.HParam('AttentionNet')
HP_NUM_UNITS = hp.HParam('connections_per_layer')
HP_RUN_DENSE = hp.HParam('run_dense_layers')
HP_NUM_LAYERS = hp.HParam('number_of_dense_layers')
HP_IS_3D = hp.HParam('Is3D')
HP_DROPOUT = hp.HParam('dropout')
HP_OPTIMIZER = hp.HParam('optimizer')
HP_LR = hp.HParam('learning_rate')
HP_MIN_DELTA = hp.HParam("min_delta")
HP_NUM_CONV_LAYERS = hp.HParam("num_conv_layers")
HP_CONV_PER_LAYER = hp.HParam("conv_per_layer")
HP_INITIAL_FEATURES = hp.HParam("initial_features")
HP_MAX_FEATURES = hp.HParam("max_features")
HP_FEATURES_DOUBLE = hp.HParam("conv_double")
HP_DENSE_DOUBLE = hp.HParam("dense_double")
HP_MODEL_TYPE = hp.HParam("model_type")
HP_RELOAD = hp.HParam("reload_index")
HP_ALL_TRAINABLE = hp.HParam("all_trainable")
HP_OUTPUT_ACTIVATION = hp.HParam("output_activation")
HP_AUTO_ENCODE = hp.HParam("auto_encode")
HP_IMAGE_REDUCTION = hp.HParam('train_image_reduction')
HP_OUTPUT_NUMBER = hp.HParam("output_number")
HP_COMPRESSED_SIZE = hp.HParam("compressed_size")
HP_IS_FLATTENED = hp.HParam("is_flattened")
HP_LOSS = hp.HParam("loss")
HP_BATCH = hp.HParam("Batch")
HP_PATCH_Z = hp.HParam("patch_z")
HP_PATCH_X = hp.HParam("patch_x")
HP_PATCH_Y = hp.HParam("patch_y")
HP_COMPRESSION = hp.HParam("Compression")
HP_TRANSFORMER = hp.HParam("transformer")
HP_SKIP = hp.HParam("SkipNumber")
METRIC_MSE = 'MeanSquaredError'

# METRIC_MSE = 'MeanSquaredError'

log_path = os.path.join(save_data_base_path, "Data", "Logs", "hparam_tuning")


def return_hparam(df, ind):
    hparams = {
        HP_SKIP: int(df["skip_number"][ind]),
        HP_LR: df["learning_rate"][ind],
        HP_RUN_DENSE: bool(df["run_dense_layers"][ind]),
        HP_IS_3D: bool(df["Is3D"][ind]),
        HP_TRANSFORMER: bool(df["transformer"][ind]),
        HP_RESNET: bool(df["ResNet"][ind]),
        HP_ATTENTIONNET: bool(df["AttentionNet"][ind]),
        HP_NUM_LAYERS: int(df["number_of_dense_layers"][ind]),
        HP_DROPOUT: df["Dropout"][ind],
        HP_PATCH_Z: int(df["patch_z"][ind]),
        HP_PATCH_X: int(df["patch_x"][ind]),
        HP_PATCH_Y: int(df["patch_y"][ind]),
        HP_COMPRESSION: int(df['Compression'][ind]),
        HP_IMAGE_REDUCTION: int(df['train_image_reduction'][ind]),
        HP_OPTIMIZER: df["optimizer"][ind],
        HP_REDUCTION_FACTOR: df["reduction_factor"][ind],
        HP_MIN_DELTA: df["min_delta"][ind],
        HP_BATCH: int(df['Batch'][ind]),
        HP_CONV_PER_LAYER: int(df['conv_per_layer'][ind]),
        HP_NUM_CONV_LAYERS: int(df['num_conv_layers'][ind]),
        HP_INITIAL_FEATURES: int(df["initial_features"][ind]),
        HP_MAX_FEATURES: int(df["max_features"][ind]),
        HP_FEATURES_DOUBLE: df["conv_double"][ind],
        HP_DENSE_DOUBLE: df["dense_double"][ind],
        HP_MODEL_TYPE: df["model_type"][ind],
        HP_RELOAD: int(df["reload_index"][ind]),
        HP_ALL_TRAINABLE: bool(df["all_trainable"][ind]),
        HP_AUTO_ENCODE: bool(df["auto_encode"][ind]),
        HP_OUTPUT_ACTIVATION: df["output_activation"][ind],
        HP_OUTPUT_NUMBER: int(df["output_number"][ind]),
        HP_COMPRESSED_SIZE: int(df["compressed_size"][ind]),
        HP_IS_FLATTENED: bool(df["is_flattened"][ind]),
        'folder_path_headers': str(df['folders'][ind]),
        HP_LOSS: str(df["loss"][ind])
    }
    return hparams


class ReconstructVolumePatchesLayer(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        """
        Args:
            patch_size: Tuple of three ints (pD, pH, pW).
        """
        super().__init__(**kwargs)
        self.pD, self.pH, self.pW = patch_size

    def call(self, inputs):
        # inputs: [patches, original]
        patches, original = inputs

        # dynamic original shape
        orig_shape = tf.shape(original)   # [B, D, H, W, C]
        B, D, H, W, C = orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3], orig_shape[4]

        self.B, self.D, self.H, self.W, self.C = B, D, H, W, C
        # compute total padding that was applied
        pad_d = (self.pD - D % self.pD) % self.pD
        pad_h = (self.pH - H % self.pH) % self.pH
        pad_w = (self.pW - W % self.pW) % self.pW

        # how much went on the “front” of each axis
        pad_d_front = tf.math.floordiv(pad_d + 1, 2)
        pad_h_front = tf.math.floordiv(pad_h + 1, 2)
        pad_w_front = tf.math.floordiv(pad_w + 1, 2)

        # padded dims
        Dp = D + pad_d
        Hp = H + pad_h
        Wp = W + pad_w

        # how many patches along each axis
        grid_d = tf.math.floordiv(Dp, self.pD)
        grid_h = tf.math.floordiv(Hp, self.pH)
        grid_w = tf.math.floordiv(Wp, self.pW)

        # reshape patches back into the full padded volume
        # patches: [B, num_patches, pD, pH, pW, C] where 
        # num_patches = grid_d * grid_h * grid_w
        x = tf.reshape(patches,
                       [B, grid_d, grid_h, grid_w,
                        self.pD, self.pH, self.pW, C])
        # invert the transpose from extraction
        x = tf.transpose(x, [0, 1, 4, 2, 5, 3, 6, 7])
        # collapse back into [B, Dp, Hp, Wp, C]
        x = tf.reshape(x, [B, grid_d * self.pD,
                               grid_h * self.pH,
                               grid_w * self.pW,
                               C])

        # now slice away the padding to recover [B, D, H, W, C]
        begin = tf.stack([0, pad_d_front, pad_h_front, pad_w_front, 0])
        size  = tf.stack([B, D,          H,           W,           C])
        return tf.slice(x, begin, size)

    def compute_output_shape(self, input_shape):
        # input_shape is a tuple: (patches_shape, original_shape)
        _, orig_shape = input_shape
        # batch, D, H, W, C = orig_shape
        # pD, pH, pW = self.pD, self.pH, self.pW

        # # total pad needed so each dim % patch == 0
        # pad_d = (pD - D % pD) % pD
        # pad_h = (pH - H % pH) % pH
        # pad_w = (pW - W % pW) % pW

        # # how much went on the “front” of each axis
        # pad_d_front = (pad_d + 1) // 2
        # pad_h_front = (pad_h + 1) // 2
        # pad_w_front = (pad_w + 1) // 2

        # # intermediate padded dims
        # Dp = D + pad_d
        # Hp = H + pad_h
        # Wp = W + pad_w

        # after tf.slice we end up back at the original volume size:
        return orig_shape


class ExtractVolumePatchesLayer(tf.keras.layers.Layer):
    def __init__(self, patch_size, **kwargs):
        """
        Args:
            patch_size: Tuple or list of three ints (patch_depth, patch_height, patch_width)
        """
        super().__init__(**kwargs)
        self.pD, self.pH, self.pW = patch_size

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, images):
        # images shape: [B, D, H, W, C]
        shape = tf.shape(images)
        B, D, H, W, C = shape[0], shape[1], shape[2], shape[3], shape[4]

        # compute how much to pad so each dim is divisible by patch size
        pad_d = (self.pD - D % self.pD) % self.pD
        pad_h = (self.pH - H % self.pH) % self.pH
        pad_w = (self.pW - W % self.pW) % self.pW

        # pad on the “end” of each spatial dimension
        pad_d_front = (pad_d + 1) // 2
        pad_d_back  =  pad_d     // 2
        pad_h_front = (pad_h + 1) // 2
        pad_h_back  =  pad_h     // 2
        pad_w_front = (pad_w + 1) // 2
        pad_w_back  =  pad_w     // 2

        # construct the paddings tensor: [[0,0], [d_front,d_back], [h_front,h_back], [w_front,w_back], [0,0]]
        paddings = tf.stack([
            [0, 0],
            [pad_d_front, pad_d_back],
            [pad_h_front, pad_h_back],
            [pad_w_front, pad_w_back],
            [0, 0],
        ])
        images_padded = tf.pad(images, paddings)

        # new padded sizes
        Dp = D + pad_d
        Hp = H + pad_h
        Wp = W + pad_w

        # reshape into the grid of patches
        # [B, Dp/self.pD, self.pD, Hp/self.pH, self.pH, Wp/self.pW, self.pW, C]
        reshaped = tf.reshape(
            images_padded,
            (B,
             Dp // self.pD, self.pD,
             Hp // self.pH, self.pH,
             Wp // self.pW, self.pW,
             C)
        )

        # move the patch‑grid axes up front: [B, Dg, Hg, Wg, pD, pH, pW, C]
        transposed = tf.transpose(reshaped, [0, 1, 3, 5, 2, 4, 6, 7])

        # collapse the grid into one “num_patches” dimension
        grid = tf.shape(transposed)
        num_patches = grid[1] * grid[2] * grid[3]
        output = tf.reshape(transposed, [B, num_patches, self.pD, self.pH, self.pW, C])

        return output

    def compute_output_shape(self, input_shape):
        # input_shape: (batch, D, H, W, C)
        batch, D, H, W, C = input_shape

        # static ceil division for number of patches per dimension
        def ceil_div(x, p):
            return None if x is None else (x + p - 1) // p

        out_d = ceil_div(D, self.pD)
        out_h = ceil_div(H, self.pH)
        out_w = ceil_div(W, self.pW)

        num_patches = None
        if out_d is not None and out_h is not None and out_w is not None:
            num_patches = out_d * out_h * out_w

        return (batch, num_patches, self.pD, self.pH, self.pW, C)


class FractionDenseLayer(layers.Layer):
    def __init__(self, patch_size, channels, reduction_factor, activation=None, **kwargs):
        """
        Args:
            patch_size: Tuple of three ints (patch_depth, patch_height, patch_width)
            channels: Number of channels in the patch.
            reduction_factor: Factor by which to reduce the total number of features.
            activation: Optional activation function.
        """
        super(FractionDenseLayer, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.channels = channels
        self.reduction_factor = reduction_factor
        self.activation = tf.keras.activations.get(activation)

        # Compute number of units using the formula:
        # units = (patch_depth * patch_height * patch_width * channels) / reduction_factor
        self.units = int(np.prod(patch_size) * channels / reduction_factor)

    def build(self, input_shape):
        # The input is expected to be flattened so that its last dimension
        # equals patch_size[0] * patch_size[1] * patch_size[2] * channels.
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(
            name="kernel",
            shape=(input_dim, self.units),
            initializer="glorot_uniform",
            trainable=True
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )
        super(FractionDenseLayer, self).build(input_shape)

    def call(self, inputs):
        # Standard dense layer computation: inputs @ kernel + bias.
        output = tf.matmul(inputs, self.kernel) + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)


class ExpandFlattenExceptFirstTwo(layers.Layer):
    def __init__(self, target_shape, **kwargs):
        """
        Args:
            target_shape: Tuple or list specifying the original shape of the dimensions
                          that were flattened (e.g., (16, 32, 32, 4)).
        """
        super(ExpandFlattenExceptFirstTwo, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def call(self, inputs):
        """
        Args:
            inputs: A tensor of shape (batch, T, flat_dim) where flat_dim = product(target_shape).
        Returns:
            A tensor reshaped to (batch, T) + target_shape.
        """
        # Dynamically get the first two dimensions.
        batch = tf.shape(inputs)[0]
        T = tf.shape(inputs)[1]
        # Reshape to (batch, T, *target_shape)
        return tf.reshape(inputs, tf.concat([[batch, T], tf.constant(self.target_shape, dtype=tf.int32)], axis=0))

    def compute_output_shape(self, input_shape):
        # The first two dimensions are preserved; then we append the static target shape.
        return (input_shape[0], input_shape[1]) + self.target_shape


class ReshapeExceptFirstTwo(layers.Layer):
    def __init__(self, **kwargs):
        super(ReshapeExceptFirstTwo, self).__init__(**kwargs)

    def call(self, inputs):
        # Get dynamic shape of the input tensor.
        input_shape = tf.shape(inputs)
        batch = input_shape[0]
        second = input_shape[1]
        # Compute the product of all dimensions beyond the first two.
        remaining = tf.reduce_prod(input_shape[2:])
        # Reshape to [batch, second, product_of_remaining_dimensions]
        return tf.reshape(inputs, [batch, second, remaining])

    def compute_output_shape(self, input_shape):
        # If static shape information is available, compute the product of the dimensions
        # after the first two. If any of these dimensions is None, the output will be partially undefined.
        if None in input_shape[2:]:
            remaining = None
        else:
            remaining = 1
            for dim in input_shape[2:]:
                remaining *= dim
        return (input_shape[0], input_shape[1], remaining)


def return_model(hparams) -> models.Model:
    input_channels = 1
    if hparams[HP_AUTO_ENCODE]:
        patch_z = hparams[HP_PATCH_Z]
        patch_x = hparams[HP_PATCH_X]
        patch_y = hparams[HP_PATCH_Y]
        initial_patch_size = (patch_z, patch_x, patch_y)
        max_compression = hparams[HP_COMPRESSION]
    if hparams[HP_IS_3D]:
        conv = layers.Conv3D
        conv_kernel = (3, 3, 3)
        conv_kernel_ones = (1, 1, 1)
        max_pool = layers.MaxPool3D
        pool_kernel = (2, 2, 2)
        up_sampling = layers.UpSampling3D
        # inputs = x = layers.Input(shape=[16, 64, 64, input_channels])
        inputs = x = layers.Input(shape=[None, None, None, input_channels])
    else:
        conv = layers.Conv2D
        conv_kernel = (3, 3)
        conv_kernel_ones = (1, 1)
        max_pool = layers.MaxPool2D
        pool_kernel = (2, 2)
        up_sampling = layers.UpSampling2D
        inputs = x = layers.Input(shape=[None, None, input_channels])

    encoding_list = []

    # og_size = x.shape
    # x = ExtractVolumePatches(initial_patch_size, trainable=False, name='First_Extract')(x)
    # x = ReconstructVolume(og_size, initial_patch_size, name='First_Reconstruct')(x)
    # model = tf.keras.Model(inputs, x)
    # return model
    encoding_shape_list = []
    filters_list = []
    initial_conv = hparams[HP_INITIAL_FEATURES]
    max_filters = hparams[HP_MAX_FEATURES]
    skip = hparams[HP_SKIP]
    for _ in range(hparams[HP_NUM_CONV_LAYERS]):
        # x = layers.Multiply()([x, mask_image])
        filters_list.append(initial_conv)
        skip -= 1
        for __ in range(hparams[HP_CONV_PER_LAYER]):
            x = conv(initial_conv, conv_kernel, padding="same", name=f"Encoding_Conv_{_}_{__}")(x)
            if __ == 0:
                input_conv = x
            if hparams[HP_RESNET] and __ == hparams[HP_CONV_PER_LAYER] - 1:
                x = layers.Add()([input_conv, x])
                encoding_list.append(x)
                x = layers.BatchNormalization(name=f"Encoding_BN_{_}_{__}")(x)
                x = layers.Activation("elu", name=f"Encoding_Activation_{_}_{__}")(x)
            else:
                x = layers.BatchNormalization(name=f"Encoding_BN_{_}_{__}")(x)
                x = layers.Activation("elu", name=f"Encoding_Activation_{_}_{__}")(x)
                if __ == hparams[HP_CONV_PER_LAYER] - 1 and skip < 0:
                    if not hparams[HP_TRANSFORMER]:
                        encoding_list.append(x)
                    else:
                        """
                        Branching here, we have the last convolution here as 'x', the rest if to bridge the gap
                        """
                        x = conv(1, conv_kernel, padding="same", name=f"Encoding_Conv_Single{_}_{__}")(x)
                        x = layers.BatchNormalization(name=f"Encoding_BN_Single{_}_{__}")(x)
                        x = layers.Activation("elu", name=f"Encoding_Activation_Single_{_}_{__}")(x)
                        encoding_x = x
                        """
                        Maintain the initial, full shape
                        """
                        encoding_shape_list.append(x)
                        """
                        Convert our large volume into small chunks
                        """
                        x = ExtractVolumePatchesLayer(initial_patch_size, name=f'Extract_{_}')(x)
                        """
                        Then the post-extraction shape. We will need to do this in two parts after all
                        """
                        encoding_shape_list.append(x)
                        # initial_patch_size = tuple(i//2 for i in initial_patch_size)
                        # x = layers.Reshape((x.shape[1], -1))(x)
                        x = ReshapeExceptFirstTwo()(x)
                        if hparams[HP_DROPOUT] != 0:
                            x = layers.Dropout(hparams[HP_DROPOUT])(x)
                        x = layers.Dense(int(x.shape[-1] // max_compression), activation='elu',
                                         name=f"Encoding_Dense_{_}_{__}")(x)
                        if hparams[HP_DROPOUT] != 0:
                            x = layers.Dropout(hparams[HP_DROPOUT])(x)
                        encoding_list.append(x)
                        x = encoding_x
        initial_conv = int(initial_conv * hparams[HP_FEATURES_DOUBLE])
        if initial_conv > max_filters:
            initial_conv = max_filters
        x = max_pool(pool_kernel)(x)
    """
    Make a bottle neck
    """
    if hparams[HP_RUN_DENSE]:
        encoding_tensor = x
        """
        Convert our large volume into small chunks
        """
        x = ExtractVolumePatchesLayer(initial_patch_size, name=f'Extract_Base')(x)
        before_flattened = x
        x = ReshapeExceptFirstTwo()(x)
        original_dense = x.shape[-1]
        if hparams[HP_DROPOUT] != 0:
            x = layers.Dropout(hparams[HP_DROPOUT])(x)
        # x = FractionDenseLayer(patch_size=initial_patch_size, channels=initial_conv,
        #                        reduction_factor=max_compression, activation='elu',
        #                        name=f"Encoding_Dense_Base")(x)
        x = layers.Dense(int(original_dense // max_compression), activation='elu', name=f"Encoding_Dense_Base")(x)
        """
        Bring it back to normal size
        """
        x = layers.Dense(original_dense, activation='elu', name="Decoding_Dense_Base")(x)
        x = ExpandFlattenExceptFirstTwo(before_flattened.shape[2:])(x)
        x = ReconstructVolumePatchesLayer(patch_size=initial_patch_size, name='Base_Reconstruct')([x, encoding_tensor])
    else:
        for __ in range(hparams[HP_CONV_PER_LAYER]):
            x = conv(initial_conv, conv_kernel, padding="same", name=f"Bottom_Conv_{_}_{__}")(x)
            if __ == 0:
                input_conv = x
            if hparams[HP_RESNET] and __ == hparams[HP_CONV_PER_LAYER] - 1:
                x = layers.Add()([input_conv, x])
                x = layers.BatchNormalization(name=f"Bottom_BN_{_}_{__}")(x)
                x = layers.Activation("elu", name=f"Bottom_Activation_{_}_{__}")(x)
            else:
                x = layers.BatchNormalization(name=f"Bottom_BN_{_}_{__}")(x)
                x = layers.Activation("elu", name=f"Bottom_Activation_{_}_{__}")(x)
    for _ in range(hparams[HP_NUM_CONV_LAYERS]):
        x = up_sampling(pool_kernel)(x)
        if encoding_list:
            prev_side = encoding_list.pop()
            if hparams[HP_TRANSFORMER]:
                chunked_encode = encoding_shape_list.pop()
                prev_side = layers.Dense(int(tf.reduce_prod(chunked_encode.shape[2:])), activation='elu',
                                        name=f"Decoding_Dense_{_}")(prev_side)
                prev_side = ExpandFlattenExceptFirstTwo(chunked_encode.shape[2:])(prev_side)
                # prev_side = layers.Reshape(chunked_encode_shape[1:])(prev_side)
                encode_side = encoding_shape_list.pop()
                prev_side = ReconstructVolumePatchesLayer(patch_size=initial_patch_size, name=f'Decoding_Reconstruct_{_}')(
                    [prev_side, encode_side])
                # prev_side = ReconstructVolume(encode_shape, initial_patch_size, name=f'Decoding_Reconstruct_{_}')(prev_side)
            initial_conv = filters_list.pop()
            if hparams[HP_ATTENTIONNET]:
                """
                Need to create an attention gate
                """
                inter_channels = initial_conv // 2
                decoder_x = conv(inter_channels, conv_kernel_ones, padding='same', name=f"AG_Decoder_{_}")(x)
                decoder_x = layers.BatchNormalization()(decoder_x)
                encoder_x = conv(inter_channels, conv_kernel_ones, padding='same', name=f"AG_Encoder_{_}")(prev_side)
                encoder_x = layers.BatchNormalization()(encoder_x)
                added = layers.Add()([decoder_x, encoder_x])
                activated = layers.Activation("relu", name=f"AG_Activation_{_}")(added)
                activated = layers.BatchNormalization()(activated)
                focused = conv(1, conv_kernel_ones, padding='same', name=f"AG_added_{_}")(activated)
                focused = layers.BatchNormalization()(focused)
                alpha = layers.Activation('sigmoid')(focused)
                prev_side = layers.Multiply()([alpha, prev_side])
            x = layers.Concatenate()([x, prev_side])
        for __ in range(hparams[HP_CONV_PER_LAYER]):
            x = conv(initial_conv, conv_kernel, padding="same", name=f"Decoding_Conv_{_}_{__}")(x)
            if __ == 0:
                input_conv = x
            if hparams[HP_RESNET] and __ == hparams[HP_CONV_PER_LAYER] - 1:
                x = layers.Add()([input_conv, x])
                x = layers.BatchNormalization(name=f"Decoding_BN_{_}_{__}")(x)
                x = layers.Activation("elu", name=f"Decoding_Activation_{_}_{__}")(x)
            else:
                x = layers.BatchNormalization(name=f"Decoding_BN_{_}_{__}")(x)
                x = layers.Activation("elu", name=f"Decoding_Activation_{_}_{__}")(x)

    x = conv(hparams[HP_OUTPUT_NUMBER], conv_kernel, padding="same",
             activation=hparams[HP_OUTPUT_ACTIVATION], name="EndConv")(x)
    model = tf.keras.Model(inputs, x)
    return model


def return_hparams(session_id):
    excel_file = os.path.join(save_data_base_path, "HyperparametersNew.xlsx")
    df = pd.read_excel(excel_file)
    indexes = list(df.index)
    for ind in indexes:
        session_num = str(df['session_number'][ind])
        if session_num != session_id:
            continue
        hparams = return_hparam(df, ind)
        return hparams


def train_model(tensorboard_output, session_num, hparams, train_generator, validation_generator, model: tf.keras.Model):
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_output, profile_batch=0,
                                                 write_graph=True)
    hp_callback = hp.KerasCallback(tensorboard_output, hparams=hparams, trial_id='Trial_ID:{}'.format(session_num))
    checkpoint_path = os.path.join(tensorboard_output, "checkpoint", f"Session_{session_num}/cp.weights.h5")

    # Create a callback that saves the model's weights
    monitor_save = "val_loss"  # "val_"
    patience = 15
    reduce_lr_patience = 7
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     monitor=monitor_save,
                                                     verbose=1)
    early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor=monitor_save, min_delta=hparams[HP_MIN_DELTA],
                                                     patience=patience, start_from_epoch=10)
    reduce_on_plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_save, factor=0.1,
                                                             patience=reduce_lr_patience,
                                                             min_delta=hparams[HP_MIN_DELTA],
                                                             cooldown=reduce_lr_patience)
    callbacks = [tensorboard, cp_callback, hp_callback, early_stop_cb, reduce_on_plateau]  # reduce_on_plateau
    if 'Mounting' in tensorboard_output:
        tf.keras.utils.plot_model(model, to_file=os.path.join(tensorboard_output, f"Model_{session_num}.png"),
                                  show_shapes=True)
    optimizer_name = hparams[HP_OPTIMIZER]
    learning_rate = hparams[HP_LR]
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError("unexpected optimizer name: %r" % (optimizer_name,))
    reload_index = hparams[HP_RELOAD]
    if reload_index >= 0:
        local_tb_path = os.path.join(tb_path, f"{reload_index}")
        load_path = os.path.join(local_tb_path, "checkpoint", f"Session_{reload_index}/cp.weights.h5")
        # print(help(model.load_weights))
        model.load_weights(load_path, skip_mismatch=True)
        if not hparams[HP_ALL_TRAINABLE]:
            """
            If everything is not trainable, freeze all of the encoding parts
            """
            for layer in model.layers:
                if layer.name.find("Encoding") == 0:
                    print(f"Freezing {layer.name}")
                    layer.trainable = False
    if hparams[HP_AUTO_ENCODE] and hparams[HP_RUN_DENSE] and hparams[HP_OUTPUT_NUMBER] == 1:
        metrics = [tf.keras.metrics.MeanSquaredError(), ]
        loss = tf.keras.losses.MeanSquaredError()
    else:
        loss = tf.keras.losses.CategoricalCrossentropy()
        if hparams[HP_LOSS] == 'WeightedCategoricalCrossentropy':
            loss = WeightedCategoricalCrossEntropy([0.5, 5.0])
        metrics = [tf.keras.metrics.CategoricalCrossentropy(), MeanDSC(num_classes=2)]

    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    # x, y = next(iter(train_generator.data_set))
    # train_dataset = tf.data.Dataset.from_generator(lambda: iter(train_generator.data_set),
    #                                                output_types=(tf.float32, tf.float32),
    #                                                output_shapes=(tf.TensorShape([batch, None, None, None, 1]),
    #                                                               tf.TensorShape([batch, None, None, None, 2])))
    # validation_dataset = tf.data.Dataset.from_generator(lambda: iter(train_generator.data_set),
    #                                                     output_types=(tf.float32, tf.float32),
    #                                                     output_shapes=(tf.TensorShape([batch, None, None, None, 1]),
    #                                                                    tf.TensorShape([batch, None, None, None, 2])))
    model.fit(train_generator.data_set, epochs=1000, callbacks=callbacks, steps_per_epoch=len(train_generator),
              validation_steps=len(validation_generator), validation_data=validation_generator.data_set,
              validation_freq=1)


def update_model_layer_names(session_id):
    local_tb_path = os.path.join(tb_path, f"{session_id}")
    checkpoint_path = os.path.join(local_tb_path, "checkpoint", f"Session_{session_id}/cp.keras")
    hparams = return_hparams(session_id)
    model = return_model(hparams)
    model.load_weights(checkpoint_path, by_name=False)
    model.save_weights(checkpoint_path.replace(".keras", ".ckpt"))


def load_model(session_id, by_name=False):
    local_tb_path = os.path.join(tb_path, f"{session_id}")
    checkpoint_path = os.path.join(local_tb_path, "checkpoint", f"Session_{session_id}/cp.weights.h5")
    hparams = return_hparams(session_id)
    model = return_model(hparams)
    model.load_weights(checkpoint_path)
    return model


def evaluate_model(session_id):
    model = load_model(session_id, by_name=False)
    # save_model(session_id)
    hparams = return_hparams(session_id)
    tf.keras.utils.plot_model(model, to_file=os.path.join(tb_path, f"{session_id}",
                                                          f"Model_{session_id}.png"), show_shapes=True)
    val_generator = return_validation_generator(auto_encoder=hparams[HP_AUTO_ENCODE], batch=1, eval_output=True,
                                                validation=True, out_channel=hparams[HP_OUTPUT_NUMBER])
    iterator = iter(val_generator.data_set)
    predictions = []
    true_values = []
    combined = []
    for _ in range(len(val_generator)):
        print(_)
        x, y = next(iterator)
        pred = model.predict(x[0])
        out_path = os.path.join(save_data_base_path, "Data", "Predictions", f"{session_id}")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        sitk.WriteImage(sitk.GetImageFromArray(np.squeeze(x)), os.path.join(out_path, f"Image_{_}.nii.gz"))
        sitk.WriteImage(sitk.GetImageFromArray(pred[0, ..., 1]), os.path.join(out_path, f"Pred_{_}.nii.gz"))
        sitk.WriteImage(sitk.GetImageFromArray(y[0, ..., 1]), os.path.join(out_path, f"Truth_{_}.nii.gz"))
        continue
        file_name = x[1]
        index = str(file_name.numpy()[0]).split("'")[1].split('_')[0]
        print(index)
        x = x[0]
        y = y[0]

        out_path = os.path.join(save_data_base_path, "Data", "Predictions", f"{session_id}")
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if hparams[HP_AUTO_ENCODE]:
            for __ in range(x.shape[-1]):
                sitk.WriteImage(sitk.GetImageFromArray(y[..., __]), os.path.join(out_path, f"Truth_{__}.nii.gz"))
                sitk.WriteImage(sitk.GetImageFromArray(pred[..., __]), os.path.join(out_path, f"Pred_{__}.nii.gz"))
            return
        else:
            x = np.squeeze(x.numpy())
            y = np.squeeze(y.numpy()[..., 1].astype('float32'))
            pred = np.squeeze(pred[..., -1])
            image_handle = sitk.GetImageFromArray(x)
            image_handle.SetSpacing((1.25, 1.25, 3.0))
            truth_handle = sitk.GetImageFromArray(y)
            truth_handle.SetSpacing((1.25, 1.25, 3.0))
            pred_handle = sitk.GetImageFromArray(pred)
            pred_handle.SetSpacing((1.25, 1.25, 3.0))
            sitk.WriteImage(image_handle, os.path.join(out_path, f"{index}_Image.nii"))
            sitk.WriteImage(truth_handle, os.path.join(out_path, f"{index}_Truth.nii"))
            sitk.WriteImage(pred_handle, os.path.join(out_path, f"{index}_Pred.nii"))
    return predictions, true_values, combined


def run():
    attempts = 0
    while True:
        gc.collect()
        excel_file = os.path.join(save_data_base_path, "HyperparametersNew.xlsx")
        df = pd.read_excel(excel_file)
        indexes = list(df.index)
        # random.shuffle(indexes)
        for ind in indexes:
            session_num = str(int(df['session_number'][ind]))
            if int(session_num) < 80: # Do not change this or we will go back to running 0-40
                continue
            local_tb_path = os.path.join(tb_path, f"{session_num}")
            if os.path.exists(local_tb_path):
                continue
            hparams = return_hparam(df, ind)
            model = return_model(hparams)
            _ = model(tf.random.normal((1, 64, 256, 256, 1)))
            # tf.keras.utils.plot_model(model, to_file=os.path.join(save_data_base_path, f"Model_{session_num}.png"),
            #                 show_shapes=True)
            # continue
            batch = hparams[HP_BATCH]
            auto_encode = min([hparams[HP_AUTO_ENCODE], hparams[HP_RUN_DENSE]])
            folder_headers = hparams['folder_path_headers']
            image_reduction = hparams[HP_IMAGE_REDUCTION]
            train_generator = return_train_generator(batch, folder_strings=folder_headers,
                                                     auto_encoder=auto_encode, image_reduction=image_reduction,
                                                     out_channel=hparams[HP_OUTPUT_NUMBER])
            # x, y = next(iter(train_generator.data_set))
            # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
            # loss = tf.keras.losses.MeanSquaredError()
            # metrics = [tf.keras.metrics.MeanSquaredError(), ]
            # model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            # model.train_on_batch(x,y)
            # pred = model(x)
            # k = reconstruct_volume_from_patches(pred, original_shape=(1, 64, 256, 320, 1), patch_size=(32, 64, 64))
            validation_generator = return_validation_generator(1, folder_strings=folder_headers,
                                                               auto_encoder=auto_encode, validation=True,
                                                               out_channel=hparams[HP_OUTPUT_NUMBER])
            # val_iterator = iter(validation_generator.data_set)
            # for _ in range(3):
            #     print(_)
            #     for __ in range(len(validation_generator)):
            #         print(__)
            #         x, y = next(val_iterator)
            print(f"Running {df['session_number'][ind]}")
            # next(iter(train_generator.data_set))
            # next(iter(validation_generator.data_set))
            train_model(local_tb_path, session_num, hparams, train_generator, validation_generator, model)
            gc.collect()
            time.sleep(60)  # Release memory
            return
        print("Reached the end, sleeping for 10 seconds")
        time.sleep(10)
        attempts += 1
        if attempts > 10:
            return


def save_model(model_number: str):
    model = load_model(model_number, by_name=False)
    out_path = os.path.join(save_data_base_path, 'Data', 'Test', 'Models', "Model_" + str(model_number))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    model.save(os.path.join(out_path, 'model.keras'))
    tf.keras.models.load_model(os.path.join(out_path, 'model.keras'))


if __name__ == "__main__":
    run()
    # evaluate_model('79')
