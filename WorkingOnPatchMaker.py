import tensorflow as tf
import numpy as np


def return_gradient_image(depth=64, height=128, width=128, channels=1):
    # Create 1D arrays for each dimension ranging from 0 to 1.
    d = np.linspace(0, 1, depth)
    h = np.linspace(0, 1, height)
    w = np.linspace(0, 1, width)
    c = np.linspace(0, 1, channels)

    # Create a 3D meshgrid. 'ij' indexing ensures that the output has shape (depth, height, width).
    D, H, W, C = np.meshgrid(d, h, w, c, indexing='ij')

    # Combine the coordinates to form a gradient.
    # For example, you can average them to get a gradient that smoothly changes in all directions.
    gradient = (D + H + W + C) / 4.0

    # Add batch and channel dimensions to get shape [1, 64, 128, 128, 1]
    gradient_5d = gradient[None, ...]
    return gradient_5d


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
        batch, D, H, W, C = orig_shape
        pD, pH, pW = self.pD, self.pH, self.pW

        # total pad needed so each dim % patch == 0
        pad_d = (pD - D % pD) % pD
        pad_h = (pH - H % pH) % pH
        pad_w = (pW - W % pW) % pW

        # how much went on the “front” of each axis
        pad_d_front = (pad_d + 1) // 2
        pad_h_front = (pad_h + 1) // 2
        pad_w_front = (pad_w + 1) // 2

        # intermediate padded dims
        Dp = D + pad_d
        Hp = H + pad_h
        Wp = W + pad_w

        # after tf.slice we end up back at the original volume size:
        return (batch, D, H, W, C)


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


gradient_image = return_gradient_image(depth=32, height=64, width=64, channels=2)
reduced_image = gradient_image[:, :25, :59, :55]
patch_size = (16, 32, 32)
extracted_x = ExtractVolumePatchesLayer(patch_size)(reduced_image)
extracted_x.shape
x = ReconstructVolumePatchesLayer(patch_size)([extracted_x, reduced_image])
x.shape
x = 1