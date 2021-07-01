import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def beta_seg(truth, null_p=[1, 3], pos_p=[3, 1]):
    """
    Bootstrap a model map using a true binary and
    a beta binomial distribution for null and positive response.
        
    Args:
        truth (ndarray): Binary map of observation.
        null_p (list): Parameters of beta distribution for null response.
        pos_p (list): Parameters of beta distribution for positive response.
        
    Returns:
        model (ndarray): Model response map (0-1 float values).
    """
    
    model = truth.copy()
    model[model == 0] = np.random.beta(
        null_p[0], null_p[1],
        size=model[model == 0].shape
    )
    model[model == 1] = np.random.beta(
        pos_p[0], pos_p[1],
        size=model[model == 1].shape
    )

    return model


def tf_smooth(tensor, kernel, size=3):
    """
    Smooth a four-dimensional image tensor (batch, img_x, img_y, channel)
    using a pre-defined or custom kernel.
    
    Args:
        tensor (tf.Tensor): Binary map of observation.
        kernel (str or tf.Tensor): Parameters of beta distribution for null response.
        
    Returns:
        smooth (tf.Tensor): Model response map (0-1 float values).
    """

    if type(kernel) != str:
        pass
    elif kernel == 'box':
        kernel = tf.constant(np.ones((size, size)), dtype=float)
    elif kernel == 'ring':
        kernel = np.ones((size, size)) * 10
        kernel[1, 1] = 1
        kernel = tf.convert_to_tensor(kernel, dtype=float)

    # dummy dimensions
    kernel = tf.reshape(kernel, [kernel.shape[0], kernel.shape[1], 1, 1])

    smooth = tf.reshape(
        tf.nn.conv2d(
            tensor, kernel, strides=[1, 1, 1, 1], 
            padding='SAME'
        ), 
        tensor.shape
    )
    
    smooth = tf.math.divide(smooth, tf.math.reduce_sum(kernel))

    return smooth


def tf_likelihood(model, observed, weights=None):
    """
    Compute log likelihood of segmented image given a probabilistic model.
    
    Args:
        model (tf.Tensor): Pixel-level probability map over time (t, img_x, img_y, channel).
        observed (tf.Tensor): Binary segmented image (img_x, img_y, channel).
        weights (tf.Tensor): Likelihood weights (in t dimension), used for ignoring
            pre/post differences.
            
    Returns:
        logl (float): Log-likelihood of model/observation (more probable > less).
        
    """

    # binarize observation in case it is not already
    observed = tf.math.sigmoid(observed)
    
    # copy it across the whole time series (truth is unchanging)
    observed = tf.tile(observed, tf.constant([model.shape[0], 1, 1, 1]))

    likelihood = tf.math.add_n((
        tf.ones_like(model), -model, -observed,
        tf.math.multiply(2., tf.math.multiply(model, observed))
    ))
    logl = tf.math.log(likelihood)

    if weights is not None:
        logl = tf.math.multiply(logl, weights)

    return tf.math.reduce_sum(logl)


def fit_observation(model, null=True, num_steps=1000,
                    learning_rate=0.001, sharpness=5,
                    return_model=False, bounds_penalty=1):
    """
    Fit a series of segmented pixel probabilites with either a single (null=True) or
    a set of two (null=False) observation truths separated by a floating change point.
    
    Args:
        model (tf.Tensor): Pixel-level probability map over time (t, img_x, img_y, channel).
        null (bool): Set False to allow for a change point.
        num_steps (int): Maximum number of steps in the fit (default 1000).
        learning_rate (float): Initial learning rate (default 0.001).
        sharpness (float): Speed of transition for change point (default 5, higher=faster).
        return_model (bool): Return fitted model.
        bounds_penalty (float): How much to penalize low and high values of t
        
    Returns:
        logl (float): Maximum log-likelihood of best-fitting observation.
        pre (ndarray): Image of pre-change observation. (optional)
        post (ndarray): Image of post-change observation. (optional)
        t (int): Time step of change point. (optional)
    """

    # img is embedded in [time_step, img_x, img_y, channel]
    img_shape = model.shape.as_list()[1:-1]
    time_steps = model.shape.as_list()[0]

    # truth has only one value over time
    obs_shape = [1] + model.shape.as_list()[1:]

    # initialize observations (will be joined in the null case)
    x0 = tf.Variable(tf.random.uniform(obs_shape, minval=-10, maxval=10))
    x1 = tf.Variable(tf.random.uniform(obs_shape, minval=-10, maxval=10))

    if null:
        # sets transition time to 0
        t = tf.constant(-100.)
    else:
        t = tf.Variable(time_steps*0.5)

    # tensor that will activate time steps as a ~binary function (softmax)
    t_weight = tf.constant([sharpness * i * np.ones(img_shape)
                            for i in range(time_steps)],
                           dtype=tf.float32)
    
    # fill in a channel dimension 
    t_weight = tf.reshape(
        t_weight, 
        tf.constant([
            t_weight.shape[0], t_weight.shape[1],
            t_weight.shape[2],1
        ])
    )
    
    
    def loss_fn():
        # change weight function to deactivate likelihood at t
        weight = tf.sigmoid(t_weight - t * sharpness)

        # use the reverse of weights to represent pre-change observation
        logl = -tf_likelihood(model, x0, weights=1. - weight)
        logl -= tf_likelihood(model, tf.math.maximum(x0, x1), weights=weight)
        
        return logl 


    losses = tfp.math.minimize(
        loss_fn,
        num_steps=num_steps,
        optimizer=tf.optimizers.Adam(
            learning_rate=learning_rate, 
            epsilon=1e-7, amsgrad=False
        )
    )

    loss = losses.numpy()[-1]

    pre = tf.math.sigmoid(x0.numpy()[0].sum(axis=2))
    post = tf.math.sigmoid(x1.numpy()[0].sum(axis=2))
    post = tf.math.maximum(pre, post)
    
    if return_model:
        return losses.numpy(), pre, post, t.numpy()    else:
        return loss


def change_ts(model, return_model=False, seed=1234, **kwargs):
    """
    Maximize likelihood for null and change-point models and return test statistic.
    
    Args:
        model (tf.Tensor): Pixel-level probability map over time (t, img_x, img_y, channel).
        return_model (bool): Return fitted models.
        kwargs (dict): Passed to fit_observation.
        
    Returns:
        ts (float): Test statistic indicating likelihood of change.
        null_obs (ndarray): Image of no-change observation. (optional)
        test_pre (ndarray): Image of pre-change observation. (optional)
        test_post (ndarray): Image of post-change observation. (optional)
        t (int): Time step of change point. (optional)
        
    """
    tf.random.set_seed(seed)
    
    null, null_pre, null_post, _ = fit_observation(
        model, null=True, return_model=True, **kwargs
    )
    test, test_pre, test_post, t = fit_observation(
        model, null=False, return_model=True, **kwargs
    )
    
    # Test statistic 
    ts = 2*(test - null)

    null_obs = np.clip(null_pre+null_post, 0, 1)

    if return_model:
        return ts, null_obs, test_pre, test_post, t
    return ts

