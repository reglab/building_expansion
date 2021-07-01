import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def normalize(data, method='local', std=None, mean=None):
    """Normalize imagery, either for each image individually (local), 
    or across all imges (global). 
    
    Args: 
        data (tf.Tensor): images 
        method (str): either 'local' or 'global'
        std (float): std dev for global imagery (will be computed 
            if not provided). 
        mean (float): mean for global imagery (will be computed 
            if not provided)
            
    Returns: 
        
    """
    
    assert method in ["global", "normal"]
    
    if method == 'local':
        std = tf.math.reduce_std(data, axis=[2,3,4])
        std = tf.reshape(std, [std.shape[0], std.shape[1], 1, 1, 1])
        mean = tf.math.reduce_mean(data, axis=[2,3,4])
        mean = tf.reshape(mean, [mean.shape[0], mean.shape[1], 1, 1, 1])
    else:
        if std is None:
            std = tf.math.multiply(tf.ones_like(data), tf.math.reduce_std(data))
        if mean is None:
            mean = tf.math.multiply(tf.ones_like(data), tf.math.reduce_mean(data))
        
    data = tf.math.add(data, -mean)
    data = tf.math.divide_no_nan(data, std)
    
    return data, mean, std
    

def fit_observation(basis, test,
                    normalization='local', reg=1,
                    remove_outliers=True, max_outlier_iterations=5,
                    num_steps=1000, learning_rate=0.01):
    """
    Fit DDM
    
    Args:
        basis (tf.Tensor): Batch of pixel-level probability maps over time 
            (batch_size, t, img_x, img_y, channel).
        test (tf.Tensor): Batch of pixel-level probability maps at test time
            (batch_size, img_x, img_y, channel).
        num_steps (int): Maximum number of steps in the fit (default 1000).
        normalization (str): Type of normalization to perform 
            (local->by image, global->all images).
        reg (float): Regularization (L1) strength.
        learning_rate (float): Initial learning rate (default 0.01).
        
    Returns:
        dictionary with model parameters: 
        gamma (tf.Tensor): learned weights 
        losses (np.ndarray): loss over time 
        rmse (float): final root mean squared error
        std (float): final standard deviation
    """

    # img is embedded in [batch, time_step, img_x, img_y, channel]
    img_shape = basis.shape.as_list()[2:-1]
    time_steps = basis.shape.as_list()[1]
    batch_size = basis.shape.as_list()[0]
    
    # normalize each image to have same mean/std
    if normalization == 'local' or normalization == 'global':
        basis, mean, std = normalize(basis, method=normalization)
        test, mean, std = normalize(test, method=normalization, 
                                    mean=mean, std=std) 
    else:
        mean = None
        std = None
    
    # either this or expand gamma
    test = test[:, 0, :, :, :]

    outliers = np.zeros_like(test)
    iterations = 0
    while iterations < max_outlier_iterations:
        print(f'iterating with {outliers.sum()} outliers present')
        iterations += 1
        gamma = tf.Variable(tf.random.uniform(
            [1, time_steps+1, 1, 1, 1], minval=0, maxval=1 # +1 for constant term 
        )) 

        def loss_fn():
            pred = predict(gamma, basis)
            diff = tf.math.abs(pred - test)
            diff = tf.boolean_mask(diff, 1-outliers)
            diff_sum = tf.math.reduce_sum(diff)
            l1 = tf.reduce_sum(tf.abs(gamma))
            return diff_sum + l1 * reg

        losses = tfp.math.minimize(
            loss_fn,
            num_steps=num_steps,
            optimizer=tf.optimizers.Adam(
                learning_rate=learning_rate,
                epsilon=1e-7, amsgrad=True),
            convergence_criterion=(
                tfp.optimizer.convergence_criteria.LossNotDecreasing(atol=1))
        )
        loss = losses.numpy()[-1]

        pred = predict(gamma, basis)
        m = tf.keras.metrics.RootMeanSquaredError()
        m.update_state(test, pred)
        rmse = m.result().numpy()

        if remove_outliers:
            outliers[np.where((tf.math.abs(pred-test) / rmse > 5))] = 1
            print(f'{outliers.sum()} outliers')
            diff = tf.math.abs(pred - test)
            diff = tf.boolean_mask(diff, 1-outliers)
            z = tf.math.divide_no_nan(diff, rmse)
            print(f'max z {tf.math.reduce_max(z)}')
            if tf.math.reduce_max(z) < 5:
                break
        else:
            break
    
    return {'gamma': gamma,
            'loss': losses.numpy(),
            'rmse': rmse,
            'mean': mean,
            'std': std}


def predict(gamma, basis, normalization='none', mean=None, std=None):
    """Predict next image from basis images. 
    
    Args: 
        gamma (tf.Tensor): fitted DDM params 
        basis (tf.Tensor): basis imagery. same number as length of 
            gamma. 
        normalization (str): image normalization type 
        mean (float): enforced mean for global image normalization 
        std (float): enforced standard dev for image normalization 
        
    Returns: 
        pred (tf.Tensor): predicted image 
          
    
    """ 
    if normalization == 'local' or normalization == 'global':
        basis, mean, std = normalize(basis, method=normalization, mean=mean, std=std)

    pred = tf.reduce_sum(tf.math.multiply(gamma[:,1:,:,:,:], basis), axis=1)
    pred = gamma[:,0,:,:,:] + pred
    return pred


def hot_detect(gamma, basis, test, rmse, normalization='none', mean=None, std=None,
               pos_only=True, reduce=False):
    """Detect anamolies between predicted image and true image. 
    
    Args: 
        gamma (tf.Tensor): fitted DDM params 
        basis (tf.Tensor): basis imagery. same number as length of 
            gamma. 
        test (tf.Tensor): test image(s). 
        rmse (float): rmse from fitted model 
        normalization (str): image normalization type 
        mean (float): enforced mean for global image normalization 
        std (float): enforced standard dev for image normalization 
        pos_only (bool): Return positive anomalies only , i.e., 
            max(0, (true-pred)/rmse). 
        reduce (bool): If true, reduces pixel-wise anomalies to single
            value by computing sum of absolute anomaly values across all 
            pixels. If False, returns pixel-level anomalies (useful for 
            plotting). Default: False
            
    Returns: 
        anamoly score - either pixel by pixel or across entire image
        
   """
    
    # predict 
    pred = predict(gamma, basis, normalization=normalization, mean=mean, std=std)
    
    if normalization == 'global' or normalization == 'local':
        true, mean, std = normalize(test, method=normalization, mean=mean, std=std)
    else:
        true = test
        
    # calculate analomy scores 
    pred = tf.reshape(pred, true.shape)
    hot_map = (true - pred) / rmse
    if pos_only:
        hot_map = tf.clip_by_value(hot_map, clip_value_min=0, clip_value_max=1e6)

    if reduce:
        return tf.math.reduce_sum(abs(hot_map), axis=[1,2,3,4])
    else:
        return hot_map

