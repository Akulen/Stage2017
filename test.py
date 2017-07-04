
def tft():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    import tensorflow as tf

    x = tf.placeholder(tf.float32, [None, 784])

    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(-1 * tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()

    for _ in range(1):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

def slt():
    import numpy as np
    from sklearn.tree import DecisionTreeRegressor
    import matplotlib.pyplot as plt

    # Create a random dataset
    rng = np.random.RandomState(1)
    X = np.sort(5 * rng.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - rng.rand(16))

    # Fit regression model
    regr_1 = DecisionTreeRegressor(max_depth=2)
    regr_2 = DecisionTreeRegressor(max_depth=5)
    regr_1.fit(X, y)
    regr_2.fit(X, y)

    # Predict
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_1 = regr_1.predict(X_test)
    y_2 = regr_2.predict(X_test)

    # Plot the results
    plt.figure()
    plt.scatter(X, y, c="darkorange", label="data")
    plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
    plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
    plt.xlabel("data")
    plt.ylabel("target")
    plt.title("Decision Tree Regression")
    plt.legend()
    plt.show()

def cube_show_slider(cube, axis=2, **kwargs):
    """
    Display a 3d ndarray with a slider to move along the third dimension.

    Extra keyword arguments are passed to imshow
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, RadioButtons

    # check dim
    if not cube.ndim == 3:
        raise ValueError("cube should be an ndarray with ndim == 3")

    # generate figure
    fig = plt.figure()
    ax = plt.subplot(111)
    fig.subplots_adjust(left=0.25, bottom=0.25)

    # select first image
    s = [slice(0, 1) if i == axis else slice(None) for i in range(3)]
    im = cube[s].squeeze()

    # display image
    l = ax.imshow(im, **kwargs)

    # define slider
    axcolor = 'lightgoldenrodyellow'
    ax = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)

    slider = Slider(ax, 'Axis %i index' % axis, 0, cube.shape[axis] - 1,
                    valinit=0, valfmt='%i')

    def update(val):
        ind = int(slider.val)
        s = [slice(ind, ind + 1) if i == axis else slice(None)
                 for i in range(3)]
        im = cube[s].squeeze()
        l.set_data(im, **kwargs)
        fig.canvas.draw()

    slider.on_changed(update)

    plt.show()

def embed():
    import tensorflow as tf
    import os
    from tensorflow.contrib.tensorboard.plugins import projector

    LOG_DIR = "./train/"
    
    # Create randomly initialized embedding weights which will be trained.
    N = 10000 # Number of items (vocab size).
    D = 200 # Dimensionality of the embedding.
    embedding_var = tf.Variable(tf.random_normal([N,D]), name='word_embedding')

    # Format: tensorflow/tensorboard/plugins/projector/projector_config.proto
    config = projector.ProjectorConfig()

    # You can add multiple embeddings. Here we add only one.
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')

    # Use the same LOG_DIR where you stored your checkpoint.
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
    # read this file during startup.
    projector.visualize_embeddings(summary_writer, config)
