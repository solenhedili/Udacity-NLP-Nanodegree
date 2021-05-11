from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input,
                          TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, MaxPooling1D)


def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation, return_sequences=True, implementation=2, name='rnn')(input_data)

    bn_rnn = BatchNormalization(name="batch_norm")(simp_rnn)

    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
                  conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu', return_sequences=True, implementation=2, name='rnn')(bn_cnn)

    bn_rnn = BatchNormalization(name="batch_norm")(simp_rnn)

    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model


def cnn_output_length(input_length, filter_size, border_mode, stride,
                      dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # Needed for the loop
    bn_rnn = input_data
    # Iterate over number of desired rnn layers
    for i in range(recur_layers):
        # Name each layer
        layer_name = "rnn" + str(i)
        simp_rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, name=layer_name)(bn_rnn)
        # Add batch normalization after each rnn layer
        batch_norm_name = "batch_norm" + str(i)
        bn_rnn = BatchNormalization(name=batch_norm_name)(simp_rnn)

    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    bidir_rnn = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2), name='bidir_rnn')(
        input_data)

    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def final_model(input_dim, filters, kernel_size, conv_stride,
                conv_border_mode, units, recur_layers, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # Add a convolutional layer
    conv_1d = Conv1D(filters, kernel_size,
                     strides=conv_stride,
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    # bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add Max Pooling
    max_pooling = MaxPooling1D(pool_size=kernel_size, strides=1, padding='same', name='mp_1d')(conv_1d)
    # Add batch normalization
    bn_cnn2 = BatchNormalization(name='bn_conv_2')(max_pooling)

    bidir_rnn = bn_cnn2
    # Iterate over number of desired rnn layers
    for i in range(recur_layers):
        # Name each layer
        layer_name = "rnn" + str(i)
        simp_rnn = Bidirectional(GRU(units, activation='relu', return_sequences=True, implementation=2,
                                     dropout=0.2), name=layer_name)(bidir_rnn)
        # Add batch normalization after each rnn layer
        batch_norm_name = "batch_norm" + str(i)
        bidir_rnn = BatchNormalization(name=batch_norm_name)(simp_rnn)

    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)

    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)

    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model
