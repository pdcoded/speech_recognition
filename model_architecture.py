import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, MultiRNNCell, GRUCell, BasicRNNCell

def variable_on_worker_level(name, shape, initializer):
    r'''
    Next we concern ourselves with graph creation.
    However, before we do so we must introduce a utility function ``variable_on_worker_level()``
    used to create a variable in CPU memory.
    '''
    # Use the /cpu:0 device on worker_device for scoped operations
    if not cluster:
        device = worker_device
    else:
        device = tf.train.replica_device_setter(worker_device=worker_device, cluster=cluster_spec)
    with tf.device(device):
        # Create or get apropos variable
        var = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return var
    

def model(encoder_inputs_,source_seq_length,decoder_inputs,decoder_lengths):
    with tf.contrib.slim.arg_scope([tf.contrib.slim.model_variable, tf.contrib.slim.variable], device="/cpu:0"):

        with tf.variable_scope('encoder_1') as scope:
          # Build RNN cell
          encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(encoder_hidden_units)
          # Run Dynamic RNN
          #   encoder_outpus: [max_time, batch_size, num_units]
          #   encoder_state: [batch_size, num_units]
          encoder_outputs,encoder_state= tf.nn.dynamic_rnn(encoder_cell,inputs=encoder_inputs_,sequence_length=source_seq_length,time_major=True,dtype=tf.float64)
        with tf.variable_scope('decoder_1') as scope:

          # attention_states: [batch_size, max_time, num_units]
          attention_states = tf.transpose(encoder_outputs, [1, 0, 2])

          decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(decoder_hidden_units)

          # Create an attention mechanism
          attention_mechanism = tf.contrib.seq2seq.LuongAttention(encoder_hidden_units,attention_states,memory_sequence_length=source_seq_length)
          
          attention_cell = tf.contrib.seq2seq.AttentionWrapper(cell=encoder_cell,attention_mechanism=attention_mechanism)

          attention_zero = attention_cell.zero_state(batch_size=tf.shape(attention_states)[0], dtype=tf.float64)

          decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,attention_layer_size=encoder_hidden_units)
          # Decoder
          decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell,helper=helper,initial_state=attention_zero.clone(cell_state=encoder_state),output_layer=layers_core.Dense(vocab_size, use_bias=False))
          # Dynamic decoding
          decoder_outputs, _,_ = tf.contrib.seq2seq.dynamic_decode(decoder,output_time_major=True)

          logits = decoder_outputs.rnn_output
    return (logits,decoder_outputs.sample_id)
        #Without the projection layer , logits shape would be [Time_Steps,batch_Size,hidden units]
        #After dense projection, logits shape transforms to [ time_steps,batch_size,vocab_size]