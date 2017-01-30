import tensorflow as tf
import numpy as np


class MemN2N(object):

    def __init__(self, 
            embedding_size, # 20
            vocab_size,
            memory_size, # 50
            sentence_size,
            hops = 3,
            max_grad_norm=40.):

        # random seed
        tf.set_random_seed(None)

        # graph building method
        def __graph__():

            tf.reset_default_graph()
            # initializer
            _init = tf.random_normal_initializer(stddev=0.1)
            # encoding
            encoding = tf.constant(position_encoding(sentence_size, embedding_size))
            # graph
            nil_word_slot = tf.zeros([1, embedding_size])
            A = tf.concat(0, [ nil_word_slot, _init([vocab_size-1, embedding_size]) ])
            B = tf.concat(0, [ nil_word_slot, _init([vocab_size-1, embedding_size]) ])
            A = tf.Variable(A, name='A')
            B = tf.Variable(B, name='B')
            # add A,B to nil_vars; 0's shouldn't affect gradient
            nil_vars = [A.name] + [B.name]
            TA = tf.Variable(_init([memory_size, embedding_size ]))
            H = tf.Variable(_init([embedding_size, embedding_size ]))
            W = tf.Variable(_init([embedding_size, vocab_size ]))

            # placeholders
            queries = tf.placeholder(shape=[None, sentence_size], name='queries', dtype=tf.int32)
            stories = tf.placeholder(shape=[None, memory_size, sentence_size], name='stories', dtype=tf.int32)
            answers = tf.placeholder(shape=[None, vocab_size], name='answers', dtype=tf.int32)

            # inference
            q_emb = tf.nn.embedding_lookup(B, queries)
            u_0 = tf.reduce_sum(q_emb * encoding, 1)
            u = [u_0]

            for _ in range(hops):
                # inside MEMORY LOOP
                m_emb = tf.nn.embedding_lookup(A, stories)
                m = tf.reduce_sum(m_emb * encoding, 2) + TA

                # reduce dot
                u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m*u_temp, 2)
                probs = tf.nn.softmax(dotted)

                # calc output memory representation
                probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                c_temp = tf.transpose(m, [0, 2, 1])
                o = tf.reduce_sum(c_temp * probs_temp, 2)

                # calc op
                u_k = tf.nn.relu(tf.matmul(u[-1], H) + o)
                # add new internal state to list
                u.append(u_k)
                
            logits = tf.matmul(u[-1], W)

            # loss
            losses = tf.nn.softmax_cross_entropy_with_logits(logits, tf.cast(answers, tf.float32), name="cross_entropy")
            #losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, answers)
            loss = tf.reduce_mean(losses)

            # apply gradient
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            grads_and_vars = optimizer.compute_gradients(loss)
            grads_and_vars = [(tf.clip_by_norm(g, max_grad_norm), v) for g,v in grads_and_vars]

            # mask zeros
            updated_g_v = []
            for g,v in grads_and_vars:
                if v.name in nil_vars:
                    updated_g_v.append((mask_zeros(g), v))
                else:
                    updated_g_v.append((g,v))        


            # train op
            train_op = optimizer.apply_gradients(updated_g_v)

            # prediction
            predict = tf.argmax(logits, axis=1)

            # attach ops, vars to model
            self.train_op = train_op
            self.loss = loss
            self.stories = stories
            self.queries = queries
            self.answers = answers

        # build graph
        __graph__()


    def train(self, data_, epochs):

        if not hasattr(self, 'sess'):
            sess = tf.Session()
            # init session
            sess.run(tf.global_variables_initializer())
            self.sess = sess

        print('\n>> Training started\n')

        # prepare data batches
        batches = data_['batches']
        trainS = data_['trS']
        trainQ = data_['trQ']
        trainA = data_['trA']

        # train loop
        mean_train_loss = 0
        for i in range(epochs):
            try:
                # shuffle batches
                np.random.shuffle(batches)
                total_cost = 0.0
                for start, end in batches:
                    s = trainS[start:end]
                    q = trainQ[start:end]
                    a = trainA[start:end]
                
                _, train_loss = self.sess.run([self.train_op, self.loss], feed_dict = {
                        self.stories : s,
                        self.queries : q,
                        self.answers : a
                    })
                mean_train_loss += train_loss
                
                # every 'm' epochs
                if i%1000==0 and i:
                    print('epoch : ', i, ', train loss : ', mean_train_loss/1000)
                    mean_train_loss = 0

            except KeyboardInterrupt: # this will most definitely happen, so handle it
                print('Interrupted by user at iteration {}'.format(i))
                self.sess = sess # attach session to model
                return

        print('\n>> Training complete\n')

def mask_zeros(t):
    s = tf.shape(t)[-1]
    # create a row filled with zeros of shape [1,s]
    z = tf.zeros(tf.pack([1, s]))
    # slice away 1st row of t
    #  replace it with z
    return tf.concat(0, [z, tf.slice(t, [1,0], [-1,-1])]) 
    # concat along 0th axis
    # [1,0] : start from 2nd row; [-1, -1] : keep everything


def position_encoding(sentence_size, embedding_size):

    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    ls = sentence_size+1
    le = embedding_size+1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)


