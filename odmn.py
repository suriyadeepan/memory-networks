import tensorflow as tf
import numpy as np


class ODMN:

    def __init__(self, 
            batch_size,
            sent_len,
            num_facts,
            ques_len, 
            vocab_size,
            state_size,
            num_episodes):

        # aliases
        F, L, Q  = num_facts, sent_len, ques_len
        N, d = batch_size, state_size

        def __graph__():
            #
            # you know what it does
            tf.reset_default_graph()
            # placeholders
            #  notice the shape [batch_size, num_facts, sent_len]
            inputs_ = tf.placeholder(shape=[N,F,L], dtype=tf.int32)
            questions_ = tf.placeholder(shape=[N,Q], dtype=tf.int32)
            answers_ = tf.placeholder(shape=[N,vocab_size], dtype=tf.int32)
            #
            # embeddings
            embs = tf.get_variable('emb', [vocab_size, d])
            # [batch_size, num_facts, sent_len]
            #   -> [batch_size, num_facts, sent_len, state_size] 
            rnn_inputs = tf.nn.embedding_lookup(embs, inputs_)
            # [batch_size, num_facts, sent_len, state_size] 
            #   -> [sent_len, batch_size, num_facts, state_size] 
            rnn_inputs = tf.transpose(rnn_inputs, [1,2,0,3])
            rnn_q_inputs = tf.nn.embedding_lookup(embs, questions_)

            # setup a gru cell of state_size,
            #   for reuse in graph
            gru = tf.contrib.rnn.GRUCell(d)
           
            # encode inputs_ to 'F' facts
            facts = []
            with tf.variable_scope('input') as scope:
                for i in range(F):
                    # encode each fact in inputs_
                    istates, _ = tf.nn.dynamic_rnn(gru, dtype=tf.float32, 
                                             inputs = rnn_inputs[i])
                    # add final state to facts (a list)
                    facts.append(istates[-1])
                    # reuse gru
                    scope.reuse_variables()
                    
                scope.reuse_variables()
                # encode questions_ to q of shape [batch_size, state_size]
                qstates, _ = tf.nn.dynamic_rnn(gru, dtype=tf.float32, 
                                             inputs = rnn_q_inputs)
                # final state is the encoding for questions (q)
                q = tf.transpose(qstates, [1,0,2])[-1]

            ####
            # params of episodic memory module
            w1 = tf.get_variable('w1', shape=[d, 7*d], dtype=tf.float32, 
                                 initializer= tf.contrib.layers.xavier_initializer())
            b1 = tf.get_variable('b1', shape=[N], dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.))
            w2 = tf.get_variable('w2', shape=[1, d], dtype=tf.float32, 
                                 initializer= tf.contrib.layers.xavier_initializer())
            b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.))

            # episode of episodic memory module
            def episode(q, facts, memory):
                # attention
                def attention(c,m,q):
                    c,m,q = tf.transpose(c),tf.transpose(m),tf.transpose(q)
                    # concat the weird combinations of c,m,q
                    #   into a giant freaky vector
                    vec = tf.concat([c, m, q, c*q, c*m, (c-q)**2, (c-m)**2], axis=0)
                    #
                    # a feed-forward NN to learn attention over each fact
                    h1 = tf.tanh(tf.matmul(w1,vec) + b1)
                    g = tf.tanh(tf.matmul(w2,h1) + b2)
                    # shape of g becomes [batch_size,1]
                    return tf.transpose(g)
                
                # transposed list of facts
                facts_transposed = [tf.transpose(c) for c in facts]
                # state = memory; will be updated
                #   as we iterate through facts
                state = tf.zeros_like(facts[0])
                #
                # gru cell of shape [state_size]
                gru_f = tf.contrib.rnn.GRUCell(d)
                #
                # iterate through facts
                for ct, c in zip(facts, facts_transposed):
                    # gather attention value(s) for each fact
                    #   note : memory and q remain the same for each episode
                    g = attention(ct,memory,q)
                    # update state
                    state = g*gru_f(facts[0], state)[0] + (1 - g) * state
                    tf.get_variable_scope().reuse_variables()
                
                # final state is returned as the new memory vector
                return state

            with tf.variable_scope('episode') as scope:
                # initial memory vector
                memory = tf.identity(q)
                # 
                # run multiple passes over memory
                #  passing facts, question and updated memory
                #   for every pass
                for i in range(num_episodes):
                    # run an episode and
                    #  get a new memory vector
                    ei = episode(q, facts, memory)
                    # apply gru
                    memory = gru(ei, memory)[0]
                    scope.reuse_variables()

            # output transformation
            w_a = tf.get_variable(name='w_a', shape=[d,vocab_size], dtype=tf.float32,
                                 initializer=tf.contrib.layers.xavier_initializer())
            ##
            # get logits
            logits = tf.matmul(memory, w_a)
            # predictions/probabilities
            preds = tf.nn.softmax(logits)
            # optimization
            #losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=answers_)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=answers_)
            loss = tf.reduce_mean(losses)
            train_op = tf.train.AdadeltaOptimizer(0.01).minimize(loss)        

            # attach symbols to objects
            self.train_op = train_op
            self.loss = loss
            self.preds = preds
            self.inputs_ = inputs_
            self.questions_ = questions_
            self.answers_ = answers_
            
        ###
        # build graph
        __graph__()

    ####
    # training method
    def train(self, data_, epochs=1000):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        #
        # prepare batches of data
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
                
                _, train_loss = sess.run([self.train_op, self.loss], feed_dict = {
                        self.inputs_ : s,
                        self.questions_ : q,
                        self.answers_ : a
                    })
                mean_train_loss += train_loss
                
                # every 'm' epochs
                if i%100==0 and i:
                    print('epoch : ', i, ', train loss : ', mean_train_loss/100)
                    mean_train_loss = 0

            except KeyboardInterrupt: # this will most definitely happen, so handle it
                print('Interrupted by user at iteration {}'.format(i))
                return

        print('\n>> Training complete\n')


        

if __name__ == '__main__':

    batch_size = 32
    # fetch data
    data_, metadata = data.fetch(task_id=1, batch_size=batch_size)

    # parameters
    sent_len = metadata['sentence_size']
    num_facts = metadata['memory_size']
    ques_len = sent_len
    vocab_size = metadata['vocab_size']
    state_size = 20
    num_episodes = 3

    model = ODMN(batch_size= batch_size,
            sent_len= metadata['sentence_size'],
            num_facts= metadata['memory_size'],
            ques_len= metadata['sentence_size'],
            vocab_size= metadata['vocab_size'],
            state_size= 20,
            num_episodes= 3)
