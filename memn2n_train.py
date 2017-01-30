import memn2n
import bAbI.data as data


if __name__ == '__main__':
    # fetch data
    data_, metadata = data.fetch(task_id=1, batch_size=32)


    # parameters
    embedding_size = 20
    vocab_size = metadata['vocab_size']
    memory_size = metadata['memory_size']
    sentence_size = metadata['sentence_size']
    hops = 3

    # create model
    model = memn2n.MemN2N(embedding_size= embedding_size,
                         vocab_size= vocab_size,
                         memory_size= memory_size,
                         sentence_size= sentence_size,
                         hops= 3)

    # train model
    model.train(data_, epochs=6000)
