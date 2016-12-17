# data.py
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy


# lowest-level stream
class textreader:

    def __init__(self, name, shuffle=False, readall=False):
        if not isinstance(name, (list, tuple)):
            name = [name]

        stream = [open(item, "r") for item in name]

        if shuffle or readall:
            texts = [fd.readlines() for fd in stream]
        else:
            texts = None

        if shuffle:
            readall = True

            if not isinstance(shuffle, bool):
                randstate = numpy.random.RandomState(shuffle)
                shuffle = randstate.shuffle
            else:
                shuffle = numpy.random.shuffle

            linecnt = min([len(text) for text in texts])
            indices = numpy.arange(linecnt)
            shuffle(indices)
        else:
            indices = None
            shuffle = False

        self.eos = False
        self.count = 0
        self.names = name
        self.texts = texts
        self.stream = stream
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def readline(self):
        # read directly from memory
        if self.texts:
            linecnt = min([len(text) for text in self.texts])
            # end of file
            if self.count == linecnt:
                return None

            if self.shuffle:
                texts = [text[self.indices[self.count]] for text in self.texts]
            else:
                texts = [text[self.count] for text in self.texts]
        else:
            # read from file
            texts = [fd.readline() for fd in self.stream]
            flag = any([line == "" for line in texts])

            if flag:
                return None

        self.count += 1
        texts = [text.strip() for text in texts]

        return texts

    def next(self):
        data = self.readline()

        if data == None:
            self.reset()
            raise StopIteration

        return data

    def reset(self):
        self.count = 0
        self.eos = False

        for fd in self.stream:
            fd.seek(0)

        if self.shuffle:
            linecnt = min([len(text) for text in self.texts])
            indices = numpy.arange(linecnt)
            self.shuffle(indices)
            self.indices = indices

    def close(self):
        for fd in self.stream:
            fd.close()

    def get_indices(self):
        return self.indices

    def set_indices(self, indices):
        self.indices = indices


class textiterator:

    def __init__(self, reader, size, processor=None, maxlen=None, sort=False):
        if not isinstance(size, (list, tuple)) or len(size) != 2:
            raise ValueError("size must be format (batch_size, buffer_size)")

        if size[0] > size[1]:
            raise ValueError("buffer_size must >= batch_size")

        if processor and not isinstance(processor, (list, tuple)):
            processor = [processor]

        if not processor and (maxlen or sort):
            raise ValueError("length processor must provided")

        if processor and len(processor) != len(reader.stream):
            raise ValueError("must provide processor for each stream")

        if maxlen and not isinstance(maxlen, (list, tuple)):
            maxlen = [maxlen for i in range(len(reader.stream))]

        if maxlen and len(maxlen) != len(reader.stream):
            raise ValueError("len(maxlen) != len(reader.stream)")

        data = [[] for i in range(len(reader.stream))]

        self.end = False
        self.data = data
        self.size = size
        self.sort = sort
        self.limit = maxlen
        self.reader = reader
        self.processor = processor

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def read_data(self):
        data_size = len(self.data[0])
        batch_size = self.size[0]
        buffer_size = self.size[1]

        # fill buffer
        if batch_size > data_size:
            count = buffer_size - data_size

            while count:
                new_data = self.reader.readline()

                # end of file
                if not new_data:
                    break

                if self.limit and self.processor:
                    ndata = len(new_data)
                    exceed_lim = False

                    for i in range(ndata):
                        if not self.limit[i]:
                            continue

                        if not self.processor[i](new_data[i]):
                            exceed_lim = True
                            break

                        if self.processor[i](new_data[i]) > self.limit[i]:
                            exceed_lim = True
                            break

                    if exceed_lim:
                        continue

                # add to buffer
                for bdata, data in zip(self.data, new_data):
                    bdata.append(data)

                count -= 1

            # sort batch data
            if self.sort:
                lens = []

                for getlen, data in zip(self.processor, self.data):
                    lens.append(map(getlen, data))

                lens = numpy.asarray(lens)
                order = numpy.argsort(lens.max(axis=0))
                newdata = []

                for data in self.data:
                    newdata.append([data[ind] for ind in order])

                self.data = newdata

        new_data_size = len(self.data[0])

        if new_data_size == 0:
            return None
        elif batch_size > new_data_size:
            data = self.data
            self.data = [[] for i in range(len(self.reader.stream))]
            return data
        else:
            data = [item[:batch_size] for item in self.data]
            self.data = [item[batch_size:] for item in self.data]
            return data

    def next(self):
        data = self.read_data()

        if data == None:
            self.reset()
            raise StopIteration

        return data

    def reset(self):
        self.reader.reset()

    def close(self):
        self.reader.close()
