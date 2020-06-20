def delete_first_lines(filename, count):
    with open(filename, 'r') as fin:
        a = fin.readlines()
    with open(filename, 'w') as fout:
        b = ''.join(a[count:])
        fout.write(b)


def delete_last_lines(filename, count):
    with open(filename, 'r') as fin:
        a = fin.readlines()
    with open(filename, 'w') as fout:
        b = ''.join(a[:-count])
        fout.write(b)


def cycle_last_lines(filename, count):
    with open(filename, 'r') as fin:
        a = fin.readlines()
    with open(filename, 'w') as fout:
        b = ''.join(a[-count:] + a[:-count])
        fout.write(b)


def cycle_first_lines(filename, count):
    with open(filename, 'r') as fin:
        a = fin.readlines()
    with open(filename, 'w') as fout:
        b = ''.join(a[count:] + a[:count])
        fout.write(b)


def merge(file_read, file_write):
    with open(file_read, 'r') as fin:
        read = fin.readlines()
    with open(file_write, 'a') as fout:
        read = ''.join(read)
        fout.write(read)


def cover(file_read, file_write):
    with open(file_read, 'r') as fin:
        read = fin.readlines()
    with open(file_write, 'w') as fout:
        read = ''.join(read)
        fout.write(read)


if __name__ == '__main__':
    # filename = 'data_train.csv'
    # delete_first_lines(filename, 200000)
    # delete_last_lines(filename, 100000)
    filename1 = 'data_train.csv'
    filename2 = './back_up/data_train_clean.csv'
    cover(file_read=filename2, file_write=filename1)


