def cut_sentences():
    f = open('data/text.txt', 'r', encoding='utf-8')
    g = open('data/src_data/src_data.txt', 'a+', encoding='utf-8')
    line = f.read()
    lines = line.split('。')
    for sentence in lines:
        g.write(sentence + '。' + '\n')


if __name__ == '__main__':
    cut_sentences()
