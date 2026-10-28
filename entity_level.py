import pandas as pd
import csv

FILE = './assertions.csv'

data_en = []
similar_pair = []
t = 0


def get_word_pair():
    with open(FILE, "r", encoding='utf-8') as f:
        dataset = csv.reader(f)

        for row in dataset:
            if row[0] == '/a/[/r/SimilarTo/' and row[1][3:5] == 'en' and len(row) > 7:
                data_en.append(row)
            else:
                continue

        # print()

        for i in range(len(data_en)):
            start = data_en[i][6]
            end = data_en[i][5]

            start_word = start[17:]
            start_word = start_word[1:-1]
            end_word = end[15:]
            end_word = end_word[1:-1]
            if start_word[0].isdigit():
                continue
            else:
                similar_pair.append([start_word, end_word])

    return similar_pair

    # print()


# print(get_word_pair())

# similar_pair1 = get_word_pair()
# a = 0
#
# for i in range(len(similar_pair1)):
#     simi_data_0 = similar_pair1[i][0]
#     if simi_data_0 == 'accept':
#         a = 12
#         print()