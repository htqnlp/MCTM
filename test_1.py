with open('mcnc_data/mcnc_test_final.txt', encoding='utf-8') as file:
    dataset = file.readlines()


def mcnc_data():
    evt_chain1 = {'HOLDOUT': [], 'NEG': [], 'DOC': []}
    evt_chain = []
    j = 0
    i = 0
    # for i in range(len(dataset)):
    while i < len(dataset):
        while dataset[i] != 'HOLDOUT_SEP\n':
            evt_chain1['HOLDOUT'].append(dataset[i])
            i = i + 1
            if dataset[i] == 'HOLDOUT_SEP\n':
                i = i + 2
                break
        # if dataset[i] == '\n':
        #     continue
        while dataset[i] != 'NEG_SEP\n':
            evt_chain1['NEG'].append(dataset[i])
            i = i + 1
            if dataset[i] == 'NEG_SEP\n':
                i = i + 2
                break
        while dataset[i] != 'DOC_SEP\n':
            evt_chain1['DOC'].append(dataset[i])
            i = i + 1
            if dataset[i] == 'DOC_SEP\n':
                i = i + 2
                break
        evt_chain.append(evt_chain1)
        evt_chain1 = {'HOLDOUT': [], 'NEG': [], 'DOC': []}

    return evt_chain
