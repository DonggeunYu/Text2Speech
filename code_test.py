

path = './datasets/kss' + '/transcript.txt'

with open(path, encoding='utf-8') as f:
    lines = f.readlines()
    wav_path = []
    text = []

    for i in lines:
        sp = i.split('|')
        if sp[1] == sp[2]:
            wav_path.append(sp[0])
            text.append(sp[1])

        else:
            wav_path.append(sp[0])
            text.append(sp[1])
            wav_path.append(sp[0])
            text.append(sp[2])

