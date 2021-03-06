import csv


def cal_sentiment_prf(tp, fp, fn, num_of_aspect, verbal=False, modelName="", fisrtModel=True):
    p = [tp[i] / (tp[i] + fp[i]) for i in range(num_of_aspect)]
    r = [tp[i] / (tp[i] + fn[i]) for i in range(num_of_aspect)]
    f1 = [2 * p[i] * r[i] / (p[i] + r[i]) for i in range(num_of_aspect)]

    micro_p = sum(tp) / (sum(tp) + sum(fp))
    micro_r = sum(tp) / (sum(tp) + sum(fn))
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r)

    macro_p = sum(p) / num_of_aspect
    macro_r = sum(r) / num_of_aspect
    macro_f1 = sum(f1) / num_of_aspect

    if verbal:
        print('p:', p)
        print('r:', r)
        print('f1:', f1)
        print('micro:', (micro_p, micro_r, micro_f1))
        print('macro:', (macro_p, macro_r, macro_f1))
    title = '\tgiá\t\t\t\tdịch_vụ\t\t\t\tan_toàn\t\t\t\tchất_lượng\t\t\tship\t\tchính_hãng\n'
    output = f"'p': {p}\n'r': {r}\n'f1': {f1}\n'micro': ({micro_p}, {micro_r}, {micro_f1})\n'macro': ({macro_p}, {macro_r}, " \
             f"{macro_f1})".format(p, r, f1, micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1)
    outputs = title + output

    p.append(micro_p)
    p.append(macro_p)
    r.append(micro_r)
    r.append(macro_r)
    f1.append(micro_f1)
    f1.append(macro_f1)
    rowp = ['p']
    rowp.extend(p)
    rowr = ['r']
    rowr.extend(r)
    rowf1 = ['f1']
    rowf1.extend(f1)
    with open('score.csv', ('w' if fisrtModel else 'a'), newline='') as scoreFile:
        scoreWriter = csv.writer(scoreFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        scoreWriter.writerow(
            [modelName, 'aspect0', 'aspect1', 'aspect2', 'aspect3', 'aspect4', 'aspect5', 'micro', 'macro'])
        scoreWriter.writerow(rowp)
        scoreWriter.writerow(rowr)
        scoreWriter.writerow(rowf1)
        scoreWriter.writerow([])
    return outputs