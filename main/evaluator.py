from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sacrebleu
import json
def evaluate(type, golds, preds):
    print(len(golds), len(preds))
    # assert len(golds) == len(preds)
    if type == "acc":
        correct = 0.0
        for _ in range(len(golds)):
            gold = golds[_]
            gold_str = " ".join(gold).strip()

            pred = preds[_]
            pred_str = " ".join(pred).strip()

            if gold_str == pred_str:
                correct += 1.0
        return correct/len(preds)
    elif type == "bleu":
        sf = SmoothingFunction()
        maps = []
        with open('../data/no_cycle/dev.data', 'r') as f:
            for line in f:
                js = json.loads(line)
                # maps.append(js['map'])
        score = 0.0
        refs = []
        cands = []
        for _ in range(len(golds)):
            # gold = [golds[_]]
            gold = golds[_]
            # pred = preds[_]
            pred = preds[_]
            # for k, v in maps[_].items():
            #     gold = gold.replace(k, v)
            #     pred = pred.replace(k, v)
            gold = gold.split()
            pred = pred.split() 
            # refs.append(' '.join(gold))
            # cands.append(pred)
            score += sentence_bleu([gold], pred, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf.method2)
            # print([gold])
            # print(pred)
            # print('score: ', score)
            # print('gold: ' + ' '.join(gold).replace(u'\xa0', ' ').replace('\n', ''))
            # print('pred: ' + ' '.join(pred).replace(u'\xa0', ' ').replace('\n', ''))        
        return score/len(preds)
        # refs = [refs]
        # return sacrebleu.corpus_bleu(cands, refs, force=True, lowercase=True, tokenize='none').score
