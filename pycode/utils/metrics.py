import numpy as np
def ctrEval(ctr,prob,flag='train'):
        model_ctr = ctr[(-prob).argsort()]
        truth_ctr = ctr[(-ctr).argsort()]
        batch = len(model_ctr)/10
        model_pt = "%s\tmodel predict:"%flag
        truth_pt = "%s\tground truths:"%flag
        for i in range(10):
                model_pt+="\t%s"%round(np.mean(model_ctr[i*batch:(i+1)*batch]),3)
                truth_pt+="\t%s"%round(np.mean(truth_ctr[i*batch:(i+1)*batch]),3)
        print(model_pt)
        print(truth_pt)
