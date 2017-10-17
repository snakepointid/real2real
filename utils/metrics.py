import numpy as np
def regression_model_eval(ground_truth,predict,flag='train',verbose=True):
	ground_truth_resort = ground_truth[(-ground_truth).argsort()]
    	predict_flag_resort = ground_truth[(-predict).argsort()]
	pearson_corr,info = diff_eval(ground_truth_resort,predict_flag_resort)
	diff_info = "%s\t%s"%(flag,info)
	truth_bin_info = "%s\tgroud truth:\t%s"%(flag,bin_eval(ground_truth_resort))
	predict_bin_info = "%s\tpredict:\t%s"%(flag,bin_eval(predict_flag_resort))
	if verbose:
		print diff_info
		print truth_bin_info
		print predict_bin_info
	return pearson_corr

def bin_eval(logits,bins=10):
        batch = logits.shape[0]/bins
	bin_info = ""
        for i in range(bins):
                bin_info+="\t%s"%round(np.mean(logits[i*batch:(i+1)*batch]),3)
	return bin_info
def diff_eval(ground_truth_resort,predict_flag_resort,epsilon=0.01):
	pearson_corr = np.corrcoef(ground_truth_resort,predict_flag_resort)[0][1]
	abs_diff = np.mean(np.abs(ground_truth_resort-predict_flag_resort))
	relative_diff = np.mean(np.abs((ground_truth_resort-predict_flag_resort)/(ground_truth_resort+epsilon)))
	info = "eval_num:%s\tpearson_corr:%s\tabs_diff:%s\trelative_diff:%s"%(len(ground_truth_resort),pearson_corr,abs_diff,relative_diff)
	return pearson_corr,info
