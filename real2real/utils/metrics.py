import numpy as np
def regression_model_eval(ground_truth,predict,flag='train'):
	ground_truth_resort = ground_truth[(-ground_truth).argsort()]
    predict_flag_resort = ground_truth[(-predict).argsort()]
	diff_info = "%s\t%s"%(flag,diff_eval(ground_truth_resort,predict_flag_resort))
	print diff_info
	truth_bin_info = "%s\tgroud truth:\t%s"%(flag,bin_eval(ground_truth_resort))
	print truth_bin_info
	predict_bin_info = "%s\tpredict:\t%s"%(flag,bin_eval(predict_flag_resort))
	print predict_bin_info

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
	info = "pearson_corr:%s\tabs_diff:%s\trelative_diff:%s"%(pearson_corr,abs_diff,relative_diff)
	return info
