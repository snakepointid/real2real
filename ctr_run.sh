cd /home/hdp-reader-tag/shechanglue/models/real2real
HADOOP_HOME=/usr/bin/hadoop/software/hadoop
HBOX_HOME=/usr/bin/hadoop/software/hbox
#get yearterday's date
INPUT_FILE=hdfs://namenodefd1v.qss.zzzc.qihoo.net:9000/home/hdp-reader-tag/user/shechanglue/recallTagTitleCTR/charEncode
MODEL_FILE=hdfs://namenodefd1v.qss.zzzc.qihoo.net:9000/home/hdp-reader-tag/user/shechanglue/recallTagTitleCTR/model/$1
#
$HADOOP_HOME/bin/hadoop fs -test -e $MODEL_FILE
if [ $? -ne 0 ]; then
	echo "$MODEL_FILE not exist"
	$HADOOP_HOME/bin/hadoop fs -mkdir $MODEL_FILE     
fi

$HBOX_HOME/bin/hbox-submit \
   --app-type "tensorflow" \
   --files real2real \
   --worker-memory 20000 \
   --worker-cores 2 \
   --worker-gpus 1 \
   --boardEnable true \
   --launch-cmd "python real2real/app/ctrPredict.py --restore_path=./model --save_path=$MODEL_FILE" \
   --input $INPUT_FILE#data \
   --cacheFile $MODEL_FILE#model \
   --inputformat-enable true \
   --priority VERY_HIGH \
   --conf hbox.task.timeout=1200000\
   --appName "ctrPredict_$1" 

echo "view result~~~"
$HADOOP_HOME/bin/hadoop fs -ls $MODEL_FILE
