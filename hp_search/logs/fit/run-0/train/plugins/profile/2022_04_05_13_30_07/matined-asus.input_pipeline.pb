	????(?@????(?@!????(?@	?7/?6q??7/?6q?!?7/?6q?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$????(?@??.?u???A?????&?@Y?X???R??*	???QhH@2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch????M??!?`??NS@)????M??1?`??NS@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?????!?'??*?W@)??/Ȁ?1{?+?0@:Preprocessing2F
Iterator::ModelX??G???!      Y@)?c#??g?12?m?T?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?7/?6q?I?Cw%??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??.?u?????.?u???!??.?u???      ??!       "      ??!       *      ??!       2	?????&?@?????&?@!?????&?@:      ??!       B      ??!       J	?X???R???X???R??!?X???R??R      ??!       Z	?X???R???X???R??!?X???R??b      ??!       JCPU_ONLYY?7/?6q?b q?Cw%??X@