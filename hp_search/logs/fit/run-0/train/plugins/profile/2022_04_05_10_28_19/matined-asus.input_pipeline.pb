	?r???@?r???@!?r???@	@(P?+?W?@(P?+?W?!@(P?+?W?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?r???@?HZ????A=|?(?@Y?&p???*	+?q2?A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator???????@!?????X@)???????@1?????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch\W?o??!?m?`DL?)\W?o??1?m?`DL?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism7¢"N'??!????S?)m?Yg|_|?1??;46?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???u??@!G?^C??X@)V*???e?1??I3!?:Preprocessing2F
Iterator::Model-????ƛ?!??/??U?)??g??d?1????k ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9@(P?+?W?I?i?-??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?HZ?????HZ????!?HZ????      ??!       "      ??!       *      ??!       2	=|?(?@=|?(?@!=|?(?@:      ??!       B      ??!       J	?&p????&p???!?&p???R      ??!       Z	?&p????&p???!?&p???b      ??!       JCPU_ONLYY@(P?+?W?b q?i?-??X@