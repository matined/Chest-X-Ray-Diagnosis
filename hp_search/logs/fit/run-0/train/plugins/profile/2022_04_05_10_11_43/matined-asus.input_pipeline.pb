	b??M?@b??M?@!b??M?@	?ܵ?`??ܵ?`?!?ܵ?`?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$b??M?@iUK:????A?.L?@Y%?S;ì?*	&??P<A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?b~n?ɜ@!:????X@)?b~n?ɜ@1:????X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?W zR&??!?KZ?]R?)?W zR&??1?KZ?]R?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismD?|?F??!???)?([?)k???@??1?c??E?A?:Preprocessing2F
Iterator::Model#M?<i??!?&;?#=^?)???o^l?1?H!?(?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapI??r?ɜ@!?=????X@)?b?J!`?1??X??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?ܵ?`?IF?T???X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	iUK:????iUK:????!iUK:????      ??!       "      ??!       *      ??!       2	?.L?@?.L?@!?.L?@:      ??!       B      ??!       J	%?S;ì?%?S;ì?!%?S;ì?R      ??!       Z	%?S;ì?%?S;ì?!%?S;ì?b      ??!       JCPU_ONLYY?ܵ?`?b qF?T???X@