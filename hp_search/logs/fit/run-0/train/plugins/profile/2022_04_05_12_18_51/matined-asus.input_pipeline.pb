	+j0J?@+j0J?@!+j0J?@	rb?O??g?rb?O??g?!rb?O??g?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$+j0J?@	?v???A???I?@Y(v?U???*	&1???EA2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?g?K?$?@!??-???X@)?g?K?$?@1??-???X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch4J??%??!?*	>\S?)4J??%??1?*	>\S?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism|?q7??!Ml???W?)?e?ikD??1r?G?]2?:Preprocessing2F
Iterator::Model<f?2?}??!????dY?)	??Lnd?1??ڣ??:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???@?$?@!?zN???X@)b?[>??^?1?k?pB?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9rb?O??g?I?`?>??X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
		?v???	?v???!	?v???      ??!       "      ??!       *      ??!       2	???I?@???I?@!???I?@:      ??!       B      ??!       J	(v?U???(v?U???!(v?U???R      ??!       Z	(v?U???(v?U???!(v?U???b      ??!       JCPU_ONLYYrb?O??g?b q?`?>??X@