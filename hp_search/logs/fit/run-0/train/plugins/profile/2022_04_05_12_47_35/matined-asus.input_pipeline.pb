	?tps0?@?tps0?@!?tps0?@	?Z&vd??Z&vd?!?Z&vd?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?tps0?@,???)W??A5??6v/?@YDܜJ???*	_??b?9A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::Generator?B;????@!??"_??X@)?B;????@1??"_??X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?(???ǒ?!?7?	?KR?)?(???ǒ?1?7?	?KR?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?H?<???!??)??Z?)??)????1?s??5A?:Preprocessing2F
Iterator::Model}?q ???!AK?/^?)?????j?1?݆K*?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap???(???@!?????X@)5_%?d?1?;c??#?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?Z&vd?I?????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	,???)W??,???)W??!,???)W??      ??!       "      ??!       *      ??!       2	5??6v/?@5??6v/?@!5??6v/?@:      ??!       B      ??!       J	DܜJ???DܜJ???!DܜJ???R      ??!       Z	DܜJ???DܜJ???!DܜJ???b      ??!       JCPU_ONLYY?Z&vd?b q?????X@