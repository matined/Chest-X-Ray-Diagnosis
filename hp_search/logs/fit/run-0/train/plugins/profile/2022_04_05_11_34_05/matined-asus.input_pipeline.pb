	i9?C]Ś@i9?C]Ś@!i9?C]Ś@	???qY,?????qY,??!???qY,??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$i9?C]Ś@????
??Ao?\?@Yl@???9@*	??N9?3A2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::FlatMap[0]::GeneratorX??Cu?@!?N>Ճ?X@)X??Cu?@1?N>Ճ?X@:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?3g}??9@!??Ka????)?3g}??9@1??Ka????:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelisml?<*??9@!OS??V???)???f???1bu?V?E?:Preprocessing2F
Iterator::Modelt#,*??9@!t@m???)?!??l?1?J??k1?:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::FlatMapJ?Gw?@!0??J??X@)??W?`?1???:?#?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 1.5% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???qY,??IT8?N?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????
??????
??!????
??      ??!       "      ??!       *      ??!       2	o?\?@o?\?@!o?\?@:      ??!       B      ??!       J	l@???9@l@???9@!l@???9@R      ??!       Z	l@???9@l@???9@!l@???9@b      ??!       JCPU_ONLYY???qY,??b qT8?N?X@