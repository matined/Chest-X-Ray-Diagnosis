	8??wh??@8??wh??@!8??wh??@	??c"? @??c"? @!??c"? @"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$8??wh??@m?i?*|??A(~???N?@Yr?߅?\f@*	?n???A2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch%W@![f@!??`?X@)%W@![f@1??`?X@:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismy ?H?[f@!H?vN??X@)?"????1lo??:Preprocessing2F
Iterator::Model|{נ?[f@!      Y@)?[[%Xl?1ް?C??_?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.4% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9??c"? @I?K????V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	m?i?*|??m?i?*|??!m?i?*|??      ??!       "      ??!       *      ??!       2	(~???N?@(~???N?@!(~???N?@:      ??!       B      ??!       J	r?߅?\f@r?߅?\f@!r?߅?\f@R      ??!       Z	r?߅?\f@r?߅?\f@!r?߅?\f@b      ??!       JCPU_ONLYY??c"? @b q?K????V@