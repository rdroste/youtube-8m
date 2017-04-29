import tensorflow as tf

from subprocess import check_output
check_output(["ls", "../input"]).decode("utf8")

video_filename = '../input/video_level/video_level.tfrecord'
video_level_writer = tf.python_io.TFRecordWriter(video_filename)

frame_filename = '../input/frame_level/frame_level.tfrecord'
frame_level_writer = tf.python_io.TFRecordWriter(frame_filename)

nInput_Files = 10

# Max label ID for the examples.
nLabels = 50
for i in range(nInput_Files):
    video_lvl_record = "../input/video_level/train-{}.tfrecord".format(i)
    frame_lvl_record = "../input/frame_level/train-{}.tfrecord".format(i)

    print('Processing', video_lvl_record)

    for video_level, frame_level in zip(tf.python_io.tf_record_iterator(video_lvl_record),
                                        tf.python_io.tf_record_iterator(frame_lvl_record)) :

        tf_video = tf.train.Example.FromString(video_level)
        video_id = tf_video.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8')
        label_list = list(tf_video.features.feature['labels'].int64_list.value)

        if all(label < nLabels for label in label_list):
            video_level_writer.write(tf_video.SerializeToString())

            tf_frame = tf.train.SequenceExample.FromString(frame_level)
            frame_level_writer.write(tf_frame.SerializeToString())
            frame_id = tf_frame.context.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8')
            if video_id != frame_id:
                raise ValueError('Video ids in lists do not match')