# http://blog.csdn.net/muyiyushan/article/details/64124953
# https://www.cnblogs.com/afangxin/p/6933649.html

# 1。按上面，先安装好tensorflow
# 2.安装Inception-V3模型到任意目录中，并解压出来：

# mkdir /mnt/tensorflow/model
# cd /mnt/tensorflow/model
# wget http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# tar -zxvf inception-2015-12-05.tgz

# 3.把要识别的jpg放到一个目录下面：

# 例如：/mnt/tensorflow/test-images/1.jpg

#  4.执行

# cd /usr/lib/python2.7/site-packages/tensorflow/models/image/imagenet

python classify_image.py --model_dir=/home/liuxz/work/dlws/models/ --image_file=/home/liuxz/work/dlws/models/imgs/girl.jpg 


#python classify_image.py --model_dir=/home/liuxz/work/dlws/models/inception_dec_2015 –-image_file=/home/liuxz/work/dlws/models/imgs/girl.jpg
